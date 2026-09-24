"""Articulated tabletop scene: the Panda tabletop plus drawers, doors, lids and flaps.

The fixtures come from the embodiment-independent library pai.envs.fixtures; this module only
places them on the table (one mocap base per fixture, so the pose can be re-sampled every reset
without recompiling) and wires them into observations, predicates and the event log.

Per episode (reset seed): each fixture's position within its configured box, its yaw (facing the
robot base, plus noise), the objects' layout outside the fixtures' swept areas, and a hidden cause
per part (none / locked / stuck / blocked / latched) drawn from cfg.causes or forced via
reset(causes=...). The fixture types (geometry, handles, joint feel) are fixed per env instance:
build another env to test on unseen types.

Observation dict: everything TabletopEnv returns, plus `articulations` (part name -> joint position,
opening 0..1, handle position/orientation, face normal, joint axis/anchor, pull direction, lever
state). Predicates gain open(part), closed(part) and grasped(part). Hidden causes and the type
parameters are ground truth: they appear in `env.info` and in the `fixture_reset` event only.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pai.envs.disturbances import Disturbance
from pai.envs.events import EventLog
from pai.envs.fixtures import (CLOSED_FRAC, OPEN_FRAC, Fixture, FixtureRuntime, FixtureSpec, add_fixture,
                               make_fixture, sample_fixture)
from pai.envs.predicates import ObjectSpec
from pai.envs.tabletop import TabletopEnv, build_tabletop_model

PLACE_TRIES = 300
# Where a fixture's front bottom-centre goes (world xy box) and its yaw noise, unless cfg says otherwise.
# Lids open towards their back hinge, so the box sits closer to the robot.
DEFAULT_PLACEMENT = {"pos_low": [0.56, -0.03], "pos_high": [0.6, 0.03], "yaw_range": 0.12}
FAMILY_PLACEMENT = {"lid": {"pos_low": [0.42, -0.03], "pos_high": [0.46, 0.03]}}


def _yaw_quat(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


class ArticulatedEnv(TabletopEnv):
    def __init__(self, cfg, disturbances: list[Disturbance] | None = None,
                 fixtures: dict[str, FixtureSpec] | None = None):
        """fixtures, if given, replaces the scene's fixtures (name -> spec); placement comes from the
        cfg.fixtures entry of the same name, else DEFAULT_PLACEMENT."""
        self._plan = self._fixture_plan(cfg, fixtures)
        self._forced_causes: dict[str, str] = {}
        self.fixture_pose: dict[str, tuple[np.ndarray, float]] = {}
        super().__init__(cfg, disturbances)

    @staticmethod
    def _fixture_plan(cfg, fixtures: dict[str, FixtureSpec] | None) -> list[tuple[str, FixtureSpec, dict]]:
        entries = {e["name"]: e for e in cfg.get("fixtures", [])}
        rng = np.random.default_rng(int(cfg.get("fixture_seed", 0)))
        plan = []
        if fixtures is not None:
            for name, fx in fixtures.items():
                plan.append((name, fx, {**DEFAULT_PLACEMENT, **FAMILY_PLACEMENT.get(fx.family, {}), **entries.get(name, {})}))
            return plan
        for name, e in entries.items():
            if e.get("type"):
                fx = make_fixture(rng, e["type"])
            else:
                fx = sample_fixture(rng, e.get("family"), e.get("split", "train"), tabletop=True)
            plan.append((name, fx, {**DEFAULT_PLACEMENT, **FAMILY_PLACEMENT.get(fx.family, {}), **e}))
        return plan

    def _build_model(self, cfg) -> tuple[mujoco.MjModel, dict[str, ObjectSpec]]:
        self.fixtures: list[Fixture] = []

        def extra(spec: mujoco.MjSpec) -> None:
            for name, fx, _ in self._plan:
                base = spec.worldbody.add_body(name=f"{name}_mocap", mocap=True)
                self.fixtures.append(add_fixture(spec, base, fx, name))

        model, objects = build_tabletop_model(cfg, extra=extra)
        self.fx = FixtureRuntime(model, self.fixtures)
        self._mocap = {name: int(model.body_mocapid[model.body(f"{name}_mocap").id]) for name, _, _ in self._plan}
        self._fixture_body = self.fx.body_names(model)
        return model, objects

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None, layout: dict[str, np.ndarray] | None = None,
              causes: dict[str, str] | None = None) -> dict:
        """causes optionally forces hidden causes per part (e.g. {"drawer": "locked"}); the others are
        drawn from cfg.causes."""
        if not isinstance(self.events, ArticulatedEventLog):
            self.events = ArticulatedEventLog(self)
        self._forced_causes = dict(causes or {})
        return super().reset(seed, layout)

    def _place_objects(self, layout: dict) -> None:
        """Called by TabletopEnv.reset after mj_resetData: fixtures first, then objects around them."""
        m, d = self.model, self.data
        for name, _, place in self._plan:
            xy = self.rng.uniform(place["pos_low"], place["pos_high"])
            yaw = float(np.arctan2(-xy[1], -xy[0]) + self.rng.uniform(-1, 1) * float(place["yaw_range"]))
            d.mocap_pos[self._mocap[name]] = [xy[0], xy[1], 0.0]
            d.mocap_quat[self._mocap[name]] = _yaw_quat(yaw)
            self.fixture_pose[name] = (xy, yaw)
        causes = self.fx.sample_causes(self.rng, dict(self.cfg.get("causes", {"none": 1.0})))
        unknown = set(self._forced_causes) - set(causes)
        if unknown:
            raise KeyError(f"unknown parts {sorted(unknown)}; parts are {sorted(causes)}")
        causes.update(self._forced_causes)
        self.fx.apply_causes(m, d, causes)
        super()._place_objects(self._object_layout(layout))

    def _in_keepout(self, xy: np.ndarray, margin: float) -> bool:
        for f in self.fixtures:
            (fxy, yaw) = self.fixture_pose[f.name]
            c, s = np.cos(yaw), np.sin(yaw)
            rel = xy - fxy
            local = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
            x0, x1, y0, y1 = f.keepout
            if x0 - margin < local[0] < x1 + margin and y0 - margin < local[1] < y1 + margin:
                return True
        return False

    def _object_layout(self, layout: dict) -> dict:
        """Object xy positions clear of each other and of the fixtures' swept areas."""
        lo, hi = np.array(self.cfg.spawn_low), np.array(self.cfg.spawn_high)
        out, placed = {}, []
        for name, spec in self.objects.items():
            if name in layout:
                xy = np.asarray(layout[name], float)
            else:
                for _ in range(PLACE_TRIES):
                    xy = self.rng.uniform(lo, hi)
                    if not self._in_keepout(xy, spec.radius + 0.01) and all(
                            np.linalg.norm(xy - p) > spec.radius + r + 0.02 for p, r in placed):
                        break
            placed.append((xy, spec.radius))
            out[name] = xy
        return out

    def step(self, action: np.ndarray) -> dict:
        self.fx.update(self.data)  # latch: engage / release the lock constraints
        return super().step(action)

    # ------------------------------------------------------------------ sensing
    def _observe(self) -> dict:
        obs = super()._observe()
        arts = self.fx.observe(self.data)
        obs["articulations"] = arts
        grasped = self.events.grasped_handles() if isinstance(getattr(self, "events", None), ArticulatedEventLog) else set()
        preds = obs["predicates"]
        for name, a in arts.items():
            if a["opening"] > OPEN_FRAC:
                preds.add(("open", name))
            if a["opening"] < CLOSED_FRAC:
                preds.add(("closed", name))
            if name in grasped:
                preds.add(("grasped", name))
        return obs

    def render_segmentation(self, camera: str | int | None = None) -> np.ndarray:
        """As TabletopEnv, plus 2 + len(objects) + i for the i-th name in self.fixture_labels
        (moving parts by part name, carcasses by fixture name)."""
        labels = super().render_segmentation(camera)
        self.renderer.enable_segmentation_rendering()
        try:
            self.renderer.update_scene(self.data, camera=self.cam_id if camera is None else camera)
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        geom_ids, obj_types = seg[..., 0], seg[..., 1]
        is_geom = (obj_types == mujoco.mjtObj.mjOBJ_GEOM) & (geom_ids >= 0)
        body = np.full(geom_ids.shape, -1)
        body[is_geom] = self.model.geom_bodyid[geom_ids[is_geom]]
        base = 2 + len(self.objects)
        for bid, name in self._fixture_body.items():
            labels[body == bid] = base + self.fixture_labels.index(name)
        return labels

    @property
    def fixture_labels(self) -> list[str]:
        return list(dict.fromkeys(self._fixture_body.values()))

    @property
    def info(self) -> dict:
        """Ground truth for scoring (never part of observations): fixture types, parameters, poses,
        and each part's hidden cause."""
        gt = self.fx.ground_truth()
        for g in gt:
            xy, yaw = self.fixture_pose.get(g["fixture"], (np.zeros(2), 0.0))
            g["pose"] = {"xy": [float(xy[0]), float(xy[1])], "yaw": float(yaw)}
        return {"fixtures": gt}


class ArticulatedEventLog(EventLog):
    """EventLog plus fixture events: fixture_reset (ground truth incl. hidden causes),
    articulation_opened / articulation_closed, and grasp / release of a part's handle."""

    def __init__(self, env: ArticulatedEnv):
        super().__init__(env)
        self.tracked.update(env._fixture_body)
        self.reset()

    def reset(self) -> None:
        super().reset()
        env = self.env
        self._handles = self._compute_handle_grasped(self._contacts)
        self._art_state = {n: _art_state(a["opening"]) for n, a in env.fx.observe(env.data).items()}
        if env.fixture_pose:
            self._log("fixture_reset", fixtures=env.info["fixtures"])

    def grasped_handles(self) -> set[str]:
        return set(self._handles)

    def _compute_handle_grasped(self, contacts: set[frozenset]) -> set[str]:
        return {n for n in self.env.fx.parts
                if frozenset((n, "left_finger")) in contacts and frozenset((n, "right_finger")) in contacts}

    def update(self, obs: dict) -> None:
        super().update(obs)
        raw = self._compute_handle_grasped(self._contacts)
        for n in self._debounced("handle_grasp", raw - self._handles):
            self._log("grasp", object=n)
            self._handles.add(n)
        for n in self._debounced("handle_release", self._handles - raw):
            self._log("release", object=n)
            self._handles.discard(n)
        arts = obs["articulations"]
        # Debounced state machine per part; only entering open / closed is an event.
        for state, kind in (("open", "articulation_opened"), ("closed", "articulation_closed"), ("between", None)):
            entering = {n for n, a in arts.items() if _art_state(a["opening"]) == state and self._art_state[n] != state}
            for n in sorted(self._debounced(f"art_{state}", entering)):
                if kind:
                    self._log(kind, part=n, fixture=arts[n]["fixture"], opening=float(arts[n]["opening"]))
                self._art_state[n] = state


def _art_state(opening: float) -> str:
    return "open" if opening > OPEN_FRAC else "closed" if opening < CLOSED_FRAC else "between"
