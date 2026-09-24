"""Door/drawer scene for any embodiment: floor, one articulated fixture, one body, two cameras.

`DoorSceneEnv(embodiment="shadow", fixture_adder=add_test_door)` builds the scene with MjSpec:
1. the fixture is added through `fixture_adder(spec, pose, rng) -> HandleInfo` (or a dict with the same
   fields). `library_fixture(type)` wraps the embodiment-independent library pai.envs.fixtures
   ("lib:<type>" by name); a small local pair lives here too ("door": a cabinet door with a long
   vertical bar, "drawer": a drawer with a horizontal bar), sized for every hand including the G1's;
2. the scene is compiled once to find where the handle is; the embodiment is placed relative to it
   (hands hover in front, the Panda stands on a pedestal, the G1 stands on the floor);
3. cameras: "front" (an oblique view of body and fixture) and "fixture" (looking at the fixture).

The handle description is geometric and joint-agnostic: a site at the grasp point, the bar axis and the
approach direction in that site's frame, the fixture joint and its closed/open range. The opening
direction of the handle is the joint column of the handle-site Jacobian, so doors and drawers (and
anything else with one DoF) look the same to a controller.

step(action) takes the common embodiment action (see embodiments.py). Observation: the embodiment's
obs plus privileged fixture state under "fixture" (opening, handle pose, opening direction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from pai.envs.embodiments import Embodiment, make_embodiment
from pai.envs.scene import lookat_xyaxes


@dataclass
class HandleInfo:
    name: str
    kind: str                     # "door" | "drawer" | ...
    joint: str                    # the fixture's DoF; opening = increasing q
    handle_site: str              # site at the grasp centre of the handle bar
    axis_local: tuple             # bar axis in the site frame
    approach_local: tuple         # direction a hand moves to reach the bar (into the fixture), site frame
    open_range: tuple             # (closed q, fully open q)
    success_q: float              # opening counted as achieved beyond this q


def as_handle_info(info) -> HandleInfo:
    if isinstance(info, HandleInfo):
        return info
    if isinstance(info, dict):
        return HandleInfo(**{k: info[k] for k in HandleInfo.__dataclass_fields__ if k in info})
    raise TypeError(f"fixture_adder returned {type(info).__name__}; expected HandleInfo or dict")


# ---------------------------------------------------------------------------------------------- fixtures
WOOD = [0.62, 0.48, 0.34, 1]
PANEL = [0.78, 0.66, 0.5, 1]
METAL = [0.75, 0.75, 0.78, 1]


def _yaw_quat(yaw: float) -> list[float]:
    return [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def _carcass(body, half, z0, thick=0.015, name="cab"):
    """Open-front box: back, sides, top, bottom. The front plane is x = 0 of `body`, depth along +x."""
    dx, dy, dz = half
    box = mujoco.mjtGeom.mjGEOM_BOX
    parts = {
        "back": ([2 * dx - thick, 0, z0 + dz], [thick, dy, dz]),
        "left": ([dx, dy - thick, z0 + dz], [dx, thick, dz]),
        "right": ([dx, -dy + thick, z0 + dz], [dx, thick, dz]),
        "top": ([dx, 0, z0 + 2 * dz - thick], [dx, dy, thick]),
        "bottom": ([dx, 0, z0 + thick], [dx, dy, thick]),
    }
    for k, (p, s) in parts.items():
        body.add_geom(name=f"{name}_{k}", type=box, pos=p, size=s, rgba=WOOD)


def _bar_handle(body, center, axis: str, half_len=0.07, standoff=0.065, name="handle"):
    """Bar handle in front (-x) of a panel face at `center`, bar along the body's `axis` ('y' or 'z')."""
    cap, box = mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX
    c = np.asarray(center, float)
    bar_c = c + [-standoff, 0, 0]
    along = np.array([0, 1, 0]) if axis == "y" else np.array([0, 0, 1])
    quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0] if axis == "y" else [1, 0, 0, 0]
    body.add_geom(name=f"{name}_bar", type=cap, size=[0.012, half_len, 0], pos=bar_c, quat=quat, rgba=METAL,
                  friction=[1.2, 0.02, 0.001], condim=4)
    for s in (-1, 1):
        body.add_geom(name=f"{name}_post{'ab'[s > 0]}", type=box, size=[standoff / 2, 0.008, 0.008],
                      pos=c + [-standoff / 2, 0, 0] + s * (half_len - 0.012) * along, rgba=METAL)
    body.add_site(name=f"{name}_site", pos=bar_c, size=[0.01] * 3, rgba=[0.2, 0.9, 0.2, 0.5], group=4)


def add_test_door(spec: mujoco.MjSpec, pose=((0.5, 0.0, 0.0), 0.0), rng=None) -> HandleInfo:
    """Cabinet with a door hinged on its left edge (seen from the front) and a vertical bar handle.
    pose = (front-centre position on the floor, yaw); the front faces -x at yaw 0."""
    rng = rng or np.random.default_rng()
    pos, yaw = pose
    height = 0.8 + rng.uniform(-0.03, 0.03)          # handle height
    root = spec.worldbody.add_body(name="door_cabinet", pos=list(pos), quat=_yaw_quat(yaw))
    w, h = 0.22, 0.28                                # door half width / half height
    _carcass(root, (0.2, w + 0.015, h + 0.015), z0=height - h - 0.015, name="door_cab")
    root.add_geom(name="door_plinth", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, w + 0.015, (height - h - 0.015) / 2],
                  pos=[0.2, 0, (height - h - 0.015) / 2], rgba=WOOD)
    door = root.add_body(name="door", pos=[-0.012, w, height])      # hinge line at the left edge
    door.add_joint(name="door_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, -1], range=[0, 1.8],
                   damping=0.4, frictionloss=0.05, armature=0.01)
    door.add_geom(name="door_panel", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, w, h], pos=[0, -w, 0],
                  rgba=PANEL, mass=2.0)
    _bar_handle(door, center=[-0.01, -2 * w + 0.05, 0], axis="z", name="door_handle")
    return HandleInfo("door", "door", "door_hinge", "door_handle_site", (0, 0, 1), (1, 0, 0), (0.0, 1.8), 0.3)


def add_test_drawer(spec: mujoco.MjSpec, pose=((0.5, 0.0, 0.0), 0.0), rng=None) -> HandleInfo:
    """Cabinet with one drawer and a horizontal bar handle on its front."""
    rng = rng or np.random.default_rng()
    pos, yaw = pose
    height = 0.72 + rng.uniform(-0.03, 0.03)
    root = spec.worldbody.add_body(name="drawer_cabinet", pos=list(pos), quat=_yaw_quat(yaw))
    w, h, depth = 0.2, 0.08, 0.34
    t = 0.015
    _carcass(root, (0.2, w + 2 * t + 0.005, h + 2 * t + 0.005), z0=height - h - 2 * t - 0.005, name="drawer_cab")
    z0 = height - h - 2 * t - 0.005
    root.add_geom(name="drawer_plinth", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, w + 2 * t + 0.005, z0 / 2],
                  pos=[0.2, 0, z0 / 2], rgba=WOOD)
    drawer = root.add_body(name="drawer", pos=[0, 0, height])
    drawer.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1, 0, 0], range=[0, 0.3],
                     damping=3.0, frictionloss=0.2, armature=0.05)
    box = mujoco.mjtGeom.mjGEOM_BOX
    drawer.add_geom(name="drawer_front", type=box, size=[0.01, w + t, h + t], pos=[-0.01, 0, 0], rgba=PANEL, mass=0.8)
    drawer.add_geom(name="drawer_bottom", type=box, size=[depth / 2, w - t, 0.005], pos=[depth / 2, 0, -h + 0.005],
                    rgba=WOOD, mass=0.4)
    for s in (-1, 1):
        drawer.add_geom(name=f"drawer_side{'ab'[s > 0]}", type=box, size=[depth / 2, 0.005, h - 0.01],
                        pos=[depth / 2, s * (w - t), 0], rgba=WOOD, mass=0.2)
    drawer.add_geom(name="drawer_back", type=box, size=[0.005, w - t, h - 0.01], pos=[depth, 0, 0], rgba=WOOD, mass=0.2)
    _bar_handle(drawer, center=[-0.02, 0, 0], axis="y", half_len=0.08, name="drawer_handle")
    return HandleInfo("drawer", "drawer", "drawer_slide", "drawer_handle_site", (0, 1, 0), (1, 0, 0), (0.0, 0.3), 0.08)


LOCAL_FIXTURES: dict[str, Callable] = {"door": add_test_door, "drawer": add_test_drawer}
LIBRARY_SUCCESS_FRAC = 0.25     # a library fixture counts as opened beyond this fraction of its range


def library_fixture(type_name: str, table_height: float | None = None, **overrides) -> Callable:
    """Adder for a fixture type of the embodiment-independent library (pai.envs.fixtures). Tabletop-sized
    types stand on a table (default 0.6 m) so every body reaches them; room doors stand on the floor.
    The library's fixture frame has +x out of the front; here the front faces -x (the body)."""
    from pai.envs import fixtures as F

    def adder(spec: mujoco.MjSpec, pose=((0.5, 0.0, 0.0), 0.0), rng=None) -> HandleInfo:
        rng = rng or np.random.default_rng()
        (x, y, z), yaw = pose
        h = (0.6 if F.FIXTURE_TYPES[type_name].tabletop else 0.0) if table_height is None else table_height
        fx = F.make_fixture(rng, type_name, **overrides)
        f = F.add_fixture(spec, spec.worldbody, fx, "fx", pos=(x, y, z + h), quat=_yaw_quat(yaw + np.pi))
        if h > 0:
            xmin, _, ymin, ymax = f.keepout            # local; the carcass lies at local x < 0
            depth = -xmin
            c = np.array([x, y, z]) + np.array([np.cos(yaw), np.sin(yaw), 0]) * depth / 2
            spec.worldbody.add_geom(name="fx_table", type=mujoco.mjtGeom.mjGEOM_BOX, quat=_yaw_quat(yaw),
                                    size=[depth / 2 + 0.03, (ymax - ymin) / 2 + 0.05, h / 2],
                                    pos=[c[0], c[1], z + h / 2], rgba=[0.42, 0.36, 0.3, 1])
        part = next(p for p in f.parts if p.name == f.target)
        lo, hi = part.range
        return HandleInfo(f.name, fx.family, part.joint, part.handle_site, (0, 0, 1), (-1, 0, 0), (lo, hi),
                          lo + LIBRARY_SUCCESS_FRAC * (hi - lo))

    return adder


def get_fixture_adder(name: str) -> Callable:
    """Local test fixtures ("door", "drawer") or a library type ("lib:drawer_bar", "lib:room_door_bar")."""
    if name.startswith("lib:"):
        return library_fixture(name[4:])
    return LOCAL_FIXTURES[name]


# ---------------------------------------------------------------------------------------------- env
class DoorSceneEnv:
    def __init__(self, embodiment: str | Embodiment = "shadow", fixture_adder: Callable | str = "door",
                 control_dt: float = 0.02, image_size: int = 128, render_images: bool = False,
                 fixture_pose=((0.5, 0.0, 0.0), 0.0), seed: int | None = 0):
        self.emb = make_embodiment(embodiment) if isinstance(embodiment, str) else embodiment
        adder = get_fixture_adder(fixture_adder) if isinstance(fixture_adder, str) else fixture_adder
        self.rng = np.random.default_rng(seed)
        self.render_images = render_images
        self.image_size = image_size

        spec = mujoco.MjSpec()
        spec.option.timestep = 0.002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.compiler.degree = False
        wb = spec.worldbody
        wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05], rgba=[0.3, 0.32, 0.36, 1])
        wb.add_light(name="key", pos=[-0.5, -1.0, 2.5], dir=[0.4, 0.4, -1.0], diffuse=[0.7, 0.7, 0.7])
        wb.add_light(name="fill", pos=[-1.5, 1.0, 2.0], dir=[0.6, -0.3, -1.0], diffuse=[0.35, 0.35, 0.4])
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]

        self.fixture = as_handle_info(adder(spec, fixture_pose, self.rng))
        # locate the handle before the body exists, then place the body relative to it
        probe = spec.compile()
        pd = mujoco.MjData(probe)
        mujoco.mj_kinematics(probe, pd)
        sid = probe.site(self.fixture.handle_site).id
        R = pd.site_xmat[sid].reshape(3, 3)
        self.anchor = pd.site_xpos[sid].copy()
        self.approach0 = R @ np.asarray(self.fixture.approach_local, float)
        self.emb.attach(spec, self.anchor, self.approach0)

        a, n = self.anchor, self.approach0
        side = np.cross([0, 0, 1], n)
        cams = {"front": (a - 0.6 * n - 1.3 * side + [0, 0, 0.4], a - 0.3 * n + [0, 0, -0.1]),
                "fixture": (a - 0.8 * n - 0.7 * side + [0, 0, 0.3], a - 0.1 * n + [0, 0, -0.05])}
        for name, (p, t) in cams.items():
            wb.add_camera(name=name, pos=list(p), xyaxes=lookat_xyaxes(p, t), fovy=55)

        self.spec = spec
        self.model = spec.compile()
        m = self.model
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, image_size)
        m.vis.global_.offheight = max(m.vis.global_.offheight, image_size)
        self.data = mujoco.MjData(m)
        self.emb.bind(m, self.data)
        self.handle_site = m.site(self.fixture.handle_site).id
        j = m.joint(self.fixture.joint)
        self.fix_qadr, self.fix_vadr = j.qposadr[0], j.dofadr[0]
        self.n_substeps = max(1, round(control_dt / m.opt.timestep))
        self.control_dt = self.n_substeps * m.opt.timestep
        self._renderer = None
        self.t = 0
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int | None = None, palm_noise: float = 0.0) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        offset = self.rng.uniform(-palm_noise, palm_noise, 3) if palm_noise > 0 else None
        self.emb.reset(offset)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):  # let the fingers reach the open posture
            self.emb.apply(np.r_[np.zeros(6), 1.0], self.control_dt)
            mujoco.mj_step(self.model, self.data)
        self.t = 0
        return self._observe()

    def step(self, action: np.ndarray) -> dict:
        self.emb.apply(action, self.control_dt)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        return self._observe()

    def stable(self) -> bool:
        d = self.data
        return bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)) and np.abs(d.qvel).max() < 1e3
                    and d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0)

    # ------------------------------------------------------------------ sensing
    def fixture_state(self) -> dict:
        m, d, f = self.model, self.data, self.fixture
        R = d.site_xmat[self.handle_site].reshape(3, 3)
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, None, self.handle_site)
        od = jp[:, self.fix_vadr]
        q = float(d.qpos[self.fix_qadr])
        lo, hi = f.open_range
        return {
            "kind": f.kind,
            "opening": q,
            "opening_frac": (q - lo) / (hi - lo),
            "opened": q >= f.success_q,
            "handle_pos": d.site_xpos[self.handle_site].copy(),
            "handle_rot": R.copy(),
            "handle_axis": R @ np.asarray(f.axis_local, float),
            "handle_approach": R @ np.asarray(f.approach_local, float),
            "open_dir": od / max(np.linalg.norm(od), 1e-9),
        }

    def handle_contact_fingers(self) -> int:
        """Number of fingers touching the fixture's moving body (privileged; used to score grasps)."""
        m, d = self.model, self.data
        fixture_body = m.jnt_bodyid[m.joint(self.fixture.joint).id]
        touching = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for b, o in ((b1, b2), (b2, b1)):
                k = self.emb.body_finger[b]
                if k >= 0 and o == fixture_body:
                    touching.add(int(k))
        return len(touching)

    def _observe(self) -> dict:
        obs = self.emb.observe()
        obs["fixture"] = self.fixture_state()
        obs["image"] = self.render() if self.render_images else None
        return obs

    def render(self, camera: str = "front", size: int | None = None) -> np.ndarray:
        size = size or self.image_size
        if self._renderer is None or self._renderer.width != size:
            if self._renderer is not None:
                self._renderer.close()
            m = self.model
            m.vis.global_.offwidth = max(m.vis.global_.offwidth, size)
            m.vis.global_.offheight = max(m.vis.global_.offheight, size)
            self._renderer = mujoco.Renderer(m, size, size)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
