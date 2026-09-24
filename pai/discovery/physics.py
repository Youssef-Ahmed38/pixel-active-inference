"""PhysicsWorld: the discovery interface (interface.World) over the MuJoCo home-door scene.

The body (pai.envs.embodiments, the Robotiq gripper by default) stands in front of a home door
(pai.envs.home_doors) with its side cabinet, drawer, shelf and keys. This module is the body's side of
the contract: it turns the scene into opaque entities and generic features, and it carries out the
action it is given by composing low-level motions (the oracle's P-controllers on the palm target:
pai.envs.home_oracle, and pai.skills' stall detection). It never decides WHAT to do: it does not read
the lock state, which key matches, or where a key was put (env.info), except in truth().

    entities   parts: every jointed thing on the fixtures the hand could act on (door leaf, the door's
               handle, a thumb-turn, a lock cylinder face, the drawer, grasped by its bar); objects:
               the free bodies (keys) that are in view. A key lying inside the closed drawer (its bow
               inside the cabinet cavity) or behind the wall while the door is shut is not in view; that
               is decided from geometry every time, never from the placement record.
               Ids are "p<k>" / "o<k>", shuffled per reset seed. Attributes are generic and use the
               MockWorld's vocabulary: joint type; graspable (false for a flat face flush with its
               panel: the door leaf, the lock cylinder); shape from the geom primitive (panel, plate,
               box, bar, wing = a short bar, round, disc, ball; a free object with a long geom is
               "elongated"); size; colour as an RGB tuple. The keys' colours are drawn afresh from the
               reset seed every episode, so a colour never tells which key fitted last time.
    features   "part:<id>.open" closed | ajar | open         (the door leaf, the drawer)
               "part:<id>.angle" rest | pos | neg            (a part's own joint: a knob's or thumb-turn's
                                                              hinge, a push bar's slide)
               "obj:<id>.where" free | held | in:<part id>   (tip inside a part along its axis)
               "obj:<id>.angle" rest | pos | neg             (an inserted object, turned about the part
                                                              axis; "rest" for any other object)
               "body:hand"      empty | <object id>          (what the hand holds)
               A thumb-turn's angle is its physical orientation, so a thrown thumb deadbolt reads "pos"
               from the start (as in the mock); a key lock's state is not visible.
    actions    from attributes only: every graspable part gets the probes (pull, push, turn_pull +/-1,
               turn_push +/-1, press_push) and turn +/-1; a non-graspable jointed part: push (a closed
               hand pressed against it); a visible object not in the hand: pick; while holding: insert
               into every part, release and turn_held +/-1 (in the hand, or the part it is in; and
               pick, which withdraws an inserted one). One hand: while it holds something, parts cannot
               be probed.

How an action runs (all motion through the common embodiment action; "carry" = the palm follows the
arc of whatever joint the grasped part rides on, a door hinge or a drawer slide, with a small lead, and
stops when that joint stops making progress):

    pull / push        grasp the part, carry it towards / away from the body
    turn_pull / _push  grasp, turn the part about its own joint (its normal if it has none) and hold it
                       turned while carrying
    press_push         press a closed hand into the part along its approach, then carry away
    turn               grasp, turn, let go
    pick               grasp the object from above (then along its long axis if that fails), lift
    insert             bring the held object's tip onto the part's axis, aligned, and push it in
    turn_held          turn the inserted object about the part's joint; a free one in the hand is rolled
                       a quarter turn about its long axis and back (nothing else happens)
    release            let go: in a part it stays there; a free object is put back where it was picked
    push (not grasp.)  press a closed hand into the part's face, then carry it away

Between actions the empty hand opens, backs off and returns to its home pose, so actions compose; a
held object stays in the hand. When an action opens the door, the hand keeps holding it until the next
action, so a door closer does not swing it shut before the agent sees it open. goal_reached() is the
goal condition on features(), nothing else; truth()["opened"] records whether it was ever reached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from pai.discovery.interface import Action, Entity, Outcome
from pai.envs.door_scene import DoorSceneEnv
from pai.envs.embodiments import rot_error, rotvec_to_mat
from pai.envs.home_doors import DOOR_OPEN_FRAC, home_door_fixture
from pai.envs.home_oracle import HomeDoorOracle
from pai.skills.core import StallMonitor

TURN_ANGLE = 1.75        # rad: a turn goes this far (a bit past a quarter turn) or until the part stops
TURN_PROBE = 0.7         # rad: how far turn_pull / turn_push turn before moving (held turned while moving)
TURN_SPEED = 0.8         # rad/s
TURN_LEAD = 0.35         # rad: the commanded turn runs at most this far ahead of the part
TURN_MOVED = 0.15        # rad: a turn that moved the part less than this has stalled
CARRY_HINGE = 0.9        # rad a carried hinge is moved (the door counts as open from 0.66)
CARRY_SLIDE = 0.21       # m a carried slide is moved
LEAD_HINGE, LEAD_SLIDE = 0.12, 0.03
CARRY_PATIENCE = 80      # control steps without progress before a carry gives up
MOVED_HINGE, MOVED_SLIDE = 0.03, 0.01   # a carry that moved less than this has stalled
PRESS_DEPTH = 0.035      # m a closed hand presses past a part's grasp point
INSERT_DEPTH = 0.034     # m the tip is pushed past a part's mouth
INSERTED_DEPTH = 0.012   # m past the mouth, on the axis: the object is "in" the part
ANGLE_REST, SLIDE_REST = 0.3, 0.008     # rad / m from the joint's model default counted as moved
CLOSED_FRAC = 0.02
DRAWER_OPEN_FRAC = 0.5
LIFT = 0.2               # m an object is lifted after the grasp
NEUTRAL_RISE = 0.15      # m the neutral (home) pose sits above the body's start pose
RISE_FIRST = 0.15        # m: a pre-grasp this far above the hand is reached by rising in place first

WING_LEN = 0.1           # m: a bar shorter than this is a "wing" (a tab to turn), longer a "bar"
HELD_ROLL = np.pi / 2    # rad: turn_held on a free object in the hand rolls it this far (and back)
OBJ_RGB = (0.2, 0.95)    # range of the per-episode object colours (each channel)


class _Abort(Exception):
    pass


@dataclass
class _Part:
    name: str                        # internal (never shown to the agent)
    geom: int                        # the geom that gives it its look
    frame: object                    # () -> (pos, R): R[:, 0] outward normal (approach = -x), R[:, 2] bar axis
    joint: int | None = None         # its own joint (turned / pressed / opened)
    carrier: int | None = None       # the joint it rides on (a door hinge, a drawer slide)
    graspable: bool = True
    opens: bool = False              # a door leaf / drawer: feature "open"


def _colour(rgba) -> tuple:
    """A colour as the MockWorld gives it: an RGB tuple, 2 decimals."""
    return tuple(round(float(x), 2) for x in rgba[:3])


def _shape(model: mujoco.MjModel, g: int) -> tuple[str, float]:
    """Shape class and size (largest extent, m) of a geom, from its primitive only."""
    t, s = model.geom_type[g], model.geom_size[g]
    G = mujoco.mjtGeom
    if t == G.mjGEOM_BOX:
        a, b, c = sorted(s)
        if c > 3 * b:                                   # long: a bar (a short one: a wing)
            shape = "wing" if 2 * c < WING_LEN else "bar"
        elif a < 0.3 * b:                               # flat: a panel if large, else a plate
            shape = "panel" if 2 * c >= 0.25 else "plate"
        else:
            shape = "box"
        return shape, 2 * c
    if t == G.mjGEOM_CAPSULE:
        n = 2 * (s[1] + s[0])
        return ("wing" if n < WING_LEN else "bar"), n
    if t == G.mjGEOM_CYLINDER:
        return ("disc" if s[1] < 0.3 * s[0] else "round"), 2 * max(s[0], s[1])
    if t == G.mjGEOM_SPHERE:
        return "ball", 2 * float(s[0])
    if t == G.mjGEOM_ELLIPSOID:                         # a squashed ball is still a ball, a flat one a disc
        a, _, c = sorted(s[:3])
        return ("ball" if a > 0.5 * c else "disc"), 2 * float(c)
    return "round", 2 * float(max(s[:3]))


class PhysicsWorld:
    """interface.World over DoorSceneEnv + a home door. `reset(seed, lock_state=, key_place=)` starts an
    episode (the constructor resets with `seed`)."""

    def __init__(self, type_name: str = "hd_knob_pull_left", embodiment: str = "robotiq_2f85", seed: int = 0,
                 lock_state: str | None = None, key_place: str | None = None, env: DoorSceneEnv | None = None,
                 max_action_steps: int = 3000, **fixture_overrides):
        self.env = env or DoorSceneEnv(embodiment, home_door_fixture(type_name, **fixture_overrides), seed=seed)
        self.rt, self.m, self.d, self.emb = self.env.runtime, self.env.model, self.env.data, self.env.emb
        self.dt = self.env.control_dt
        self.max_action_steps = max_action_steps
        self._motion = HomeDoorOracle(self.env)    # only its motion primitives (_goto, _grasp, _hold, _cmd)
        self._build_parts()
        self.reset(seed, lock_state=lock_state, key_place=key_place)

    # ------------------------------------------------------------------ scene description (world side)
    def _build_parts(self) -> None:
        m, d, r = self.m, self.d, self.rt.rec
        gid = lambda n: m.geom(n).id    # noqa: E731
        jid = lambda n: m.joint(n).id   # noqa: E731

        def site_frame(name):
            i = m.site(name).id
            return lambda: (d.site_xpos[i].copy(), d.site_xmat[i].reshape(3, 3).copy())

        def body_face(body, geom, depth):
            b, g = m.body(body).id, gid(geom)

            def f():
                R = d.xmat[b].reshape(3, 3).copy()
                return d.geom_xpos[g] + depth * R[:, 0], R
            return f

        head = {"knob_round": "home_head", "knob_ball": "home_head", "lever": "home_arm",
                "push_bar": "home_pushbar_bar"}[r.handle]
        door_j, drawer_j = jid(r.door_joint), jid(r.drawer["joint"])
        # the drawer is one part (as in the mock): it looks like its front and is grasped by its bar
        parts = [_Part("door", gid(f"{r.name}_door_panel"), body_face(r.door_body, f"{r.name}_door_panel", 0.02),
                       joint=door_j, graspable=False, opens=True),
                 _Part("handle", gid(head), site_frame(r.handle_site), joint=jid(r.handle_joint), carrier=door_j),
                 _Part("drawer", gid("cab_drawer_front"), site_frame(r.drawer["site"]), joint=drawer_j,
                       carrier=drawer_j, opens=True)]
        if r.thumb:
            parts.append(_Part("thumb", gid(f"{r.name}_thumb_wing"), site_frame(r.thumb["site"]),
                               joint=jid(r.thumb["joint"]), carrier=door_j))
        if r.key_lock:
            pb = m.body(r.key_lock["plug_body"]).id

            def cyl():   # plug frame: x into the door, y along the slot -> normal out, bar axis along the slot
                R = d.xmat[pb].reshape(3, 3)
                x, z = -R[:, 0], R[:, 1]
                return d.xpos[pb].copy(), np.column_stack([x, np.cross(z, x), z])
            # a face flush with the door, nothing to close a hand on: not graspable (as in the mock)
            parts.append(_Part("cylinder", gid(f"{r.name}_cyl_rim"), cyl, joint=jid(r.key_lock["plug"]),
                               carrier=door_j, graspable=False))
        self.parts = parts
        self.objects = [dict(k) for k in self.rt.keys]
        for k in self.objects:
            k["geoms"] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == k["body"]]
            # its look: colour from the geom with the most volume (a key's bow), shape from the longest
            k["look"] = max(k["geoms"], key=lambda g: float(np.prod(m.geom_size[g])))
            k["long"] = max(k["geoms"], key=lambda g: float(np.max(m.geom_size[g])))
        # cavity of the side cabinet (between its middle shelf and its top) and the wall plane, in the
        # fixture frame: what hides an object from view
        base = m.body(f"{r.name}_base").id
        self._base = base
        top, mid = gid("cab_top"), gid("cab_mid")
        self._cavity = (m.geom_pos[top], m.geom_size[top], m.geom_pos[mid], m.geom_size[mid])
        self._wall_x = float(m.geom_pos[gid("wall_hinge")][0] - m.geom_size[gid("wall_hinge")][0])

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int = 0, lock_state: str | None = None, key_place: str | None = None) -> None:
        opts = {k: v for k, v in (("lock_state", lock_state), ("key_place", key_place)) if v is not None}
        self.obs = self.env.reset(seed=seed, **opts)
        rng = np.random.default_rng([int(seed), 7919])
        self.pid = {p.name: f"p{i + 1}" for p, i in zip(self.parts, rng.permutation(len(self.parts)))}
        self.oid = {k["name"]: f"o{i + 1}" for k, i in zip(self.objects, rng.permutation(len(self.objects)))}
        self._part_by_id = {self.pid[p.name]: p for p in self.parts}
        self._obj_by_id = {self.oid[k["name"]]: k for k in self.objects}
        # the fixture (and each key body's bitting) is built once, so a key body's look must not carry
        # over: every episode draws the objects' colours afresh, independently of which one fits
        for k in self.objects:
            self.m.geom_rgba[k["look"], :3] = rng.uniform(*OBJ_RGB, 3)
        self.held: str | None = None           # object id in the hand
        self._put_back = None                  # palm pose where the held object was taken
        self._holding_part = False             # the hand is on a part (during a probe)
        self._keep = False                     # it stays on the opened door until the next action
        self._opened = False                   # the goal was reached in this episode
        self._flip: dict[str, float] = {}      # inserted object -> which way round it went in
        self.unstable = False
        self.t = 0
        self._q0 = {p.name: float(self.m.qpos0[self.m.jnt_qposadr[p.joint]]) for p in self.parts if p.joint is not None}
        # the neutral pose: the body's start pose raised above table height, so that turning the hand into
        # a grasp there does not sweep objects off the side cabinet
        self._steps = 0
        self._goto(self.obs["palm_pos"] + [0, 0, NEUTRAL_RISE], self.obs["palm_rot"].copy(), 1.0, tol=0.01, steps=150)
        self.home = (self.obs["palm_pos"].copy(), self.obs["palm_rot"].copy())
        self.t = 0

    # ------------------------------------------------------------------ interface: entities / features
    def _visible(self, k: dict) -> bool:
        d, m = self.d, self.m
        Rb = d.xmat[self._base].reshape(3, 3)
        p = Rb.T @ (d.site_xpos[k["grip"]] - d.xpos[self._base])       # the bow, fixture frame
        tp, ts, mp, ms = self._cavity
        inside = (abs(p[0] - tp[0]) < ts[0] - 0.01 and abs(p[1] - tp[1]) < ts[1] and mp[2] + ms[2] < p[2] < tp[2] - ts[2])
        if inside:
            return False
        door = next(q for q in self.parts if q.name == "door")
        if p[0] < self._wall_x and self._frac(door) < 0.3:
            return False
        return True

    def _frac(self, p: _Part) -> float:
        lo, hi = self.m.jnt_range[p.joint]
        return float(np.clip((self.d.qpos[self.m.jnt_qposadr[p.joint]] - lo) / (hi - lo), 0, 1))

    def _visible_objects(self) -> list[dict]:
        return [k for k in self.objects if self.oid[k["name"]] == self.held or self._visible(k)]

    def entities(self) -> list[Entity]:
        m, d = self.m, self.d
        out = []
        for p in self.parts:
            shape, size = _shape(m, p.geom)
            jt = "none" if p.joint is None else ("slide" if m.jnt_type[p.joint] == mujoco.mjtJoint.mjJNT_SLIDE
                                                 else "hinge")
            attrs = {"shape": shape, "size": round(size, 2), "colour": _colour(m.geom_rgba[p.geom])}
            out.append(Entity(self.pid[p.name], "part", jt, p.graspable, tuple(np.round(p.frame()[0], 3)), attrs))
        for k in self._visible_objects():
            shape, _ = _shape(m, k["long"])
            if shape in ("bar", "wing"):
                shape = "elongated"                                 # a free object with a long piece
            lo = np.min([m.geom_pos[i] - m.geom_size[i] for i in k["geoms"]], axis=0)
            hi = np.max([m.geom_pos[i] + m.geom_size[i] for i in k["geoms"]], axis=0)
            attrs = {"shape": shape, "size": round(float(np.max(hi - lo)), 3), "colour": _colour(m.geom_rgba[k["look"]])}
            out.append(Entity(self.oid[k["name"]], "object", "none", True, tuple(np.round(d.xpos[k["body"]], 3)), attrs))
        out.sort(key=lambda e: (e.kind, int(e.id[1:])))
        return out

    def _joint_state(self, p: _Part) -> str:
        m = self.m
        dq = float(self.d.qpos[m.jnt_qposadr[p.joint]]) - self._q0[p.name]
        thr = SLIDE_REST if m.jnt_type[p.joint] == mujoco.mjtJoint.mjJNT_SLIDE else ANGLE_REST
        return "rest" if abs(dq) < thr else ("pos" if dq > 0 else "neg")

    def _inserted_in(self, k: dict) -> _Part | None:
        """The part whose axis the object's tip is on, past its mouth (geometry only)."""
        tip = self.d.xpos[k["body"]]
        for p in self.parts:
            pos, R = p.frame()
            u = -R[:, 0]
            rel = tip - pos
            depth = float(rel @ u)
            if INSERTED_DEPTH < depth < 0.06 and np.linalg.norm(rel - depth * u) < 0.006:
                return p
        return None

    def _obj_angle(self, k: dict, p: _Part) -> float:
        """Roll of the object's width axis about the part's joint, from the part's rest orientation (a
        flat blade fits either way round: measured from the way it went in)."""
        m, d = self.m, self.d
        b = m.jnt_bodyid[p.joint]
        Rp = d.xmat[m.body_parentid[b]].reshape(3, 3)
        Rl = np.zeros(9)
        mujoco.mju_quat2Mat(Rl, m.body_quat[b])
        R_rest = Rp @ Rl.reshape(3, 3)
        axis = d.xaxis[p.joint]
        y0 = R_rest[:, 1]
        yk = d.xmat[k["body"]].reshape(3, 3)[:, 1]
        oid = self.oid[k["name"]]
        if oid not in self._flip:           # it can only go in at rest: remember which way round
            self._flip[oid] = 1.0 if y0 @ yk >= 0 else -1.0
        yk = yk * self._flip[oid]
        return float(np.arctan2(np.cross(y0, yk) @ axis, y0 @ yk))

    def _holding(self, k: dict) -> bool:
        """The hand is closed on the object: a finger touches it and its grip point is at the palm."""
        m, d = self.m, self.d
        if np.linalg.norm(d.site_xpos[k["grip"]] - self.obs["palm_pos"]) > 0.06:
            return False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (self.emb.body_finger[b1] >= 0 and b2 == k["body"]) or (self.emb.body_finger[b2] >= 0 and b1 == k["body"]):
                return True
        return False

    def features(self) -> dict[str, str]:
        f = {}
        for p in self.parts:
            pid = self.pid[p.name]
            if p.opens:
                fr = self._frac(p)
                open_at = DOOR_OPEN_FRAC if p.name == "door" else DRAWER_OPEN_FRAC
                f[f"part:{pid}.open"] = "closed" if fr < CLOSED_FRAC else ("open" if fr >= open_at else "ajar")
            elif p.joint is not None:          # a hinge's angle or a slide's travel: one name, as in the mock
                f[f"part:{pid}.angle"] = self._joint_state(p)
        for k in self._visible_objects():
            oid = self.oid[k["name"]]
            p = self._inserted_in(k)
            if p is not None:
                f[f"obj:{oid}.where"] = f"in:{self.pid[p.name]}"
                a = self._obj_angle(k, p)
                f[f"obj:{oid}.angle"] = "rest" if abs(a) < ANGLE_REST else ("pos" if a > 0 else "neg")
            else:
                self._flip.pop(oid, None)
                f[f"obj:{oid}.where"] = "held" if oid == self.held and self._holding(k) else "free"
                f[f"obj:{oid}.angle"] = "rest"
        f["body:hand"] = self.held or "empty"
        return f

    def actions(self) -> list[Action]:
        f = self.features()
        parts = self.parts
        objs = [self.oid[k["name"]] for k in self._visible_objects()]
        acts: list[Action] = []
        if self.held is None:
            for p in parts:
                pid = self.pid[p.name]
                if not p.graspable:
                    if p.joint is not None or p.carrier is not None:
                        acts.append(("push", pid))          # a closed hand pressed against it
                    continue
                acts += [("pull", pid), ("push", pid), ("turn_pull", pid, 1), ("turn_pull", pid, -1),
                         ("turn_push", pid, 1), ("turn_push", pid, -1), ("press_push", pid),
                         ("turn", pid, 1), ("turn", pid, -1)]
            for o in objs:
                acts.append(("pick", o))
                if f.get(f"obj:{o}.where", "").startswith("in:"):
                    acts += [("turn_held", o, 1), ("turn_held", o, -1)]
        else:
            o = self.held
            if f.get(f"obj:{o}.where", "").startswith("in:"):
                acts += [("turn_held", o, 1), ("turn_held", o, -1), ("pick", o)]
            else:
                acts += [("insert", o, self.pid[p.name]) for p in parts]
                acts += [("turn_held", o, 1), ("turn_held", o, -1)]     # rolled in the hand
            acts.append(("release", o))
        return acts

    def goal(self) -> dict[str, str]:
        return {f"part:{self.pid['door']}.open": "open"}

    def goal_reached(self) -> bool:
        """The goal condition on features(), exactly (truth()['opened']: was it ever reached)."""
        f = self.features()
        return all(f.get(k) == v for k, v in self.goal().items())

    def truth(self) -> dict:
        """Ground truth for scoring only (never read by the agent): env.info, the id map, the events,
        whether the door was ever open in this episode."""
        return {**self.env.info, "ids": {**{v: k for k, v in self.pid.items()}, **{v: k for k, v in self.oid.items()}},
                "events": [e["type"] for e in self.rt.events], "t": self.t, "opened": self._opened}

    # ------------------------------------------------------------------ motion plumbing
    def _run(self, gen):
        """Drive a motion generator (yields actions, receives observations); returns its value."""
        try:
            a = next(gen)
            while True:
                self._step(a)
                a = gen.send(self.obs)
        except StopIteration as e:
            return e.value

    def _step(self, a) -> dict:
        self.obs = self.env.step(a)
        self.t += 1
        self._steps += 1
        if not self.env.stable():
            raise _Abort("unstable")
        if self._steps > self.max_action_steps:
            raise _Abort("step budget")
        if self.goal_reached_now():
            self._opened = True
            self._keep = self._keep or self._holding_part
        return self.obs

    def goal_reached_now(self) -> bool:
        door = next(p for p in self.parts if p.name == "door")
        return self._frac(door) >= DOOR_OPEN_FRAC

    def _cmd(self, pos, R, grip):
        return self._motion._cmd(self.obs, pos, R, grip)

    def _goto(self, pos, R, grip, **kw):
        return self._run(self._motion._goto(self.obs, np.asarray(pos, float), R, grip, **kw))

    def _hold(self, grip, steps, pos=None, R=None):
        return self._run(self._motion._hold(self.obs, grip, steps, pos, R))

    def _grip_ok(self) -> bool:
        return int(np.sum(self.obs["finger_force"] > 0.3)) >= min(2, self.emb.n_fingers)

    def _grasp_at(self, pos_fn, axis, approach, standoff: float = 0.1) -> None:
        """The oracle's grasp (pre-grasp well clear, straight in, close). For a pre-grasp well above the hand
        it first rises in place, so it does not sweep up through a shelf edge on the way."""
        R = self.emb.grasp_rot(axis, approach, self.obs["palm_rot"])
        pre = pos_fn(self.obs) - (standoff + 0.12) * R[:, 2]
        if pre[2] - self.obs["palm_pos"][2] > RISE_FIRST:
            up = self.obs["palm_pos"].copy()
            up[2] = pre[2]
            self._goto(up, self.obs["palm_rot"].copy(), 1.0, tol=0.03, steps=200)
        self._run(self._motion._grasp(self.obs, pos_fn, axis, approach, standoff=standoff))

    def _grasp_part(self, p: _Part) -> bool:
        pos_fn = lambda o: p.frame()[0]     # noqa: E731
        _, R = p.frame()
        self._grasp_at(pos_fn, R[:, 2], -R[:, 0])
        return self._grip_ok()

    def _joint_q(self, j: int) -> float:
        return float(self.d.qpos[self.m.jnt_qposadr[j]])

    def _turn(self, p: _Part, sign: int, grip: float = -1.0, angle: float = TURN_ANGLE, joint: int | None = None):
        """Turn the grasped thing about `joint` (default: the part's own; its normal if it has none), holding
        the grasp pose relative to it. Returns (achieved rad, commanded palm pose to hold it turned)."""
        m, d = self.m, self.d
        j = p.joint if joint is None else joint
        if j is not None and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            axis, c = d.xaxis[j].copy(), d.xanchor[j].copy()
            q0 = self._joint_q(j)
            achieved = lambda: sign * (self._joint_q(j) - q0)     # noqa: E731
        else:
            pos0, R0p = p.frame()
            axis, c = R0p[:, 0].copy(), pos0
            achieved = lambda: sign * float(rot_error(R0p, p.frame()[1]) @ axis)   # noqa: E731
        P0, R0 = self.emb.target_pos.copy(), self.emb.target_rot.copy()
        mon = StallMonitor(window=40, min_cmd=0.1, ratio=0.2)   # a pinched key lags its command at first
        alpha, got = 0.0, 0.0
        pose = (P0, R0)
        for _ in range(int(angle / (TURN_SPEED * self.dt)) + 150):
            if got >= angle - 0.02:
                break
            alpha = min(alpha + TURN_SPEED * self.dt, angle, got + TURN_LEAD)
            Rr = rotvec_to_mat(axis * sign * alpha)
            pose = (c + Rr @ (P0 - c), Rr @ R0)
            prev = got
            self._step(self._cmd(pose[0], pose[1], grip))
            got = achieved()
            if mon.update(TURN_SPEED * self.dt, got - prev):
                break
        return got, pose

    def _carry(self, p: _Part, direction: str, pose, grip: float = -1.0, regrasp: bool = True) -> float:
        """Move the grasped part towards (pull) or away from (push) the body, the palm following the arc of
        the joint it rides on with a lead. If the grip slips off while the part is moving, take hold of it
        again (at most twice) and go on. Returns how far that joint moved (rad or m, >= 0)."""
        m, d = self.m, self.d
        j = p.carrier if p.carrier is not None else p.joint
        if j is None:
            return 0.0
        slide = m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
        axis, c = d.xaxis[j].copy(), d.xanchor[j].copy()
        pos, R = p.frame()
        want = R[:, 0] if direction == "pull" else -R[:, 0]
        v = axis if slide else np.cross(axis, pos - c)
        sgn = 1.0 if float(v @ want) >= 0 else -1.0
        goal, lead = (CARRY_SLIDE, LEAD_SLIDE) if slide else (CARRY_HINGE, LEAD_HINGE)
        moved_thr = MOVED_SLIDE if slide else MOVED_HINGE
        lo, hi = m.jnt_range[j]
        q0 = self._joint_q(j)
        if m.jnt_limited[j]:     # up to near the joint's limit, but always try (a pull against a stop stalls)
            goal = max(min(goal, (hi - q0 if sgn > 0 else q0 - lo) - 0.02), 2 * moved_thr)
        P0, R0 = pose
        s_ref = 0.0              # where the joint was when the held pose was taken
        best, best_k, lost, regrasps = 0.0, 0, 0, 0
        for k in range(2500):
            s = sgn * (self._joint_q(j) - q0)
            if s >= goal:
                break
            if s > best + (0.004 if slide else 0.01):
                best, best_k = s, k
            if k - best_k > CARRY_PATIENCE:
                break
            lost = lost + 1 if regrasp and not self._grip_ok() else 0
            if lost > 5 and s > moved_thr and regrasps < 2:
                regrasps += 1
                if not self._grasp_part(p):
                    break
                P0, R0 = self.emb.target_pos.copy(), self.emb.target_rot.copy()
                s_ref = sgn * (self._joint_q(j) - q0)
                best_k, lost = k, 0
                continue
            delta = sgn * (min(max(s, 0.0) + lead, goal + lead) - s_ref)
            if slide:
                self._step(self._cmd(P0 + axis * delta, R0, grip))
            else:
                Rr = rotvec_to_mat(axis * delta)
                self._step(self._cmd(c + Rr @ (P0 - c), Rr @ R0, grip))
        return max(0.0, sgn * (self._joint_q(j) - q0))

    def _press(self, p: _Part):
        pos, R = p.frame()
        Rg = self.emb.grasp_rot(R[:, 2], -R[:, 0], self.obs["palm_rot"])
        z = Rg[:, 2]
        self._goto(pos - 0.22 * z, Rg, -1.0, tol=0.04, rtol=0.2, steps=250)
        self._goto(p.frame()[0] - 0.1 * z, Rg, -1.0, tol=0.02, steps=300)
        self._goto(p.frame()[0] + PRESS_DEPTH * z, Rg, -1.0, tol=0.01, steps=150)
        return self.emb.target_pos.copy(), self.emb.target_rot.copy()

    def _go_home(self) -> None:
        """Open, back off along the palm normal, return to the home pose."""
        if self._keep:
            return                       # keep holding the open door
        self._hold(1.0, 25)
        back = self.obs["palm_pos"] - 0.1 * self.obs["palm_rot"][:, 2]
        self._goto(back, self.obs["palm_rot"].copy(), 1.0, tol=0.02, steps=120)
        self._goto(self.obs["palm_pos"] + [0, 0, 0.03], self.obs["palm_rot"].copy(), 1.0, tol=0.02, steps=40)
        self._goto(self.home[0], self.home[1], 1.0, tol=0.03, rtol=0.2, steps=300)
        self._holding_part = False

    # ------------------------------------------------------------------ interface: execute
    def execute(self, action: Action) -> Outcome:
        kind, eid = action[0], action[1]
        arg = action[2] if len(action) > 2 else None
        before_f, before_vis = self.features(), {e.id for e in self.entities() if e.kind == "object"}
        self._steps, t0 = 0, time.time()
        executed, stalled, notes = False, False, ""
        try:
            if self._keep:                         # let go of the door held open by the last action
                self._keep = False
                self._go_home()
            if self.unstable:
                notes = "unstable (reset the world)"
            elif action not in self.actions():
                notes = "not available"
            elif kind in ("pull", "push", "turn_pull", "turn_push", "press_push", "turn"):
                executed, stalled, notes = self._do_part(kind, self._part_by_id[eid], arg)
            elif kind == "pick":
                executed, stalled, notes = self._do_pick(eid)
            elif kind == "insert":
                executed, stalled, notes = self._do_insert(eid, self._part_by_id[arg])
            elif kind == "turn_held":
                executed, stalled, notes = self._do_turn_held(eid, int(arg))
            elif kind == "release":
                executed, stalled, notes = self._do_release(eid)
        except _Abort as e:
            notes = f"aborted: {e}"
            self.unstable = self.unstable or str(e) == "unstable"
            self.held = None if self.held and not self._holding(self._obj_by_id[self.held]) else self.held
        after = self.features()
        changed = {k: (before_f.get(k), v) for k, v in after.items() if before_f.get(k) != v}
        changed.update({k: (v, None) for k, v in before_f.items() if k not in after})
        revealed = [e for e in self.entities() if e.kind == "object" and e.id not in before_vis]
        self.last_wall = time.time() - t0
        return Outcome(tuple(action), executed, stalled, changed, revealed, self._steps * self.dt, notes)

    def _do_part(self, kind: str, p: _Part, sign) -> tuple[bool, bool, str]:
        if kind == "press_push" or not p.graspable:     # nothing to grasp: press a closed hand into it
            pose = self._press(p)
            self._holding_part = True
            moved = self._carry(p, "push", pose, regrasp=False)
            stalled = moved < self._moved_thr(p)
            self._go_home()
            return True, stalled, f"moved={moved:.3f}"
        if not self._grasp_part(p):
            self._go_home()
            return False, False, "grasp failed"
        self._holding_part = True
        pose = (self.emb.target_pos.copy(), self.emb.target_rot.copy())
        turned = None
        if kind in ("turn", "turn_pull", "turn_push"):
            turned, pose = self._turn(p, int(sign), angle=TURN_ANGLE if kind == "turn" else TURN_PROBE)
            if kind == "turn":
                stalled = turned < TURN_MOVED
                self._hold(-1.0, 10, *pose)
                self._go_home()
                return True, stalled, f"turned={turned:.2f}"
        moved = self._carry(p, "pull" if kind in ("pull", "turn_pull") else "push", pose)
        stalled = moved < self._moved_thr(p)
        self._go_home()
        return True, stalled, f"moved={moved:.3f}" + (f" turned={turned:.2f}" if turned is not None else "")

    def _moved_thr(self, p: _Part) -> float:
        j = p.carrier if p.carrier is not None else p.joint
        return MOVED_SLIDE if j is not None and self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE else MOVED_HINGE

    def _do_pick(self, oid: str) -> tuple[bool, bool, str]:
        k = self._obj_by_id[oid]
        d = self.d
        inside = self._inserted_in(k)
        if inside is not None:
            return self._withdraw(oid, k, inside)
        grip = lambda o: d.site_xpos[k["grip"]].copy()   # noqa: E731
        Rk = d.xmat[k["body"]].reshape(3, 3).copy()
        # from above; then along the object's long axis, unless it lies in the drawer (its front and walls
        # are in the way of that approach)
        approaches = [np.array([0, 0, -1.0])]
        if not self._under_rim(k):
            approaches.append(Rk[:, 0].copy())
        for approach in approaches:
            z0 = float(d.xpos[k["body"]][2])
            self._grasp_at(grip, d.xmat[k["body"]].reshape(3, 3)[:, 1].copy(), approach, standoff=0.12)
            pose = (self.emb.target_pos.copy(), self.emb.target_rot.copy())
            for _ in range(2):
                self._goto(self.obs["palm_pos"] + [0, 0, LIFT / 2], self.obs["palm_rot"].copy(), -1.0, tol=0.02,
                           steps=150)
            if float(d.xpos[k["body"]][2]) - z0 > LIFT / 2 and self._holding(k):
                self.held, self._put_back = oid, pose
                return True, False, "lifted"
            self._hold(1.0, 25)
            self._goto(self.obs["palm_pos"] - 0.1 * approach + [0, 0, 0.05], self.obs["palm_rot"].copy(), 1.0,
                       tol=0.02, steps=120)
        self._go_home()
        return False, False, "could not lift it"

    def _under_rim(self, k: dict) -> bool:
        """Is the object below the top of the drawer it lies in (a front grasp would hit the drawer walls)?"""
        m, d = self.m, self.d
        drawer = next(p for p in self.parts if p.name == "drawer")
        b = m.jnt_bodyid[drawer.joint]
        Rb = d.xmat[b].reshape(3, 3)
        p = Rb.T @ (d.xpos[k["body"]] - d.xpos[b])
        return bool(-0.3 < p[0] < 0.0 and abs(p[1]) < 0.2 and 0.0 < p[2] < 0.2)

    def _key_pose(self, k):
        return self.d.xpos[k["body"]].copy(), self.d.xmat[k["body"]].reshape(3, 3).copy()

    def _do_insert(self, oid: str, p: _Part) -> tuple[bool, bool, str]:
        k = self._obj_by_id[oid]
        m, d = self.m, self.d
        pos0, Rpart = p.frame()
        x = -Rpart[:, 0]                          # into the part
        slot = Rpart[:, 2]
        _, Rk = self._key_pose(k)
        # A flat blade fits either way round: keep the hand's body on the side of the joint the part rides
        # on (for a door: the hinge side, away from the jamb), else the pose nearer to the current one.
        cands = []
        ref = None
        if p.carrier is not None and m.jnt_type[p.carrier] == mujoco.mjtJoint.mjJNT_HINGE:
            ref = d.xanchor[p.carrier] - pos0
            ref[2] = 0.0
            ref /= max(np.linalg.norm(ref), 1e-9)
        for s in (1, -1):
            y = s * slot
            Rd = np.column_stack([x, y, np.cross(x, y)])
            Rp = Rd @ (Rk.T @ self.obs["palm_rot"])
            score = float(-Rp[:, 2] @ ref) if ref is not None else -float(np.linalg.norm(rot_error(self.obs["palm_rot"], Rp)))
            cands.append((score, Rd))
        Rd = max(cands, key=lambda c: c[0])[1]

        def aligned_cmd(tip_goal):
            pk, Rk_now = self._key_pose(k)
            Rp = Rd @ (Rk_now.T @ self.obs["palm_rot"])
            pos = self.emb.target_pos + (tip_goal - pk) + (Rd - Rk_now) @ (Rk_now.T @ (self.obs["palm_pos"] - pk))
            return self._cmd(pos, Rp, -1.0), pk, Rk_now

        aligned = False
        for goal_depth, tol, steps in ((-0.12, 0.02, 400), (-0.03, 0.004, 300)):
            for _ in range(steps):
                mouth = p.frame()[0]
                a, pk, Rk_now = aligned_cmd(mouth + goal_depth * x)
                aligned = (np.linalg.norm(pk - (mouth + goal_depth * x)) < tol and Rk_now[:, 0] @ x > 0.995
                           and abs(Rk_now[:, 1] @ Rd[:, 1]) > 0.99)
                if aligned:
                    break
                self._step(a)
        if not self._holding(k):
            self.held = None
            self._go_home()
            return False, False, "dropped it"
        mon = StallMonitor(window=30, min_cmd=0.01)
        depth = float((self._key_pose(k)[0] - p.frame()[0]) @ x)
        for _ in range(300):
            if depth >= INSERT_DEPTH - 0.005:      # all the way in (or it stops: see the monitor)
                break
            a, _, _ = aligned_cmd(p.frame()[0] + min(depth + 0.015, INSERT_DEPTH) * x)
            self._step(a)
            new = float((self._key_pose(k)[0] - p.frame()[0]) @ x)
            if mon.update(0.04 * self.dt, new - depth):
                depth = new
                break
            depth = new
        inside = self._inserted_in(k) is p
        return bool(aligned or inside), not inside, f"depth={depth:.3f}"

    def _do_turn_held(self, oid: str, sign: int) -> tuple[bool, bool, str]:
        k = self._obj_by_id[oid]
        p = self._inserted_in(k)
        if p is None:
            return self._roll_in_hand(k, sign) if self.held == oid else (False, False, "not in a part")
        if self.held != oid:                       # take hold of it where it is, from the front
            _, R = p.frame()
            Rk = self._key_pose(k)[1]
            self._grasp_at(lambda o: self.d.site_xpos[k["grip"]].copy(), Rk[:, 1].copy(), -R[:, 0])
            if not self._holding(k):
                self._go_home()
                return False, False, "grasp failed"
            self.held = oid
        turned, pose = self._turn(p, sign, angle=np.pi / 2 + 0.25)
        self._hold(-1.0, 10, *pose)
        return True, turned < TURN_MOVED, f"turned={turned:.2f}"

    def _roll_in_hand(self, k: dict, sign: int) -> tuple[bool, bool, str]:
        """Turn a free object in the hand: roll the palm about the object's long axis (through the grip)
        a quarter turn and back. It changes nothing but the time spent (unless the object slips out)."""
        R0, P0 = self.obs["palm_rot"].copy(), self.obs["palm_pos"].copy()
        axis = self._key_pose(k)[1][:, 0]
        c = self.d.site_xpos[k["grip"]].copy()
        Rr = rotvec_to_mat(axis * sign * HELD_ROLL)
        self._goto(c + Rr @ (P0 - c), Rr @ R0, -1.0, tol=0.01, rtol=0.1, steps=150)
        self._goto(P0, R0, -1.0, tol=0.01, rtol=0.1, steps=150)
        if not self._holding(k):
            self.held, self._put_back = None, None
            self._go_home()
            return False, False, "dropped it"
        return True, False, "turned in the hand"

    def _withdraw(self, oid: str, k: dict, p: _Part) -> tuple[bool, bool, str]:
        """Pull an inserted object back out along the part's axis (grasping it first if needed)."""
        _, R = p.frame()
        if self.held != oid:
            Rk = self._key_pose(k)[1]
            self._grasp_at(lambda o: self.d.site_xpos[k["grip"]].copy(), Rk[:, 1].copy(), -R[:, 0])
            if not self._holding(k):
                self._go_home()
                return False, False, "grasp failed"
            self.held = oid
        out = R[:, 0]
        P0, R0 = self.emb.target_pos.copy(), self.emb.target_rot.copy()
        for i in range(150):
            if self._inserted_in(k) is None and float((self._key_pose(k)[0] - p.frame()[0]) @ out) > 0.02:
                break
            self._step(self._cmd(P0 + out * min(0.08, 0.002 * (i + 1)), R0, -1.0))
        self._goto(self.obs["palm_pos"] + 0.06 * out, self.obs["palm_rot"].copy(), -1.0, tol=0.02, steps=100)
        free = self._inserted_in(k) is None
        if not self._holding(k):
            self.held = None
        return True, not free, "withdrawn" if free else "stuck"

    def _do_release(self, oid: str) -> tuple[bool, bool, str]:
        k = self._obj_by_id[oid]
        p = self._inserted_in(k)
        if p is None and self._put_back is not None:     # put it back where it was taken
            pos, R = self._put_back
            self._goto(pos + [0, 0, 0.05], R, -1.0, tol=0.02, steps=300)
            self._goto(pos + [0, 0, 0.01], R, -1.0, tol=0.01, steps=150)
        self._hold(1.0, 25)
        self.held, self._put_back = None, None
        if p is not None:
            _, R = p.frame()
            self._goto(self.obs["palm_pos"] + 0.1 * R[:, 0], self.obs["palm_rot"].copy(), 1.0, tol=0.02, steps=120)
        self._go_home()
        return True, False, "in part" if p is not None else "put down"
