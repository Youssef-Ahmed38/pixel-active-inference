"""Embodiments: many bodies behind one control interface, so the same upper levels drive any of them.

The door/drawer task should not be a Panda-only skill. Each embodiment here attaches itself to a scene
MjSpec (Menagerie models, attached with a name prefix) and exposes the same action and observation:

    action = [dx, dy, dz, wx, wy, wz, grasp]              palm-target velocity (m/s), angular velocity
                                                          (rad/s, world-frame axis-angle rate) and a
                                                          grasp command in [-1 closed, 1 open]
           or [dx, dy, dz, wx, wy, wz, g_1 .. g_nf]       one grasp command per finger instead

    obs    = palm_pos, palm_quat, palm_rot, palm_target_pos, opening (0 closed .. 1 open), finger_closure
             (per finger, 0..1), finger_q (all finger joints), finger_force (contact normal force summed
             per finger, N), n_fingers

The palm frame is the same convention for every body: origin at the grasp centre (where a bar sits in a
power grasp, or between the pads of a parallel gripper), z = palm normal / approach direction, x =
finger direction (multi-finger hands) or closing direction (parallel grippers), y = z x x, so a bar
handle is grasped with its axis along palm y.

The grasp command drives a per-hand synergy: every finger actuator moves along a line from an open to a
power-grasp posture (defined in actuator-length space, so tendon-coupled joints and 0..255 gripper
controls are handled alike). Per-finger commands move each finger along its own share of that line
(actuators shared by several fingers, as in the parallel grippers, take the mean). The postures were
tuned so a ~24 mm bar sits against the palm, inside the proximal phalanges; the G1's open posture is a
pre-curled hook, so its long straight fingers fit between a bar and the panel behind it.

Two families:
- floating hands (robotiq_2f85, allegro, leap, shadow): the hand hangs from a free-jointed "wrist"
  welded to a mocap body that carries the commanded palm pose. A weld to a mocap body is the most
  stable 6-DoF drive in MuJoCo (no joint chain, no gimbal lock, implicit constraint solve); its
  softness (solref) gives the compliance a hand needs when it meets a door. Gravity compensation holds
  the hand up and a leash keeps the target within a few cm of the palm, which bounds the force a
  blocked hand can build up.
- arms (panda_2f, g1_hands): joint position servos, driven by damped least-squares IK on the palm site
  (as in TabletopEnv), with a null-space pull towards the home posture. The G1 humanoid stands in front
  of the fixture with its pelvis welded to the world (free joint removed) and its legs held by their
  servos; only the right arm reaches, the right Dex3 hand grasps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from pai.envs.scene import MENAGERIE_DIR

GRASP_DIM = 1


# ---------------------------------------------------------------------------------------------- math
def quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def mat_from_quat(q) -> np.ndarray:
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def rotvec_to_mat(w) -> np.ndarray:
    w = np.asarray(w, float)
    q = np.zeros(4)
    ang = np.linalg.norm(w)
    if ang < 1e-12:
        return np.eye(3)
    mujoco.mju_axisAngle2Quat(q, w / ang, ang)
    return mat_from_quat(q)


def rot_error(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking R to R_target."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_target @ R.T).reshape(9))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def _axes(x, z) -> np.ndarray:
    """Rotation matrix with columns x, y = z x x, z (x is re-orthogonalised against z)."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def home_palm_rot(approach) -> np.ndarray:
    """Default palm orientation: palm facing along `approach`, palm y up (a vertical bar fits it)."""
    z = np.asarray(approach, float)
    y = np.array([0.0, 0.0, 1.0])
    return _axes(np.cross(y, z), z)


# ---------------------------------------------------------------------------------------------- specs
@dataclass
class Finger:
    name: str
    roots: list[str]              # root bodies of the finger subtree(s) in the Menagerie model
    actuators: list[str]          # synergy actuators of this finger (may be shared with other fingers)


@dataclass
class EmbodimentSpec:
    name: str
    xml: str                      # relative to MENAGERIE_DIR
    n_fingers: int
    human_like: bool
    palm_body: str                # Menagerie body the palm frame is fixed to
    palm_pos: tuple               # grasp centre in palm_body frame
    palm_x: tuple                 # palm-frame axes in palm_body frame (x: finger / closing direction,
    palm_z: tuple                 # z: palm normal)
    fingers: list[Finger]
    synergy: dict[str, tuple]     # actuator -> (open length, power-grasp length)
    holds: dict[str, float] = field(default_factory=dict)   # actuator -> fixed length (wrist, legs)
    grasp_roll: float = 0.0       # preferred rotation of the approach about the bar axis (rad)
    approach_hint: tuple | None = None   # world direction that picks the sign of grasp_roll
    notes: str = ""

    @property
    def path(self) -> Path:
        return MENAGERIE_DIR / self.xml

    def available(self) -> bool:
        return self.path.exists()


def _load_child(spec_def: EmbodimentSpec) -> mujoco.MjSpec:
    if not spec_def.available():
        raise FileNotFoundError(f"{spec_def.path} not found. Run `python scripts/fetch_assets.py`.")
    child = mujoco.MjSpec.from_file(str(spec_def.path))
    for key in list(child.keys):  # keyframes do not survive re-parenting (qpos sizes change)
        child.delete(key)
    return child


def _match_options(child: mujoco.MjSpec, parent: mujoco.MjSpec) -> None:
    """The scene's physics options win; copying them avoids attach-conflict warnings (e.g. LEAP's
    impratio=100)."""
    for k in ("impratio", "cone", "integrator", "timestep"):
        setattr(child.option, k, getattr(parent.option, k))


def _palm_rot_local(sd: EmbodimentSpec) -> np.ndarray:
    return _axes(sd.palm_x, sd.palm_z)


# ---------------------------------------------------------------------------------------------- base
class Embodiment:
    """Common interface. Lifecycle: attach(spec, anchor, approach) -> compile -> bind(model, data) ->
    reset() -> apply(action, dt) / observe()."""

    max_lin_vel = 0.3     # m/s
    max_ang_vel = 1.5     # rad/s
    leash = 0.06          # max distance of the commanded palm target from the actual palm (m)
    dt = 0.02             # control period, updated by apply()

    def __init__(self, sd: EmbodimentSpec):
        self.sd = sd
        self.name = sd.name
        self.prefix = f"{sd.name}/"
        self.n_fingers = sd.n_fingers

    # names
    def p(self, name: str) -> str:
        return self.prefix + name

    def available(self) -> bool:
        return self.sd.available()

    def action_dim(self, per_finger: bool = False) -> int:
        return 6 + (self.n_fingers if per_finger else GRASP_DIM)

    # ---------------------------------------------------------------- binding
    def attach(self, spec: mujoco.MjSpec, anchor: np.ndarray, approach: np.ndarray) -> None:
        raise NotImplementedError

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        m = self.model = model
        self.data = data
        sd = self.sd
        self.palm_site = m.site(self.p("palm")).id
        names = list(sd.synergy)
        self.syn_act = np.array([m.actuator(self.p(a)).id for a in names])
        self.syn_open = np.array([sd.synergy[a][0] for a in names], float)
        self.syn_closed = np.array([sd.synergy[a][1] for a in names], float)
        # length = ratio * ctrl for affine position actuators (ratio 1 for plain <position>)
        g, b = m.actuator_gainprm[:, 0], -m.actuator_biasprm[:, 1]
        self.act_ratio = np.where(np.abs(b) > 0, g / np.where(b == 0, 1, b), 1.0)
        self.hold_act = np.array([m.actuator(self.p(a)).id for a in sd.holds], int)
        self.hold_len = np.array(list(sd.holds.values()), float)
        # finger -> synergy actuator columns, finger -> body ids
        self.finger_cols = [[names.index(a) for a in f.actuators] for f in sd.fingers]
        self.body_finger = np.full(m.nbody, -1)
        own = np.zeros(m.nbody, bool)
        for i, f in enumerate(sd.fingers):
            for r in f.roots:
                rid = m.body(self.p(r)).id
                for bid in range(m.nbody):
                    if _is_descendant(m, bid, rid):
                        self.body_finger[bid] = i
        for bid in range(m.nbody):
            own[bid] = m.body(bid).name.startswith(self.prefix)
        self.own_body = own
        jnts = [j for j in range(m.njnt) if self.body_finger[m.jnt_bodyid[j]] >= 0
                and m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
        self.finger_qadr = np.array([m.jnt_qposadr[j] for j in jnts], int)
        self.finger_joint_names = [m.joint(j).name[len(self.prefix):] for j in jnts]

    # ---------------------------------------------------------------- control
    def reset(self, palm_offset: np.ndarray | None = None) -> None:
        raise NotImplementedError

    def _set_fingers(self, closure: np.ndarray) -> None:
        """closure: per synergy actuator in [0, 1]."""
        length = self.syn_open + closure * (self.syn_closed - self.syn_open)
        self.data.ctrl[self.syn_act] = length / self.act_ratio[self.syn_act]
        if len(self.hold_act):
            self.data.ctrl[self.hold_act] = self.hold_len / self.act_ratio[self.hold_act]

    def _preset_fingers(self, closure: float = 0.0) -> None:
        """Put the finger joints at a synergy posture directly (resets start with an open hand).
        Tendon-driven actuators spread their length evenly over the tendon's joints."""
        m, d = self.model, self.data
        length = self.syn_open + closure * (self.syn_closed - self.syn_open)
        for a, L in zip(self.syn_act, length):
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                joints, coef = [m.actuator_trnid[a, 0]], [1.0]
            elif m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON:
                t = m.actuator_trnid[a, 0]
                adr, num = m.tendon_adr[t], m.tendon_num[t]
                joints, coef = list(m.wrap_objid[adr:adr + num]), list(m.wrap_prm[adr:adr + num])
            else:
                continue
            for j, c in zip(joints, coef):
                lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf)
                d.qpos[m.jnt_qposadr[j]] = np.clip(L / (c * len(joints)), lo, hi)

    def _finger_command(self, grasp: np.ndarray) -> np.ndarray:
        """Grasp command(s) in [-1 closed, 1 open] -> closure per synergy actuator."""
        c = (1.0 - np.clip(np.asarray(grasp, float), -1, 1)) / 2
        if c.size == 1:
            return np.full(len(self.syn_act), float(c.reshape(-1)[0]))
        acc, cnt = np.zeros(len(self.syn_act)), np.zeros(len(self.syn_act))
        for i, cols in enumerate(self.finger_cols):  # actuators shared by fingers take the mean
            acc[cols] += c[i]
            cnt[cols] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), c.mean())

    def apply(self, action: np.ndarray, dt: float) -> None:
        a = np.asarray(action, float)
        self.dt = dt
        v = np.clip(a[:3], -self.max_lin_vel, self.max_lin_vel)
        w = a[3:6]
        n = np.linalg.norm(w)
        if n > self.max_ang_vel:
            w = w * self.max_ang_vel / n
        palm = self.data.site_xpos[self.palm_site]
        tgt = self.target_pos + v * dt
        off = tgt - palm
        d = np.linalg.norm(off)
        if d > self.leash:
            tgt = palm + off * self.leash / d
        self.target_pos = np.clip(tgt, self.ws_low, self.ws_high)
        self.target_rot = rotvec_to_mat(w * dt) @ self.target_rot
        self._set_fingers(self._finger_command(a[6:]))
        self._drive()

    def _drive(self) -> None:
        raise NotImplementedError

    def set_target(self, pos, rot) -> None:
        """Absolute palm target (for resets and tests; the agent uses the velocity action)."""
        self.target_pos = np.asarray(pos, float).copy()
        self.target_rot = np.asarray(rot, float).copy()
        self._drive()

    # ---------------------------------------------------------------- sensing
    def finger_forces(self) -> np.ndarray:
        """Contact normal force on each finger from anything that is not this body (N)."""
        m, d = self.model, self.data
        out = np.zeros(self.n_fingers)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for b, other in ((b1, b2), (b2, b1)):
                k = self.body_finger[b]
                if k >= 0 and not self.own_body[other]:
                    mujoco.mj_contactForce(m, d, i, f6)
                    out[k] += abs(f6[0])
        return out

    def finger_closure(self) -> np.ndarray:
        length = self.data.actuator_length[self.syn_act]
        span = self.syn_closed - self.syn_open
        c = np.clip((length - self.syn_open) / np.where(np.abs(span) < 1e-9, 1, span), 0, 1)
        return np.array([c[cols].mean() for cols in self.finger_cols])

    def observe(self) -> dict:
        d = self.data
        R = d.site_xmat[self.palm_site].reshape(3, 3).copy()
        closure = self.finger_closure()
        return {
            "palm_pos": d.site_xpos[self.palm_site].copy(),
            "palm_quat": quat_from_mat(R),
            "palm_rot": R,
            "palm_target_pos": self.target_pos.copy(),
            "opening": float(1.0 - closure.mean()),
            "finger_closure": closure,
            "finger_q": d.qpos[self.finger_qadr].copy(),
            "finger_force": self.finger_forces(),
            "n_fingers": self.n_fingers,
        }

    def grasp_rot(self, bar_axis: np.ndarray, approach: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Palm orientation that grasps a bar (axis `bar_axis`) approached along `approach`, using this
        body's preferred roll about the bar; of the two bar-axis signs, the one closer to `current`."""
        a = np.asarray(bar_axis, float) / np.linalg.norm(bar_axis)
        z = np.asarray(approach, float) - np.dot(approach, a) * a
        z /= np.linalg.norm(z)
        if self.sd.grasp_roll:
            cands = [rotvec_to_mat(a * s * self.sd.grasp_roll) @ z for s in (1, -1)]
            hint = np.asarray(self.sd.approach_hint if self.sd.approach_hint is not None else z, float)
            z = max(cands, key=lambda c: float(np.dot(c, hint)))
        best, best_err = None, np.inf
        for s in (1, -1):
            y = s * a
            R = np.column_stack([np.cross(y, z), y, z])
            err = np.linalg.norm(rot_error(current, R))
            if err < best_err:
                best, best_err = R, err
        return best


def _is_descendant(m: mujoco.MjModel, bid: int, root: int) -> bool:
    while True:
        if bid == root:
            return True
        if bid == 0:
            return False
        bid = m.body_parentid[bid]


# ---------------------------------------------------------------------------------------------- hands
class FloatingHand(Embodiment):
    """A hand on a 6-DoF wrist: free-jointed mount welded to a mocap body at the commanded palm pose."""

    weld_solref = (0.02, 1.0)   # timeconst 20 ms: stiff tracking, yet soft enough on contact
    reach = 0.6                 # workspace half-size around the home pose (m)
    home_standoff = 0.25        # home palm distance in front of the anchor (m)

    def attach(self, spec, anchor, approach):
        sd = self.sd
        child = _load_child(sd)
        cm = child.compile()
        cd = mujoco.MjData(cm)
        mujoco.mj_kinematics(cm, cd)
        pb = cm.body(sd.palm_body).id
        Rb = cd.xmat[pb].reshape(3, 3)
        palm_pos = cd.xpos[pb] + Rb @ np.asarray(sd.palm_pos, float)
        palm_rot = Rb @ _palm_rot_local(sd)
        # the mount frame IS the palm frame: attach the hand at the inverse palm pose
        inv_rot = palm_rot.T
        inv_pos = -inv_rot @ palm_pos

        self.home_pos = np.asarray(anchor, float) - self.home_standoff * np.asarray(approach, float)
        self.home_rot = home_palm_rot(approach)
        hq = quat_from_mat(self.home_rot)
        wb = spec.worldbody
        wb.add_body(name=self.p("target"), mocap=True, pos=self.home_pos, quat=hq)
        mount = wb.add_body(name=self.p("mount"), pos=self.home_pos, quat=hq)
        mount.add_freejoint(name=self.p("wrist"))
        mount.explicitinertial = True
        mount.mass = 0.1
        mount.inertia = [1e-4, 1e-4, 1e-4]
        mount.add_site(name=self.p("palm"), size=[0.008] * 3, rgba=[1, 0.3, 0.3, 0.6], group=4)
        frame = mount.add_frame(pos=inv_pos, quat=quat_from_mat(inv_rot))
        _match_options(child, spec)
        spec.attach(child, prefix=self.prefix, frame=frame)
        for b in spec.bodies:
            if b.name.startswith(self.prefix):
                b.gravcomp = 1.0
        spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=self.p("mount"), name2=self.p("target"),
                          data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], solref=list(self.weld_solref))
        self._tune(spec)

    def _tune(self, spec) -> None:
        """Per-model fixes after attaching (actuator strength etc.)."""

    def bind(self, model, data):
        super().bind(model, data)
        m = model
        self.mocap = m.body_mocapid[m.body(self.p("target")).id]
        j = m.joint(self.p("wrist"))
        self.free_qadr, self.free_vadr = j.qposadr[0], j.dofadr[0]
        self.ws_low, self.ws_high = self.home_pos - self.reach, self.home_pos + self.reach

    def reset(self, palm_offset=None):
        pos = self.home_pos + (0 if palm_offset is None else np.asarray(palm_offset, float))
        d = self.data
        d.qpos[self.free_qadr:self.free_qadr + 3] = pos
        d.qpos[self.free_qadr + 3:self.free_qadr + 7] = quat_from_mat(self.home_rot)
        d.qvel[self.free_vadr:self.free_vadr + 6] = 0
        self._preset_fingers(0.0)
        self._set_fingers(np.zeros(len(self.syn_act)))
        self.set_target(pos, self.home_rot)

    def _drive(self):
        self.data.mocap_pos[self.mocap] = self.target_pos
        self.data.mocap_quat[self.mocap] = quat_from_mat(self.target_rot)


class Robotiq2F85(FloatingHand):
    def _tune(self, spec):
        # Menagerie caps the tendon force at 5 N; a door handle needs a firmer pinch (~real 2F-85 range).
        a = spec.actuator(self.p("fingers_actuator"))
        a.forcerange = [-40, 40]
        a.gainprm[0] *= 4
        a.biasprm[1] *= 4
        a.biasprm[2] *= 4


class Allegro(FloatingHand):
    def _tune(self, spec):
        # kp = 1 Nm/rad is too soft to hold a handle while the wrist pulls: stiffen the finger servos.
        for a in spec.actuators:
            if a.name.startswith(self.prefix):
                a.gainprm[0] *= 3
                a.biasprm[1] *= 3


class Leap(FloatingHand):
    pass


class Shadow(FloatingHand):
    pass


# ---------------------------------------------------------------------------------------------- arms
class ArmHand(Embodiment):
    """A fixed-base arm with a hand: joint position servos + damped least-squares IK on the palm site."""

    arm_joints: list[str] = []
    seed_q: np.ndarray = np.zeros(0)
    ik_damping = 0.05
    ik_rot_weight = 1.0
    ik_nullspace_gain = 0.05
    max_joint_vel = 2.0
    solve_home = True     # IK the seed posture to a palm pose facing the fixture at reset

    def base_pose(self, anchor, approach) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def prepare_child(self, child: mujoco.MjSpec) -> None:
        """Model edits before attaching (sites, removed joints, strength)."""

    def attach(self, spec, anchor, approach):
        sd = self.sd
        child = _load_child(sd)
        child.body(sd.palm_body).add_site(name="palm", pos=list(sd.palm_pos), quat=quat_from_mat(_palm_rot_local(sd)),
                                          size=[0.008] * 3, rgba=[1, 0.3, 0.3, 0.6], group=4)
        pos, quat = self.base_pose(np.asarray(anchor, float), np.asarray(approach, float))
        self.base_pos = pos
        self.prepare_child(child)
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        _match_options(child, spec)
        spec.attach(child, prefix=self.prefix, frame=frame)
        self.anchor, self.approach = np.asarray(anchor, float), np.asarray(approach, float)
        self.home_pos = self.anchor - 0.2 * self.approach
        self.home_rot = home_palm_rot(approach)

    def bind(self, model, data):
        super().bind(model, data)
        m = model
        self.qadr = np.array([m.joint(self.p(j)).qposadr[0] for j in self.arm_joints])
        self.vadr = np.array([m.joint(self.p(j)).dofadr[0] for j in self.arm_joints])
        self.arm_act = np.array([m.actuator(self.p(a)).id for a in self.arm_actuators])
        self.jrange = np.array([m.joint(self.p(j)).range for j in self.arm_joints])
        self.home_q = self._solve_home() if self.solve_home else np.asarray(self.seed_q, float)
        self.ws_low, self.ws_high = self.home_pos - 0.8, self.home_pos + 0.8

    @property
    def arm_actuators(self) -> list[str]:
        return self.arm_joints

    def _fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        d = self.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.palm_site].copy(), d.site_xmat[self.palm_site].reshape(3, 3).copy()

    def _ik_step(self, q, pos_t, rot_t, q_rest):
        m, d = self.model, self.data
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        pos = d.site_xpos[self.palm_site]
        rot = d.site_xmat[self.palm_site].reshape(3, 3)
        err = np.concatenate([pos_t - pos, self.ik_rot_weight * rot_error(rot, rot_t)])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.palm_site)
        J = np.vstack([jp[:, self.vadr], self.ik_rot_weight * jr[:, self.vadr]])
        dq = J.T @ np.linalg.solve(J @ J.T + self.ik_damping ** 2 * np.eye(6), err)
        null = np.eye(len(q)) - np.linalg.pinv(J) @ J
        dq += null @ (self.ik_nullspace_gain * (q_rest - q))
        return dq

    def _solve_home(self) -> np.ndarray:
        saved = self.data.qpos.copy()
        q = np.asarray(self.seed_q, float).copy()
        # palm y up or down is the same grasp: take the one closer to the seed posture
        _, R0 = self._fk(q)
        flip = self.home_rot @ np.diag([-1.0, -1.0, 1.0])
        if np.linalg.norm(rot_error(R0, flip)) < np.linalg.norm(rot_error(R0, self.home_rot)):
            self.home_rot = flip
        for _ in range(300):
            q = np.clip(q + np.clip(self._ik_step(q, self.home_pos, self.home_rot, self.seed_q), -0.1, 0.1),
                        self.jrange[:, 0], self.jrange[:, 1])
        self.data.qpos[:] = saved
        return q

    def reset(self, palm_offset=None):
        d = self.data
        d.qpos[self.qadr] = self.home_q
        d.ctrl[self.arm_act] = self.home_q
        self._preset_fingers(0.0)
        self._set_fingers(np.zeros(len(self.syn_act)))
        pos, rot = self._fk(self.home_q)
        if palm_offset is not None:
            pos = pos + np.asarray(palm_offset, float)
        self.target_pos, self.target_rot = pos, rot
        self._drive()

    def _drive(self):
        """IK on the commanded joint targets (not the measured pose), as in TabletopEnv."""
        d = self.data
        q_cmd = d.ctrl[self.arm_act].copy()
        saved = d.qpos.copy()
        dq = self._ik_step(q_cmd, self.target_pos, self.target_rot, self.home_q)
        d.qpos[:] = saved
        mujoco.mj_kinematics(self.model, d)
        lim = self.max_joint_vel * self.dt
        d.ctrl[self.arm_act] = np.clip(q_cmd + np.clip(dq, -lim, lim), self.jrange[:, 0], self.jrange[:, 1])


class Panda2F(ArmHand):
    """Menagerie Panda with its parallel gripper (the TabletopEnv robot), on a pedestal."""

    arm_joints = [f"joint{i}" for i in range(1, 8)]
    seed_q = np.array([0.0, -0.2, 0.0, -2.1, 0.0, 3.47, 0.785])   # hand horizontal, pointing forward

    @property
    def arm_actuators(self):
        return [f"actuator{i}" for i in range(1, 8)]

    def base_pose(self, anchor, approach):
        base = anchor - 0.82 * approach   # the palm faces the handle from a forward-reaching posture
        base[2] = anchor[2] - 0.7
        return base, quat_from_mat(_axes(approach, [0, 0, 1]))

    def prepare_child(self, child):
        for b in child.bodies:
            b.gravcomp = 1.0
        # x40 grip stiffness: ~24 N per finger on a 24 mm bar (Menagerie gives 0.6 N; the real hand ~70 N)
        g = child.actuator("actuator8")
        g.gainprm[0] *= 40
        g.biasprm[1] *= 40
        g.biasprm[2] *= 40
        # pedestal under the base
        h = max(float(self.base_pos[2]), 0.01) / 2
        child.body("link0").add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.1, h, 0],
                                     pos=[0, 0, -h], rgba=[0.3, 0.3, 0.32, 1])


class G1Hands(ArmHand):
    """Unitree G1 humanoid with Dex3 hands, standing in front of the fixture; right arm reaches."""

    arm_joints = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
    # forearm raised forward, hand clear of the hip (the Dex3 thumb is weak: 1.4 Nm, a hip contact stalls it)
    seed_q = np.array([-0.1, -0.3, 0.0, 1.3, 0.0, 0.0, 0.0])
    ik_rot_weight = 0.5      # 7 joints but a short, limited wrist: position first
    solve_home = False       # this posture already holds the palm sideways, as the grasp wants

    def base_pose(self, anchor, approach):
        side = np.cross([0, 0, 1], approach)   # robot's left
        base = anchor - 0.36 * approach + 0.17 * side
        base[2] = 0.0
        return base, quat_from_mat(_axes(approach, [0, 0, 1]))

    def prepare_child(self, child):
        child.delete(child.joint("floating_base_joint"))   # pelvis welded to the world
        for b in child.bodies:
            if "right_" in b.name and ("shoulder" in b.name or "elbow" in b.name or "wrist" in b.name
                                       or "hand" in b.name):
                b.gravcomp = 1.0

    def bind(self, model, data):
        super().bind(model, data)
        # hold everything that is neither the reaching arm nor the synergy at the stand posture
        m = model
        stand = {"left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 1.28}
        used = set(self.arm_act) | set(self.syn_act)
        self.hold_act = np.array([a for a in range(m.nu) if m.actuator(a).name.startswith(self.prefix)
                                  and a not in used], int)
        self.hold_len = np.array([stand.get(m.actuator(a).name[len(self.prefix):], 0.0) for a in self.hold_act])

    def reset(self, palm_offset=None):
        d = self.data
        for a, v in zip(self.hold_act, self.hold_len):
            d.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[a, 0]]] = v
        super().reset(palm_offset)


# ---------------------------------------------------------------------------------------------- registry
def _allegro_finger(f):
    return Finger(f, [f"{f}_base"], [f"{f}a1", f"{f}a2", f"{f}a3"])


ALLEGRO_CLOSED = {"a1": 1.0, "a2": 1.1, "a3": 0.9}
LEAP_CLOSED = {"mcp": 1.1, "pip": 1.0, "dip": 0.8}
SHADOW_CLOSED = {"J3": 1.2, "J0": 2.0}

SPECS: dict[str, EmbodimentSpec] = {
    "panda_2f": EmbodimentSpec(
        "panda_2f", "franka_emika_panda/panda.xml", 2, False,
        palm_body="hand", palm_pos=(0, 0, 0.1034), palm_x=(0, 1, 0), palm_z=(0, 0, 1),
        fingers=[Finger("left", ["left_finger"], ["actuator8"]), Finger("right", ["right_finger"], ["actuator8"])],
        synergy={"actuator8": (0.04, 0.0)},
        notes="7-DoF arm on a pedestal, DLS IK; parallel gripper (tendon, 0..255 ctrl)"),
    "robotiq_2f85": EmbodimentSpec(
        "robotiq_2f85", "robotiq_2f85/2f85.xml", 2, False,
        palm_body="base", palm_pos=(0, 0, 0.15), palm_x=(0, 1, 0), palm_z=(0, 0, 1),
        fingers=[Finger("left", ["left_driver", "left_spring_link"], ["fingers_actuator"]),
                 Finger("right", ["right_driver", "right_spring_link"], ["fingers_actuator"])],
        synergy={"fingers_actuator": (0.0, 0.8)},
        notes="floating adaptive parallel gripper, one tendon actuator"),
    "allegro": EmbodimentSpec(
        "allegro", "wonik_allegro/right_hand.xml", 4, False,
        palm_body="palm", palm_pos=(0.03, 0.0, 0.045), palm_x=(0, 0, 1), palm_z=(1, 0, 0),
        fingers=[_allegro_finger("ff"), _allegro_finger("mf"), _allegro_finger("rf"),
                 Finger("th", ["th_base"], ["tha0", "tha1", "tha2", "tha3"])],
        synergy={**{f"{f}{k}": (0.0, v) for f in ("ffa", "mfa", "rfa") for k, v in
                    (("1", ALLEGRO_CLOSED["a1"]), ("2", ALLEGRO_CLOSED["a2"]), ("3", ALLEGRO_CLOSED["a3"]))},
                 "tha0": (0.9, 1.3), "tha1": (0.2, 0.5), "tha2": (0.0, 0.9), "tha3": (0.0, 0.9)},
        notes="floating 4-finger hand (3 fingers + thumb), 16 joint servos"),
    "leap": EmbodimentSpec(
        "leap", "leap_hand/right_hand.xml", 4, False,
        palm_body="palm", palm_pos=(0.035, -0.037, -0.053), palm_x=(1, 0, 0), palm_z=(0, 0, -1),
        fingers=[Finger(f, [f"{f}_bs"], [f"{f}_mcp_act", f"{f}_pip_act", f"{f}_dip_act"]) for f in ("if", "mf", "rf")]
        + [Finger("th", ["th_mp"], ["th_cmc_act", "th_axl_act", "th_mcp_act", "th_ipl_act"])],
        synergy={**{f"{f}_{k}_act": (0.0, v) for f in ("if", "mf", "rf") for k, v in LEAP_CLOSED.items()},
                 "th_cmc_act": (0.3, 1.4), "th_axl_act": (0.3, 1.0), "th_mcp_act": (0.0, 0.5), "th_ipl_act": (0.0, 0.6)},
        notes="floating 4-finger hand (3 fingers + thumb), 16 joint servos"),
    "shadow": EmbodimentSpec(
        "shadow", "shadow_hand/right_hand.xml", 5, True,
        palm_body="rh_palm", palm_pos=(0.0, -0.035, 0.085), palm_x=(0, 0, 1), palm_z=(0, -1, 0),
        fingers=[Finger(f.lower(), [f"rh_{f.lower()}knuckle" if f != "LF" else "rh_lfmetacarpal"],
                        [f"rh_A_{f}J3", f"rh_A_{f}J0"]) for f in ("FF", "MF", "RF", "LF")]
        + [Finger("th", ["rh_thbase"], ["rh_A_THJ5", "rh_A_THJ4", "rh_A_THJ2", "rh_A_THJ1"])],
        synergy={**{f"rh_A_{f}{k}": (0.0, v) for f in ("FF", "MF", "RF", "LF") for k, v in SHADOW_CLOSED.items()},
                 "rh_A_THJ5": (0.0, 0.17), "rh_A_THJ4": (0.4, 1.2), "rh_A_THJ2": (0.0, 0.61), "rh_A_THJ1": (0.0, 0.52)},
        holds={"rh_A_WRJ2": 0.0, "rh_A_WRJ1": 0.0},
        notes="floating anthropomorphic 5-finger hand (tendon-coupled J1/J2), wrist joints held"),
    "g1_hands": EmbodimentSpec(
        "g1_hands", "unitree_g1/g1_with_hands.xml", 3, True,
        palm_body="right_wrist_yaw_link", palm_pos=(0.13, 0.035, 0.0), palm_x=(1, 0, 0), palm_z=(0, 1, 0),
        fingers=[Finger("thumb", ["right_hand_thumb_0_link"], ["right_hand_thumb_1_joint", "right_hand_thumb_2_joint"]),
                 Finger("index", ["right_hand_index_0_link"], ["right_hand_index_0_joint", "right_hand_index_1_joint"]),
                 Finger("middle", ["right_hand_middle_0_link"], ["right_hand_middle_0_joint", "right_hand_middle_1_joint"])],
        synergy={"right_hand_thumb_1_joint": (0.7, -0.6), "right_hand_thumb_2_joint": (0.0, -0.8),
                 "right_hand_index_0_joint": (0.4, 1.3), "right_hand_index_1_joint": (0.7, 1.5),
                 "right_hand_middle_0_joint": (0.4, 1.3), "right_hand_middle_1_joint": (0.7, 1.5)},
        grasp_roll=np.pi / 2, approach_hint=(0.0, 1.0, -1.0),
        notes="humanoid, pelvis fixed, right arm DLS IK, 3-finger Dex3 hand; grasps bars from the side/above"),
}

EMBODIMENTS: dict[str, type[Embodiment]] = {
    "panda_2f": Panda2F,
    "robotiq_2f85": Robotiq2F85,
    "allegro": Allegro,
    "leap": Leap,
    "shadow": Shadow,
    "g1_hands": G1Hands,
}


def make_embodiment(name: str) -> Embodiment:
    return EMBODIMENTS[name](SPECS[name])


def available_embodiments() -> list[str]:
    return [n for n in EMBODIMENTS if SPECS[n].available()]
