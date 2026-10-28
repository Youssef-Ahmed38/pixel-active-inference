"""The Unitree G1 humanoid body for Stage 2 (L0: the only body-specific code).

Body choice: the Menagerie ``unitree_g1`` MJX variant (``scene_mjx.xml``). It is already in
``third_party`` (scripts/fetch_assets.py), it is the humanoid used by the door/drawer
embodiments, and its MJX variant has capsule colliders with explicit contact pairs, which makes
CPU stepping fast (measured ~25k control steps/s for 2048 bodies on 6 threads with
``mujoco.rollout``). Position servos (kp/kv per joint) are the "muscles"; the learned controller
sets the 12 leg servo targets. Waist and arms are held at their default posture by the same
servos (a stated simplification, not learned).

The body model adds four sensors: world position and linear velocity of each sole site, used for
foot contact (height) and slip (velocity in contact).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
G1_DIR = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1"
SCENE = G1_DIR / "scene_mjx.xml"

LEG_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)
FEET = ("left_foot", "right_foot")
PUSH_BODY = "torso_link"
STATE_SPEC = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
CONTROL_SPEC = int(mujoco.mjtState.mjSTATE_CTRL) | int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)


def build_spec(scene: Path = SCENE) -> mujoco.MjSpec:
    if not Path(scene).is_file():
        raise FileNotFoundError(f"{scene} missing; run scripts/fetch_assets.py")
    spec = mujoco.MjSpec.from_file(str(scene))
    for foot in FEET:
        spec.add_sensor(name=f"{foot}_pos", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=foot)
        spec.add_sensor(name=f"{foot}_vel", type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=foot)
    return spec


@dataclass
class Randomization:
    """Domain randomisation ranges (multiplicative unless stated)."""
    friction: tuple[float, float] = (0.5, 1.25)
    mass: tuple[float, float] = (0.9, 1.1)
    torso_extra_kg: tuple[float, float] = (-1.0, 2.0)
    kp: tuple[float, float] = (0.9, 1.1)


@dataclass
class G1Body:
    model: mujoco.MjModel
    leg_qpos: np.ndarray = field(init=False)
    leg_qvel: np.ndarray = field(init=False)
    leg_ctrl: np.ndarray = field(init=False)
    default_qpos: np.ndarray = field(init=False)
    default_ctrl: np.ndarray = field(init=False)

    def __post_init__(self):
        m = self.model
        self.leg_qpos = np.array([m.jnt_qposadr[m.joint(j).id] for j in LEG_JOINTS])
        self.leg_qvel = np.array([m.jnt_dofadr[m.joint(j).id] for j in LEG_JOINTS])
        self.leg_ctrl = np.array([m.actuator(j).id for j in LEG_JOINTS])
        key = m.key("knees_bent").id
        self.default_qpos = m.key_qpos[key].copy()
        self.default_ctrl = m.key_ctrl[key].copy()
        self.default_leg = self.default_qpos[self.leg_qpos].copy()
        self.foot_pos_adr = np.array([m.sensor(f"{f}_pos").adr[0] for f in FEET])
        self.foot_vel_adr = np.array([m.sensor(f"{f}_vel").adr[0] for f in FEET])
        self.push_body = m.body(PUSH_BODY).id
        self.total_mass = float(m.body_mass.sum())
        self.nstate = mujoco.mj_stateSize(m, STATE_SPEC)
        self.ncontrol = mujoco.mj_stateSize(m, CONTROL_SPEC)
        self.qpos0 = 1                      # state layout: [time, qpos, qvel]
        self.qvel0 = 1 + m.nq
        self.leg_lo = m.jnt_range[[m.joint(j).id for j in LEG_JOINTS], 0].copy()
        self.leg_hi = m.jnt_range[[m.joint(j).id for j in LEG_JOINTS], 1].copy()
        self.foot_floor_pairs = np.array(
            [i for i in range(m.npair) if "foot" in (m.pair(i).name or "") and "floor" in (m.pair(i).name or "")])

    def default_state(self) -> np.ndarray:
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.default_qpos
        d.ctrl[:] = self.default_ctrl
        mujoco.mj_forward(self.model, d)
        s = np.empty(self.nstate)
        mujoco.mj_getState(self.model, d, s, STATE_SPEC)
        return s


WALL_BIT = 2   # collision bit shared only by walls and the body's collision capsules (as in the maze)
FAR = 100.0    # where an unused training wall is parked


def add_training_walls(spec: mujoco.MjSpec, n: int, half: float = 0.5, height: float = 0.5) -> None:
    """``n`` solid 1 m wall blocks for wall-contact practice, parked far away; ``place_walls`` moves
    them per dynamics variant. The body's collision capsules (those paired with the floor) get the
    wall bit, so floor and self contacts are exactly as before."""
    for i in range(n):
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, height / 2],
                                pos=[FAR + 3 * i, FAR, height / 2], contype=0, conaffinity=WALL_BIT,
                                rgba=[0.42, 0.45, 0.52, 1.0], name=f"train_wall_{i}")
    for pair in spec.pairs:
        names = {pair.geomname1, pair.geomname2}
        if "floor" in names:
            (name,) = names - {"floor"}
            spec.geom(name).contype = WALL_BIT


def load_body(scene: Path = SCENE, walls: int = 0) -> G1Body:
    spec = build_spec(scene)
    if walls:
        add_training_walls(spec, walls)
    return G1Body(spec.compile())


def place_walls(models: list[mujoco.MjModel], rng: np.random.Generator, n: int, frac: float,
                radius: tuple[float, float]) -> list[list[float]]:
    """In a fraction ``frac`` of the variants put the ``n`` training walls at random places
    ``radius`` metres from the start (centre distance); elsewhere they stay parked far away.
    Returns the placements (for the run log)."""
    layout = []
    for m in models:
        here = []
        use = rng.random() < frac
        for i in range(n):
            g = m.geom(f"train_wall_{i}").id
            if use:
                r, a, yaw = rng.uniform(*radius), rng.uniform(-np.pi, np.pi), rng.uniform(0, np.pi / 2)
                m.geom_pos[g, :2] = [r * np.cos(a), r * np.sin(a)]
                m.geom_quat[g] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
                here.append([round(float(r), 2), round(float(a), 2)])
        layout.append(here)
    return layout


def randomized_models(body: G1Body, n: int, rng: np.random.Generator,
                      ranges: Randomization | None = None) -> list[mujoco.MjModel]:
    """``n`` dynamics variants with the same size signature (for ``mujoco.rollout``)."""
    import copy
    ranges = ranges or Randomization()
    out = []
    for _ in range(n):
        m = copy.deepcopy(body.model)
        f = rng.uniform(*ranges.friction)
        m.pair_friction[body.foot_floor_pairs, :2] *= f
        s = rng.uniform(*ranges.mass, size=m.nbody)
        s[0] = 1.0
        m.body_mass[:] *= s
        m.body_inertia[:] *= s[:, None]
        m.body_mass[body.push_body] = max(1.0, m.body_mass[body.push_body] + rng.uniform(*ranges.torso_extra_kg))
        k = rng.uniform(*ranges.kp, size=m.nu)
        m.actuator_gainprm[:, 0] *= k
        m.actuator_biasprm[:, 1] *= k
        out.append(m)
    return out
