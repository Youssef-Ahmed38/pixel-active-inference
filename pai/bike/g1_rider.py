"""The Unitree G1 riding a realistic bicycle (Stage 2 body on the Stage 7 bike).

The 24-inch bicycle of ``pai/bike/real_bike.py`` on its road scene; the G1 (MJX model,
33 kg, position servos on all 29 joints) is mounted on it:

- pelvis fixed to the seat (the G1's floating base removed: it sits);
- palms held to the handlebar grips (equality ``connect``), so the hands turn with the steering;
- soles held to the pedals of a crank geared to the rear wheel (crank angle = 0.5 x wheel angle),
  so the legs pedal as the bike rolls; leg servos are soft so the pedals move the legs.

Actuation stays the bicycle's (rear-wheel drive torque and steering servo); the G1's waist roll
servo takes the place of the old lean joint. The bike's balance therefore carries the G1's real mass
and inertia (48 kg in total). Stated simplifications: the drive torque acts at the wheel (the legs
follow the pedals rather than producing the torque), and steering is a servo on the fork that the
hands hold rather than steering produced by the arms; the G1 does not collide with the bike.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from pai.bike.real_bike import R_TYRE

ROOT = Path(__file__).resolve().parents[2]
G1_XML = ROOT / "third_party" / "mujoco_menagerie" / "unitree_g1" / "g1_mjx.xml"

# seated pose targets for the G1's servos (rad); legs are soft and follow the pedals
POSE = {
    "hip_pitch_joint": -1.25, "knee_joint": 1.35, "ankle_pitch_joint": -0.25,
    "waist_pitch_joint": 0.35,
    "shoulder_pitch_joint": -0.9, "shoulder_roll_joint": 0.0, "elbow_joint": 0.45,
}
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
GEAR = 0.5            # crank angle per wheel angle (as in pai/bike/real_bike.py)


def build_spec(scenery=True, seed: int = 0) -> mujoco.MjSpec:
    """The realistic bicycle (``pai/bike/real_bike.py``) with the G1 seated on it."""
    from pai.bike import real_bike as rb
    bike = rb.build_spec(scenery=scenery, seed=seed)
    frame = bike.body("frame")
    # the saddle point in the frame body's coordinates (the frame body sits at the bottom bracket)
    post_top = rb.SEAT_TOP + 0.10 * np.array([-math.cos(math.radians(73)), math.sin(math.radians(73))])
    saddle = post_top + np.array([0.0, 0.05]) - rb.BB
    g1 = mujoco.MjSpec.from_file(str(G1_XML))
    g1.delete(g1.joint("floating_base_joint"))
    g1.body("pelvis").pos = [0, 0, 0]          # the model stands it at 0.793 m; here it sits on the saddle
    seat = frame.add_frame(pos=[float(saddle[0]) - 0.02, 0, float(saddle[1]) + 0.06])
    seat.attach_body(g1.body("pelvis"), "g1_", "")
    for a, b in (("g1_left_palm", "grip_l"), ("g1_right_palm", "grip_r"),
                 ("g1_left_foot", "pedal_l"), ("g1_right_foot", "pedal_r")):
        bike.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE,
                          name1=a, name2=b, solref=[0.02, 1.0])
    return bike


class G1BikeEnv:
    """Same interface as ``BicycleEnv`` (``reset``, ``sensation``, ``step``, ``fallen``, ``travelled``)."""

    domain = "bike_g1"

    def __init__(self, distance=10.0, control_dt=.02, max_seconds=10.0, leg_kp=8.0, scenery=True,
                 arm_kp=None, steer_kp=None):
        # arm_kp: soft arm servos so the arms follow the handlebar (None keeps the stiff arms of the
        # straight-riding results, which hold the steering to about +-0.24 rad)
        self.model = build_spec(scenery).compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.distance, self.max_seconds = distance, max_seconds
        self.substeps = max(1, round(control_dt / m.opt.timestep))
        self.control_dt = self.substeps * m.opt.timestep
        self.a_pedal, self.a_steer = m.actuator("pedal").id, m.actuator("steer").id
        self.a_lean = m.actuator("g1_waist_roll_joint").id
        self.g1_act = [i for i in range(m.nu) if m.actuator(i).name.startswith("g1_")]
        self.pose = np.zeros(m.nu)
        for i in self.g1_act:
            name = m.actuator(i).name[len("g1_"):]
            for key, val in POSE.items():
                if name.endswith(key):
                    self.pose[i] = val
            if any(name.startswith(side + "_" + leg) for side in ("left", "right") for leg in LEG):
                kp = leg_kp
                m.actuator_gainprm[i, 0] = kp
                m.actuator_biasprm[i, 1] = -kp
                m.actuator_biasprm[i, 2] = -0.1 * kp
            if arm_kp is not None and any(j in name for j in ("shoulder", "elbow", "wrist")):
                m.actuator_gainprm[i, 0] = arm_kp
                m.actuator_biasprm[i, 1] = -arm_kp
                m.actuator_biasprm[i, 2] = -0.1 * arm_kp
        if steer_kp is not None:       # a stronger steering servo (the arms resist the handlebar)
            m.actuator_gainprm[self.a_steer, 0] = steer_kp
            m.actuator_biasprm[self.a_steer, 1] = -steer_kp
        self.steer_q = m.jnt_qposadr[m.joint("steering").id]
        self.rear = m.joint("rear_spin")
        self.front = m.joint("front_spin")
        self.lean_q = m.jnt_qposadr[m.joint("g1_waist_roll_joint").id]
        self.ik_q, self.ik_gap = self._solve_pose()
        for i in self.g1_act:
            jid = m.actuator_trnid[i, 0]
            if jid in self.ik_q:
                self.pose[i] = self.ik_q[jid]
        self.reset()

    def _solve_pose(self):
        """Seated posture with palms on the grips and soles on the pedals (bike upright, crank at 0):
        least squares over the G1's joint angles within their limits, starting from ``POSE``."""
        from scipy.optimize import least_squares
        m = self.model
        d = mujoco.MjData(m)
        joints = [j for j in range(m.njnt) if m.joint(j).name.startswith("g1_")
                  and m.joint(j).name not in ("g1_waist_yaw_joint", "g1_waist_roll_joint")]
        adr = [m.jnt_qposadr[j] for j in joints]
        lo = m.jnt_range[joints, 0]
        hi = m.jnt_range[joints, 1]
        x0 = np.zeros(len(joints))
        for k, j in enumerate(joints):
            for key, val in POSE.items():
                if m.joint(j).name.endswith(key):
                    x0[k] = val
        x0 = np.clip(x0, lo + 1e-3, hi - 1e-3)
        pairs = [(m.site(a).id, m.site(b).id) for a, b in (("g1_left_palm", "grip_l"), ("g1_right_palm", "grip_r"),
                                                           ("g1_left_foot", "pedal_l"), ("g1_right_foot", "pedal_r"))]

        def residual(x):
            d.qpos[:] = m.qpos0
            d.qpos[adr] = x
            mujoco.mj_kinematics(m, d)
            gaps = [d.site_xpos[a] - d.site_xpos[b] for a, b in pairs]
            return np.concatenate(gaps + [0.03 * (x - x0)])

        sol = least_squares(residual, x0, bounds=(lo, hi), max_nfev=400)
        gap = np.sqrt(np.sum(residual(sol.x)[:12].reshape(4, 3) ** 2, axis=1))
        return {j: float(v) for j, v in zip(joints, sol.x)}, gap

    def _settle(self, seconds=1.5):
        """Hold the bike still and upright while the G1 settles into the seat, bars and pedals."""
        d = self.data
        root = d.qpos[:7].copy()
        for _ in range(int(seconds / self.model.opt.timestep)):
            d.ctrl[:] = self.pose
            mujoco.mj_step(self.model, d)
            d.qpos[:7] = root
            d.qvel[:6] = 0.0

    def reset(self, seed=0, speed=4.0, roll=None, start_xy=None, yaw=0.0):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for j, val in self.ik_q.items():
            d.qpos[m.jnt_qposadr[j]] = val
        mujoco.mj_forward(m, d)
        self._settle(0.5)
        rng = np.random.default_rng(seed)
        angle = float(rng.uniform(-.03, .03)) if roll is None else float(roll)
        # heading ``yaw`` (about z), then the initial tilt ``angle`` (about the bike's own x axis)
        cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(angle / 2), math.sin(angle / 2)
        d.qpos[3:7] = [cy * cr, cy * sr, sy * sr, sy * cr]
        if start_xy is not None:
            d.qpos[0:2] = start_xy
        d.qvel[:] = 0.0
        d.qvel[0], d.qvel[1] = speed * math.cos(yaw), speed * math.sin(yaw)
        for j in (self.rear, self.front):
            d.qvel[j.dofadr[0]] = speed / R_TYRE
        d.qvel[m.jnt_dofadr[m.joint("crank").id]] = GEAR * speed / R_TYRE
        d.time = 0.0
        self.origin = d.qpos[:2].copy()
        self.z0 = float(d.qpos[2])
        mujoco.mj_forward(m, d)
        return self.sensation()

    def sensation(self):
        d = self.data
        w, x, y, z = d.sensor("orientation").data
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, d.sensor("angular_velocity").data[0], d.sensor("velocity").data[0],
                         d.qpos[self.steer_q], d.qpos[self.lean_q], yaw], dtype=float)

    @property
    def fallen(self):
        # roll beyond 0.7 rad, or the frame (at the bottom bracket) 0.12 m lower than at the start
        return bool(abs(self.sensation()[0]) > .7 or self.data.qpos[2] < self.z0 - .12)

    @property
    def travelled(self):
        return float(self.data.qpos[0] - self.origin[0])

    @property
    def done(self):
        return self.fallen or self.travelled >= self.distance or self.data.time >= self.max_seconds

    def step(self, action):
        """``action`` = (drive torque, steering angle, waist lean), as for ``BicycleEnv``."""
        torque, steer, lean = map(float, action)
        ctrl = self.pose.copy()
        ctrl[self.a_pedal] = torque
        ctrl[self.a_steer] = steer
        ctrl[self.a_lean] = lean
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(ctrl, lo, hi)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.sensation()
