"""Batched G1 locomotion environment with velocity commands as goals.

All bodies are stepped in one C call per control step (``mujoco.rollout`` on a thread pool),
so a single Python process drives thousands of humanoids on the CPU.

Goal: a velocity command ``(vx, vy, yaw_rate)`` in the body's heading frame, or a heading target
(converted to a yaw rate by a proportional rule). Preferences over outcomes (the pragmatic part of
the objective) are Gaussian-shaped scores of the command error plus "stay upright and smooth"
costs; a central pattern generator (CPG) clock gives the preferred foot-height rhythm, whose
frequency and duty factor scale with the commanded speed (walking = 50% stance, running = less
than 50% stance, i.e. a flight phase). Nothing in this file chooses joint targets: the controller
is learned.

Observation (proprioception + goal, 50 dims):
    0:3   pelvis angular velocity (body frame)          3:6   gravity in the body frame
    6:9   pelvis linear velocity (heading frame)        9:12  command (vx, vy, yaw rate)
    12:24 leg joint angles minus the default posture    24:36 leg joint velocities
    36:48 previous action                               48:50 CPG clock (sin, cos)
The pelvis linear velocity comes from the simulator (a real robot would estimate it); stated in
docs/LOCOMOTION_STAGE2.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco import rollout

from pai.locomotion.body import CONTROL_SPEC, G1Body, Randomization, load_body, place_walls, randomized_models

OBS_DIM = 50
ACT_DIM = 12
PROPRIO = np.r_[0:9, 12:36]          # what the body model predicts (33 dims)
CMD = slice(9, 12)


@dataclass
class Stage:
    name: str
    vx: tuple[float, float] = (0.0, 0.0)
    vy: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (0.0, 0.0)
    heading_prob: float = 0.0
    push_dv: float = 0.0               # largest push, as a velocity change of the whole body (m/s)
    zero_prob: float = 0.15            # share of "stand still" commands kept in every stage


DEFAULT_STAGES = (
    Stage("stand"),
    Stage("balance", push_dv=0.5),
    Stage("walk", vx=(-0.3, 1.0), vy=(-0.3, 0.3), yaw=(-0.3, 0.3), push_dv=0.5),
    Stage("turn", vx=(-0.3, 1.0), vy=(-0.3, 0.3), yaw=(-1.0, 1.0), heading_prob=0.3, push_dv=0.6),
    Stage("run", vx=(-0.5, 2.2), vy=(-0.3, 0.3), yaw=(-1.0, 1.0), heading_prob=0.3, push_dv=0.6),
)


@dataclass
class RewardWeights:
    lin: float = 2.0
    yaw: float = 0.8
    phase: float = 1.0
    air: float = 1.0
    alive: float = 0.5
    vz: float = 1.0
    ang_xy: float = 0.05
    orient: float = 2.0
    rate: float = 0.01
    qd: float = 1e-4
    hip_pose: float = 0.3
    still: float = 0.5
    slip: float = 0.2
    height: float = 10.0
    fall: float = 10.0
    scale: float = 0.1


@dataclass
class EnvConfig:
    ctrl_dt: float = 0.02
    action_scale: float = 0.3
    action_clip: float = 4.0
    episode_s: float = 20.0
    fall_height: float = 0.5
    fall_upright: float = 0.6          # cos(tilt) below this = fallen (~53 deg)
    contact_height: float = 0.03
    swing_height: float = 0.08
    phase_sigma: float = 0.003         # m^2: standing still on a stepping goal scores ~0.2
    stand_threshold: float = 0.1
    heading_gain: float = 1.0
    push_s: float = 0.2
    push_interval_s: tuple[float, float] = (3.0, 7.0)
    resample_s: tuple[float, float] = (4.0, 8.0)
    obs_noise: float = 0.0
    init_noise: float = 0.05
    randomize: bool = True
    n_variants: int = 32
    nthread: int = 6
    # wall-contact practice: ``walls`` solid 1 m blocks placed in a fraction ``wall_frac`` of the
    # dynamics variants, ``wall_radius`` metres from the start (0 = no walls, as before)
    walls: int = 0
    wall_frac: float = 0.5
    # maze-goal practice: this share of the bodies get the maze walker's goal pattern (turn on the
    # spot to a new heading, walk about one cell, stop, repeat) instead of random goals; half of them
    # with the abrupt steering (walk at full speed below 25 deg), half with the smooth one
    maze_goal_frac: float = 0.0
    wall_radius: tuple[float, float] = (1.0, 3.0)
    reward: RewardWeights = field(default_factory=RewardWeights)
    stages: tuple[Stage, ...] = DEFAULT_STAGES

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        rw = RewardWeights(**d.pop("reward", {}))
        stages = d.pop("stages", None)
        st = tuple(Stage(**s) for s in stages) if stages else DEFAULT_STAGES
        for k in ("push_interval_s", "resample_s", "wall_radius"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(reward=rw, stages=st, **d)


# ---------------------------------------------------------------- small vector maths
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gait_params(speed: np.ndarray):
    """CPG frequency (Hz) and stance share for a commanded planar speed."""
    s = np.clip(speed, 0.0, 2.5)
    freq = 1.3 + 0.35 * s
    duty = np.clip(0.5 - 0.15 * (s - 1.0), 0.35, 0.5)
    return freq, duty


def desired_foot_heights(phase: np.ndarray, duty: np.ndarray, swing: float, moving: np.ndarray):
    """Preferred sole heights (n, 2) from the CPG clock; zero (both feet down) when standing."""
    p = np.stack([phase, (phase + 0.5) % 1.0], axis=1)
    d = duty[:, None]
    sw = np.clip((p - d) / (1.0 - d), 0.0, 1.0)
    z = np.where(p >= d, swing * np.sin(np.pi * sw), 0.0)
    return z * moving[:, None]


class LocomotionEnv:
    def __init__(self, num_envs: int, cfg: EnvConfig | None = None, seed: int = 0,
                 stage: int = 0, body: G1Body | None = None):
        self.cfg = cfg or EnvConfig()
        self.n = num_envs
        self.body = body or load_body(walls=self.cfg.walls)
        self.rng = np.random.default_rng(seed)
        m = self.body.model
        self.substeps = int(round(self.cfg.ctrl_dt / m.opt.timestep))
        if self.cfg.randomize:
            variants = randomized_models(self.body, self.cfg.n_variants, self.rng, Randomization())
            self.wall_layout = (place_walls(variants, self.rng, self.cfg.walls, self.cfg.wall_frac,
                                            self.cfg.wall_radius) if self.cfg.walls else [])
            self.models = [variants[i % len(variants)] for i in range(num_envs)]
        else:
            self.models = [m] * num_envs
        self.datas = [mujoco.MjData(m) for _ in range(max(1, self.cfg.nthread))]
        self.max_steps = int(round(self.cfg.episode_s / self.cfg.ctrl_dt))
        self.stage = stage
        self.auto_commands = True
        self.base_state = self.body.default_state()
        self.state = np.tile(self.base_state, (num_envs, 1))
        self.control = np.zeros((num_envs, self.substeps, self.body.ncontrol))
        self.sensors = np.zeros((num_envs, m.nsensordata))
        self._out_state = np.zeros((num_envs, self.substeps, self.body.nstate))
        self._out_sens = np.zeros((num_envs, self.substeps, m.nsensordata))
        self.command = np.zeros((num_envs, 3))
        self.heading_target = np.zeros(num_envs)
        self.heading_mode = np.zeros(num_envs, bool)
        self.phase = np.zeros(num_envs)
        self.last_action = np.zeros((num_envs, ACT_DIM))
        self.steps = np.zeros(num_envs, int)
        self.air_time = np.zeros((num_envs, 2))
        self.contact = np.ones((num_envs, 2), bool)
        self.push_force = np.zeros((num_envs, 3))
        self.push_left = np.zeros(num_envs, int)
        self.next_push = np.zeros(num_envs, int)
        self.next_resample = np.zeros(num_envs, int)
        self.maze_mode = self.rng.random(num_envs) < self.cfg.maze_goal_frac
        self.maze_smooth = self.rng.random(num_envs) < 0.5
        self.maze_walking = np.zeros(num_envs, bool)
        self.maze_v = np.zeros(num_envs)
        self.maze_from = np.zeros((num_envs, 2))
        self.maze_pause = np.zeros(num_envs, int)
        self.pushes_enabled = True
        self.reset(np.arange(num_envs))

    # ----------------------------------------------------------------- commands and pushes
    def _sample_commands(self, ids: np.ndarray):
        st = self.cfg.stages[self.stage]
        k = len(ids)
        r = self.rng
        cmd = np.stack([r.uniform(*st.vx, k), r.uniform(*st.vy, k), r.uniform(*st.yaw, k)], 1)
        zero = r.random(k) < st.zero_prob
        cmd[zero] = 0.0
        small = np.linalg.norm(cmd[:, :2], axis=1) < self.cfg.stand_threshold
        cmd[small, :2] = 0.0
        self.command[ids] = cmd
        hm = (r.random(k) < st.heading_prob) & ~zero
        self.heading_mode[ids] = hm
        yaw = self.yaw()[ids]
        self.heading_target[ids] = wrap(yaw + r.uniform(-np.pi, np.pi, k))
        lo, hi = self.cfg.resample_s
        self.next_resample[ids] = self.steps[ids] + (r.uniform(lo, hi, k) / self.cfg.ctrl_dt).astype(int)

    def _maze_goals(self):
        """The maze walker's goal pattern for the bodies in maze mode (vectorized state machine)."""
        ids = np.nonzero(self.maze_mode)[0]
        dt = self.cfg.ctrl_dt
        yaw = self.yaw()[ids]
        xy = self.qpos()[ids, :2]
        err = np.abs(wrap(self.heading_target[ids] - yaw))
        walking = self.maze_walking[ids]
        smooth = self.maze_smooth[ids]
        # finished a cell: stop briefly, then pick the next heading (straight, left, right, back)
        done = walking & (np.linalg.norm(xy - self.maze_from[ids], axis=1) >= 1.0)
        if done.any():
            d = ids[done]
            turn = self.rng.choice([0.0, np.pi / 2, -np.pi / 2, np.pi], len(d), p=[0.4, 0.25, 0.25, 0.1])
            self.heading_target[d] = wrap(self.heading_target[d] + turn)
            self.maze_walking[d] = False
            self.maze_v[d] = 0.0
            self.maze_pause[d] = int(round(0.3 / dt))
            walking = self.maze_walking[ids]
        start = ~walking & (self.maze_pause[ids] <= 0) & np.where(smooth, err < np.radians(10), err < np.radians(25))
        if start.any():
            s_ids = ids[start]
            self.maze_walking[s_ids] = True
            self.maze_from[s_ids] = self.qpos()[s_ids, :2]
        stop = self.maze_walking[ids] & smooth & (err > np.radians(40))
        self.maze_walking[ids[stop]] = False
        walking = self.maze_walking[ids]
        target_v = np.where(walking, 0.6, 0.0)
        v = np.where(smooth, np.clip(target_v, self.maze_v[ids] - dt, self.maze_v[ids] + dt), target_v)
        if not smooth.all():
            # abrupt steering: full speed only while the heading error is below 25 deg
            v = np.where(~smooth & (err > np.radians(25)), 0.0, v)
        self.maze_v[ids] = v
        self.maze_pause[ids] = np.maximum(self.maze_pause[ids] - 1, 0)
        self.command[ids] = np.stack([v, np.zeros(len(ids)), np.zeros(len(ids))], 1)
        self.heading_mode[ids] = True

    def set_command(self, ids, vx=0.0, vy=0.0, yaw_rate=0.0, heading=None):
        """External goal (used by evaluation and the skill API); disables random resampling."""
        ids = np.atleast_1d(np.asarray(ids))
        self.auto_commands = False
        self.command[ids] = np.array([vx, vy, yaw_rate], float)
        if heading is None:
            self.heading_mode[ids] = False
        else:
            self.heading_mode[ids] = True
            self.heading_target[ids] = wrap(np.asarray(heading, float))

    def _schedule_push(self, ids):
        lo, hi = self.cfg.push_interval_s
        self.next_push[ids] = self.steps[ids] + (self.rng.uniform(lo, hi, len(ids)) / self.cfg.ctrl_dt).astype(int)

    def apply_push(self, ids, force_xyz, duration_s: float | None = None):
        """A real push: a constant horizontal force on the torso for ``duration_s``."""
        ids = np.atleast_1d(np.asarray(ids))
        self.push_force[ids] = np.asarray(force_xyz, float)
        self.push_left[ids] = int(round((duration_s or self.cfg.push_s) / self.cfg.ctrl_dt))

    def push_force_for_dv(self, dv: float, duration_s: float | None = None) -> float:
        return self.body.total_mass * dv / (duration_s or self.cfg.push_s)

    def _random_pushes(self):
        dv_max = self.cfg.stages[self.stage].push_dv
        if dv_max <= 0 or not self.pushes_enabled:
            return
        due = np.nonzero(self.steps >= self.next_push)[0]
        if len(due) == 0:
            return
        ang = self.rng.uniform(-np.pi, np.pi, len(due))
        mag = self.push_force_for_dv(1.0) * self.rng.uniform(0.0, dv_max, len(due))
        self.apply_push(due, np.stack([mag * np.cos(ang), mag * np.sin(ang), np.zeros(len(due))], 1))
        self._schedule_push(due)

    # ----------------------------------------------------------------- state access
    def qpos(self):
        return self.state[:, self.body.qpos0:self.body.qpos0 + self.body.model.nq]

    def qvel(self):
        return self.state[:, self.body.qvel0:self.body.qvel0 + self.body.model.nv]

    def yaw(self):
        q = self.qpos()[:, 3:7]
        w, x, y, z = q.T
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def feet(self):
        s = self.sensors
        pos = np.stack([s[:, a:a + 3] for a in self.body.foot_pos_adr], 1)
        vel = np.stack([s[:, a:a + 3] for a in self.body.foot_vel_adr], 1)
        return pos, vel

    def soles(self, sensors=None):
        """Sole heights above flat ground (n, 2) and sole velocities (n, 2, 3)."""
        self._default_feet()
        s = self.sensors if sensors is None else sensors
        pos = np.stack([s[:, a:a + 3] for a in self.body.foot_pos_adr], 1)
        vel = np.stack([s[:, a:a + 3] for a in self.body.foot_vel_adr], 1)
        return pos[:, :, 2] - self.sole_z0, vel

    def kinematics(self):
        qp, qv = self.qpos(), self.qvel()
        R = quat_to_mat(qp[:, 3:7])
        yaw = self.yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        v = qv[:, 0:3]
        v_head = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
        w_local = qv[:, 3:6]
        w_world = np.einsum("nij,nj->ni", R, w_local)
        grav = -R[:, 2, :]
        return dict(R=R, yaw=yaw, v_head=v_head, w_local=w_local, w_world=w_world, grav=grav,
                    height=qp[:, 2], q=qp[:, self.body.leg_qpos], qd=qv[:, self.body.leg_qvel])

    def effective_command(self, yaw=None):
        cmd = self.command.copy()
        if self.heading_mode.any():
            yaw = self.yaw() if yaw is None else yaw
            err = wrap(self.heading_target - yaw)
            hm = self.heading_mode
            cmd[hm, 2] = np.clip(self.cfg.heading_gain * err[hm], -1.0, 1.0)
        return cmd

    def observe(self, k=None):
        k = k or self.kinematics()
        cmd = self.effective_command(k["yaw"])
        obs = np.concatenate([
            k["w_local"], k["grav"], k["v_head"], cmd,
            k["q"] - self.body.default_leg, k["qd"], self.last_action,
            np.sin(2 * np.pi * self.phase)[:, None], np.cos(2 * np.pi * self.phase)[:, None],
        ], 1)
        if self.cfg.obs_noise > 0:
            noise = self.rng.normal(0, self.cfg.obs_noise, obs.shape)
            noise[:, 9:12] = 0.0
            noise[:, 36:] = 0.0
            obs = obs + noise
        return obs.astype(np.float32)

    # ----------------------------------------------------------------- reset / step
    def reset(self, ids: np.ndarray):
        ids = np.asarray(ids)
        if len(ids) == 0:
            return
        r = self.rng
        s = np.tile(self.base_state, (len(ids), 1))
        qp = s[:, self.body.qpos0:self.body.qpos0 + self.body.model.nq]
        qp[:, self.body.leg_qpos] += r.uniform(-1, 1, (len(ids), ACT_DIM)) * self.cfg.init_noise
        yaw0 = r.uniform(-np.pi, np.pi, len(ids))
        qp[:, 3] = np.cos(yaw0 / 2)
        qp[:, 4:6] = 0.0
        qp[:, 6] = np.sin(yaw0 / 2)
        qp[:, 0:2] = 0.0
        s[:, 0] = 0.0
        self.state[ids] = s
        self.sensors[ids] = 0.0
        self.steps[ids] = 0
        self.phase[ids] = r.random(len(ids))
        self.last_action[ids] = 0.0
        self.air_time[ids] = 0.0
        self.contact[ids] = True
        self.push_force[ids] = 0.0
        self.push_left[ids] = 0
        self._schedule_push(ids)
        if self.auto_commands:
            self._sample_commands(ids)
            m = ids[self.maze_mode[ids]]
            if len(m):
                self.maze_walking[m] = False
                self.maze_v[m] = 0.0
                self.maze_pause[m] = 0
                self.heading_target[m] = wrap(self.yaw()[m] + self.rng.choice([0.0, np.pi / 2, -np.pi / 2], len(m)))
        else:
            self.heading_target[ids] = wrap(self.heading_target[ids])
        # sensors are refreshed by the first step; seed them with the default posture
        feet0 = self._default_feet()
        self.sensors[ids] = feet0

    def _default_feet(self):
        if not hasattr(self, "_feet0"):
            d = mujoco.MjData(self.body.model)
            mujoco.mj_setState(self.body.model, d, self.base_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(self.body.model, d)
            self._feet0 = d.sensordata.copy()
            # sole height on flat ground: the reference for the swing preference
            self.sole_z0 = float(np.mean([d.sensordata[a + 2] for a in self.body.foot_pos_adr]))
        return self._feet0

    def step(self, action: np.ndarray):
        cfg, b = self.cfg, self.body
        a = np.clip(np.asarray(action, float), -cfg.action_clip, cfg.action_clip)
        # goals and disturbances
        if self.auto_commands:
            due = np.nonzero((self.steps >= self.next_resample) & ~self.maze_mode)[0]
            if len(due):
                self._sample_commands(due)
            if self.maze_mode.any():
                self._maze_goals()
        self._random_pushes()
        # controls: leg servo targets from the action, everything else held at the default
        target = np.tile(b.default_ctrl, (self.n, 1))
        target[:, b.leg_ctrl] = np.clip(b.default_leg + cfg.action_scale * a, b.leg_lo, b.leg_hi)
        ctrl = self.control
        ctrl[:, :, :b.model.nu] = target[:, None, :]
        pushing = self.push_left > 0
        off = b.model.nu + 6 * b.push_body
        ctrl[:, :, off:off + 3] = (self.push_force * pushing[:, None])[:, None, :]
        self.push_left = np.maximum(self.push_left - 1, 0)
        rollout.rollout(self.models, self.datas, self.state, ctrl,
                        control_spec=CONTROL_SPEC, nstep=self.substeps, skip_checks=True,
                        state=self._out_state, sensordata=self._out_sens, persistent_pool=True)
        self.state = self._out_state[:, -1, :].copy()
        self.sensors = self._out_sens[:, -1, :].copy()
        self.steps += 1
        k = self.kinematics()
        cmd = self.effective_command(k["yaw"])
        speed = np.linalg.norm(cmd[:, :2], axis=1)
        moving = (speed >= cfg.stand_threshold) | (np.abs(cmd[:, 2]) >= 0.2)
        freq, duty = gait_params(speed)
        self.phase = (self.phase + freq * cfg.ctrl_dt) % 1.0
        r, terms = self._reward(k, cmd, a, moving, duty)
        self.last_action = a
        upright = -k["grav"][:, 2]
        fell = (k["height"] < cfg.fall_height) | (upright < cfg.fall_upright) | ~np.isfinite(self.state).all(1)
        trunc = (self.steps >= self.max_steps) & ~fell
        r = np.where(fell, r - cfg.reward.fall * cfg.reward.scale, r)
        done = fell | trunc
        info = {"fell": fell, "truncated": trunc, "terms": terms, "pushing": pushing,
                "episode_steps": self.steps.copy()}
        if done.any():
            ids = np.nonzero(done)[0]
            info["final_obs"] = self.observe(k)[ids]
            info["done_ids"] = ids
            info["final_sensors"] = self.sensors[ids].copy()
            info["final_qpos"] = self.qpos()[ids].copy()
            self.reset(ids)
            obs = self.observe()
        else:
            obs = self.observe(k)
        return obs, r.astype(np.float32), done, info

    def _reward(self, k, cmd, a, moving, duty):
        w, cfg = self.cfg.reward, self.cfg
        pos, vel = self.feet()
        sole = pos[:, :, 2] - self.sole_z0
        contact = sole < cfg.contact_height
        # air time at touchdown (rewarded only while moving; never negative, so first clumsy
        # short steps are not punished: a negative version kept the body standing in place)
        touchdown = contact & ~self.contact
        air = np.sum(np.clip(self.air_time - 0.1, 0.0, 0.3) * touchdown, 1) * moving
        self.air_time = np.where(contact, 0.0, self.air_time + cfg.ctrl_dt)
        self.contact = contact
        lin_err = np.sum((cmd[:, :2] - k["v_head"][:, :2]) ** 2, 1)
        yaw_err = (cmd[:, 2] - k["w_world"][:, 2]) ** 2
        z_des = desired_foot_heights(self.phase, duty, cfg.swing_height, moving)
        phase_err = np.sum((np.clip(sole, 0, None) - z_des) ** 2, 1)
        dq = k["q"] - self.body.default_leg
        hip = dq[:, [1, 2, 7, 8]]
        slip = np.sum(contact * np.sum(vel[:, :, :2] ** 2, 2), 1)
        terms = {
            "lin": np.exp(-lin_err / 0.25),
            "yaw": np.exp(-yaw_err / 0.25),
            "phase": np.exp(-phase_err / cfg.phase_sigma),
            "air": air,
            "alive": np.ones(self.n),
            "vz": -k["v_head"][:, 2] ** 2,
            "ang_xy": -np.sum(k["w_local"][:, :2] ** 2, 1),
            "orient": -np.sum(k["grav"][:, :2] ** 2, 1),
            "rate": -np.sum((a - self.last_action) ** 2, 1),
            "qd": -np.sum(k["qd"] ** 2, 1),
            "hip_pose": -np.sum(hip ** 2, 1),
            "still": -np.sum(dq ** 2, 1) * ~moving,
            "slip": -slip,
            "height": -np.maximum(0.0, 0.68 - k["height"]) ** 2,
        }
        r = sum(getattr(w, name) * v for name, v in terms.items()) * w.scale
        terms["lin_err"] = np.sqrt(lin_err)
        terms["yaw_err"] = np.sqrt(yaw_err)
        terms["contact"] = contact
        return r, terms
