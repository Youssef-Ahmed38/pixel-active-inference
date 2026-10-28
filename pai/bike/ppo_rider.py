"""A learned riding policy for the G1 on its bicycle, trained with PPO (as CycleRL does).

The policy sees the bicycle's own state and the route ahead, and outputs the steering angle and the
drive torque. Observation (16 numbers; ``observe``):

- roll, roll rate, yaw rate, forward speed, steering angle (the bicycle's IMU and steering joint);
- the signed distance from the lane line (left positive);
- heading errors to route points 1, 2.5 and 4 m ahead;
- the route's curvature here and 1, 2, 3 and 4 m ahead;
- the previous action (2).

Action in [-1, 1]^2: steering angle = 0.6 x a0 rad, drive torque = 12 x a1 N m (the limits of the
bicycle's actuators). ``BikeVecEnv`` runs many riders at once with ``mujoco.rollout`` (CPU threads);
the network trains on the GPU (``scripts/bike_g1_ppo_train.py``). ``PPOFollower`` drives one rider
with a trained policy through the same interface as ``pai/bike/riding.LeanFollower``, so the delivery
code can use either.

Reward per 0.02 s step: 2 x metres of route progress x exp(-(distance from the lane / 0.75 m)^2)
(progress counts only near the lane; the first version paid progress anywhere, and its policy learned
to balance while drifting off the route), minus 0.1 x (distance from the lane)^2, minus
0.02 x (change of action)^2, minus 0.05 x speed above 3.5 m/s; a fall or leaving the route by more
than 3 m ends the episode with -5; finishing the route ends it with +2.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch
from mujoco import rollout

from pai.bike.g1_rider import G1BikeEnv
from pai.bike.real_bike import GEAR, R_TYRE
from pai.bike.riding import CORNER_R, LeanFollower, curvature, lane_path, wrap

OBS_DIM, ACT_DIM = 16, 2
STEER_MAX, TORQUE_MAX = 0.6, 12.0
HEAD_AHEAD = (2, 5, 8)            # path points (0.5 m apart): 1, 2.5, 4 m
KAPPA_AHEAD = (0, 2, 4, 6, 8)
OFF_ROUTE_M = 3.0
# residual mode: the policy corrects the practised lean-target rule by at most these amounts
RES_STEER, RES_TORQUE = 0.15, 4.0
ARM_KP, STEER_KP = 5.0, 300.0


def quat_roll_yaw(q):
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def progress(path, k, xy, window=40):
    lo, hi = k, min(len(path), k + window)
    j = lo + int(np.argmin(np.linalg.norm(path[lo:hi] - xy, axis=1)))
    return j


def observe(roll, roll_rate, yaw_rate, speed, steer, yaw, xy, path, kappa, k, prev_action):
    n = len(path)
    t = path[min(k + 1, n - 1)] - path[max(k - 1, 0)]
    t = t / (np.linalg.norm(t) + 1e-9)
    cross = float((xy - path[k]) @ np.array([-t[1], t[0]]))
    heads = []
    for o in HEAD_AHEAD:
        p = path[min(k + o, n - 1)]
        heads.append(wrap(math.atan2(p[1] - xy[1], p[0] - xy[0]) - yaw) if np.linalg.norm(p - xy) > 0.2 else 0.0)
    kap = [float(kappa[min(k + o, n - 1)]) for o in KAPPA_AHEAD]
    return np.array([roll, roll_rate, yaw_rate, speed, steer, cross, *heads, *kap, *prev_action], np.float32)


def to_controls(a):
    a = np.clip(a, -1, 1)
    return STEER_MAX * a[..., 0], TORQUE_MAX * a[..., 1]


def residual_controls(a, rule_steer, rule_torque):
    """The rule's steering and torque plus the policy's bounded correction."""
    a = np.clip(a, -1, 1)
    return (np.clip(rule_steer + RES_STEER * a[..., 0], -STEER_MAX, STEER_MAX),
            np.clip(rule_torque + RES_TORQUE * a[..., 1], -TORQUE_MAX, TORQUE_MAX))


class PPOFollower:
    """A trained policy behind the ``LeanFollower`` interface: ``action(s, xy) -> (action, info)``."""

    def __init__(self, path, policy, env: G1BikeEnv, residual_gains=None):
        self.path, self.ac, self.env = path, policy, env
        self.rule = None if residual_gains is None else LeanFollower(path, residual_gains)
        self.kappa = curvature(path)
        self.i = 0
        self.prev = np.zeros(2, np.float32)
        d = env.data
        self._gyro = env.model.sensor("angular_velocity").adr[0]

    def action(self, s, xy):
        roll, _, speed, steer, _, yaw = s
        gyro = self.env.data.sensordata[self._gyro:self._gyro + 3]
        self.i = progress(self.path, self.i, xy)
        o = observe(roll, gyro[0], gyro[2], speed, steer, yaw, xy, self.path, self.kappa, self.i, self.prev)
        with torch.no_grad():
            on = self.ac.obs_norm(torch.as_tensor(o[None]))
            a = self.ac.dist(on).mean[0].numpy()
        a = np.clip(a, -1, 1)
        self.prev = a.astype(np.float32)
        if self.rule is not None:
            ra, _ = self.rule.action(s, xy)
            st, tq = residual_controls(a, ra[1], ra[0])
        else:
            st, tq = to_controls(a)
        info = {"progress": self.i / max(1, len(self.path) - 1), "cross_track": abs(float(o[5])),
                "heading_error": float(o[6]), "target_speed": float("nan"), "lean_ref": float("nan")}
        return np.array([tq, st, 0.0]), info

    @property
    def finished(self):
        return self.i >= len(self.path) - 3


# ------------------------------------------------------------------ many riders at once
class BikeVecEnv:
    """``n`` riders on routes from the street maps of ``towns`` (random corners and U-turns)."""

    def __init__(self, n: int, towns, seed: int = 0, nthread: int = 12, u_turn_frac: float = 0.3,
                 max_s: float = 90.0, residual_gains=None, vary_geometry=False, car_frac: float = 0.0):
        self.n, self.towns, self.rng = n, list(towns), np.random.default_rng(seed)
        self.residual_gains = residual_gains
        self.vary_geometry = vary_geometry
        self.car_frac = car_frac
        self.rules = [None] * n
        self.u_turn_frac = u_turn_frac
        self.single = G1BikeEnv(scenery=False, max_seconds=1e9, arm_kp=ARM_KP, steer_kp=STEER_KP)
        m = self.m = self.single.model
        self.substeps = self.single.substeps
        self.dt = self.single.control_dt
        self.max_steps = int(max_s / self.dt)
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.single.reset(seed=0, speed=2.5, roll=0.0, start_xy=(0.0, 0.0), yaw=0.0)
        self.base = np.zeros(mujoco.mj_stateSize(m, self.spec))
        mujoco.mj_getState(m, self.single.data, self.base, self.spec)
        self.z0 = float(self.single.data.qpos[2])
        self.nq = m.nq
        self.datas = [mujoco.MjData(m) for _ in range(nthread)]
        self.ctrl = np.tile(self.single.pose, (n, self.substeps, 1))
        self.a_pedal, self.a_steer = self.single.a_pedal, self.single.a_steer
        self.out_state = np.zeros((n, self.substeps, self.base.size))
        self.out_sens = np.zeros((n, self.substeps, m.nsensordata))
        self.state = np.tile(self.base, (n, 1))
        self.sens = np.zeros((n, m.nsensordata))
        self.adr = {k: m.sensor(k).adr[0] for k in ("orientation", "angular_velocity", "velocity")}
        self.steer_q = 1 + self.single.steer_q
        wheel = [m.jnt_dofadr[m.joint(j).id] for j in ("rear_spin", "front_spin")]
        self.wheel_v = [1 + m.nq + a for a in wheel]
        self.crank_v = 1 + m.nq + m.jnt_dofadr[m.joint("crank").id]
        self.paths = [None] * n
        self.kappas = [None] * n
        self.arc = [None] * n
        self.k = np.zeros(n, int)
        self.prev = np.zeros((n, 2), np.float32)
        self.steps = np.zeros(n, int)
        self._town_objs = {}
        self.reset(np.arange(n))

    def _with_cars(self, path):
        """Parked cars in the rider's lane at random straight places: the path changes lane around them
        (as ``pai/bike/delivery.py`` plans it), often right after a corner."""
        from pai.bike.navigation import avoid_cars
        kap = np.abs(curvature(path))
        ok = [k for k in range(16, len(path) - 16) if kap[max(0, k - 2):k + 3].max() < 0.02]
        if not ok:
            return path
        cars = []
        for _ in range(int(self.rng.integers(1, 3))):
            k = int(self.rng.choice(ok))
            cars.append(tuple(path[k]))
        return avoid_cars(path, cars)

    def _route(self):
        from pai.bike.town import Town
        s = int(self.rng.choice(self.towns))
        if s not in self._town_objs:
            self._town_objs[s] = Town(n=4, seed=s)
        t = self._town_objs[s]
        g = t.graph
        if self.rng.random() < self.u_turn_frac:
            a = int(self.rng.choice(list(t.nodes)))
            b = int(self.rng.choice(g[a]))
            c = int(self.rng.choice([x for x in g[b] if x != a] or g[b]))
            r = [a, b, c, b, int(self.rng.choice([x for x in g[b] if x != c]))]
        else:
            r = [int(self.rng.choice(list(t.nodes)))]
            for _ in range(int(self.rng.integers(3, 6))):
                r.append(int(self.rng.choice([x for x in g[r[-1]] if len(r) < 2 or x != r[-2]])))
        if not self.vary_geometry:
            path = lane_path(t.nodes, r, radius=CORNER_R)
            return self._with_cars(path) if self.rng.random() < self.car_frac else path
        # randomized geometry: corners 3.0-3.8 m, U-turn loops 2.2-2.6 m (the practised rule is brittle to these)
        import pai.bike.riding as riding
        keep = riding.U_TURN_R
        riding.U_TURN_R = float(self.rng.uniform(2.2, 2.6))
        try:
            path = lane_path(t.nodes, r, radius=float(self.rng.uniform(3.0, 3.8)))
        finally:
            riding.U_TURN_R = keep
        return self._with_cars(path) if self.rng.random() < self.car_frac else path

    def reset(self, ids):
        for i in ids:
            path = self._route()
            self.paths[i], self.kappas[i] = path, curvature(path)
            if self.residual_gains is not None:
                self.rules[i] = LeanFollower(path, self.residual_gains)
            self.arc[i] = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
            d = path[min(5, len(path) - 1)] - path[0]
            yaw = math.atan2(d[1], d[0])
            roll = float(self.rng.uniform(-0.03, 0.03))
            speed = float(self.rng.uniform(2.2, 2.8))
            s = self.base.copy()
            q = s[1:1 + self.nq]
            q[0:2] = path[0]
            cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
            q[3:7] = [cy * cr, cy * sr, sy * sr, sy * cr]
            v = s[1 + self.nq:]
            v[:] = 0.0
            v[0], v[1] = speed * math.cos(yaw), speed * math.sin(yaw)
            for a in self.wheel_v:
                s[a] = speed / R_TYRE
            s[self.crank_v] = GEAR * speed / R_TYRE
            self.state[i] = s
            self.k[i] = 0
            self.prev[i] = 0.0
            self.steps[i] = 0
            # sensors before the first step: upright, at speed
            self.sens[i] = 0.0
            self.sens[i, self.adr["orientation"]:self.adr["orientation"] + 4] = q[3:7]
            self.sens[i, self.adr["velocity"]] = speed

    def _obs_one(self, i):
        st, se = self.state[i], self.sens[i]
        q = se[self.adr["orientation"]:self.adr["orientation"] + 4]
        roll, yaw = quat_roll_yaw(q)
        g = se[self.adr["angular_velocity"]:self.adr["angular_velocity"] + 3]
        speed = se[self.adr["velocity"]]
        xy = st[1:3]
        return observe(roll, g[0], g[2], speed, st[self.steer_q], yaw, xy, self.paths[i], self.kappas[i],
                       self.k[i], self.prev[i]), roll

    def observe(self):
        return np.stack([self._obs_one(i)[0] for i in range(self.n)])

    def _rule_action(self, i):
        st, se = self.state[i], self.sens[i]
        roll, yaw = quat_roll_yaw(se[self.adr["orientation"]:self.adr["orientation"] + 4])
        s = np.array([roll, se[self.adr["angular_velocity"]], se[self.adr["velocity"]], st[self.steer_q], 0.0, yaw])
        a, _ = self.rules[i].action(s, st[1:3])
        return a[1], a[0]

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        if self.residual_gains is not None:
            rs = np.array([self._rule_action(i) for i in range(self.n)])
            st, tq = residual_controls(a, rs[:, 0], rs[:, 1])
        else:
            st, tq = to_controls(a)
        self.ctrl[:, :, self.a_steer] = st[:, None]
        self.ctrl[:, :, self.a_pedal] = tq[:, None]
        rollout.rollout([self.m] * self.n, self.datas, self.state, self.ctrl,
                        control_spec=int(mujoco.mjtState.mjSTATE_CTRL), nstep=self.substeps, skip_checks=True,
                        state=self.out_state, sensordata=self.out_sens, persistent_pool=True)
        self.state = self.out_state[:, -1].copy()
        self.sens = self.out_sens[:, -1].copy()
        self.steps += 1
        obs = np.zeros((self.n, OBS_DIM), np.float32)
        rew = np.zeros(self.n, np.float32)
        done = np.zeros(self.n, bool)
        fell = np.zeros(self.n, bool)
        trunc = np.zeros(self.n, bool)
        finished = np.zeros(self.n, bool)
        da = a - self.prev
        for i in range(self.n):
            xy = self.state[i, 1:3]
            k0 = self.k[i]
            self.k[i] = progress(self.paths[i], k0, xy)
            self.prev[i] = a[i]
            o, roll = self._obs_one(i)
            obs[i] = o
            cross, speed = abs(float(o[5])), float(o[3])
            # progress counts fully on the lane line and less the further the rider is from it
            r = 2.0 * (self.arc[i][self.k[i]] - self.arc[i][k0]) * math.exp(-(cross / 0.75) ** 2) \
                - 0.1 * cross ** 2 - 0.02 * float(da[i] @ da[i]) - 0.05 * max(0.0, speed - 3.5)
            bad = abs(roll) > 0.7 or self.state[i, 3] < self.z0 - 0.12 or cross > OFF_ROUTE_M \
                or not np.isfinite(self.state[i]).all()
            if bad:
                r -= 5.0
                fell[i] = done[i] = True
            elif self.k[i] >= len(self.paths[i]) - 3:
                r += 2.0
                finished[i] = done[i] = True
            elif self.steps[i] >= self.max_steps:
                trunc[i] = done[i] = True
            rew[i] = r
        info = {"fell": fell, "finished": finished, "trunc": trunc}
        ids = np.nonzero(done)[0]
        if len(ids):
            info["final_obs"] = obs[ids].copy()
            info["done_ids"] = ids
            self.reset(ids)
            for i in ids:
                obs[i] = self._obs_one(i)[0]
        return obs, rew, done, info


# ------------------------------------------------------------------ the same, computed for all riders at once
class FastBikeVecEnv(BikeVecEnv):
    """``BikeVecEnv`` with observations, progress, rewards and the residual rule computed as arrays
    over all riders (the per-rider Python loop, not the physics, limited the speed). Same numbers."""

    LMAX = 1200

    def _store(self, i):
        path, L = self.paths[i], len(self.paths[i])
        P = np.empty((self.LMAX, 2))
        P[:L], P[L:] = path, path[-1]
        K = np.zeros(self.LMAX)
        K[:L] = self.kappas[i]
        A = np.full(self.LMAX, self.arc[i][-1])
        A[:L] = self.arc[i]
        T = np.zeros(self.LMAX)
        ak = np.abs(self.kappas[i])
        T[:L] = [ak[k:k + 12].max() * CORNER_R for k in range(L)]
        self.P[i], self.K[i], self.A[i], self.T[i], self.L[i] = P, K, A, T, L

    def reset(self, ids):
        if not hasattr(self, "P"):
            n = self.n
            self.P, self.K = np.zeros((n, self.LMAX, 2)), np.zeros((n, self.LMAX))
            self.A, self.T, self.L = np.zeros((n, self.LMAX)), np.zeros((n, self.LMAX)), np.zeros(n, int)
        super().reset(ids)
        for i in ids:
            assert len(self.paths[i]) <= self.LMAX
            self._store(i)

    def _senses(self):
        q = self.sens[:, self.adr["orientation"]:self.adr["orientation"] + 4]
        w, x, y, z = q.T
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        g = self.sens[:, self.adr["angular_velocity"]:self.adr["angular_velocity"] + 3]
        return roll, yaw, g, self.sens[:, self.adr["velocity"]], self.state[:, self.steer_q], self.state[:, 1:3]

    def _progress(self, xy):
        idx = np.minimum(self.k[:, None] + np.arange(40)[None], self.L[:, None] - 1)
        pts = np.take_along_axis(self.P, idx[..., None], axis=1)
        j = np.argmin(np.linalg.norm(pts - xy[:, None], axis=2), axis=1)
        return idx[np.arange(self.n), j]

    def _gather(self, arr, k):
        return np.take_along_axis(arr, np.minimum(k, self.L - 1)[:, None], axis=1)[:, 0]

    def _gather2(self, k):
        k = np.minimum(k, self.L - 1)
        return self.P[np.arange(self.n), k]

    def _observe_all(self, roll, yaw, g, speed, steer, xy):
        k = self.k
        t = self._gather2(k + 1) - self.P[np.arange(self.n), np.maximum(k - 1, 0)]
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
        cross = np.einsum("ij,ij->i", xy - self._gather2(k), np.stack([-t[:, 1], t[:, 0]], 1))
        heads = []
        for o in HEAD_AHEAD:
            p = self._gather2(k + o)
            h = (np.arctan2(p[:, 1] - xy[:, 1], p[:, 0] - xy[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
            heads.append(np.where(np.linalg.norm(p - xy, axis=1) > 0.2, h, 0.0))
        kaps = [self._gather(self.K, k + o) for o in KAPPA_AHEAD]
        return np.column_stack([roll, g[:, 0], g[:, 2], speed, steer, cross, *heads, *kaps,
                                self.prev]).astype(np.float32)

    def observe(self):
        roll, yaw, g, speed, steer, xy = self._senses()
        return self._observe_all(roll, yaw, g, speed, steer, xy)

    def _rule_all(self, roll, yaw, g, speed, xy):
        k_roll, k_rate, k_head, k_ff, k_speed, look, cruise, corner = self.residual_gains[:8]
        tight = self.residual_gains[8] if len(self.residual_gains) > 8 else corner
        k = self._progress(xy)
        tgt = self._gather2(k + max(1, int(look / 0.5)))
        err = (np.arctan2(tgt[:, 1] - xy[:, 1], tgt[:, 0] - xy[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        v = np.maximum(speed, 0.5)
        kap = self._gather(self.K, k + np.maximum(1, (0.4 * v / 0.5).astype(int)))
        span = int(2 * look / 0.5) + 1
        idx = np.minimum(k[:, None] + np.arange(span)[None], self.L[:, None] - 1)
        ahead = np.take_along_axis(self.T, idx, axis=1).max(1)
        v_t = np.where(ahead > 0.5, np.where(ahead > 1.17, tight, corner), cruise)
        lean = np.clip(-(np.arctan(v * v * kap / 9.81) + k_head * err), -0.35, 0.35)
        torque = np.clip(k_speed * (v_t - speed), -12, 12)
        steering = np.clip(k_ff * 0.98 * kap - k_roll * (roll - lean) - k_rate * g[:, 0], -0.6, 0.6)
        return steering, torque

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        roll0, yaw0, g0, speed0, _, xy0 = self._senses()
        if self.residual_gains is not None:
            st, tq = residual_controls(a, *self._rule_all(roll0, yaw0, g0, speed0, xy0))
        else:
            st, tq = to_controls(a)
        self.ctrl[:, :, self.a_steer] = st[:, None]
        self.ctrl[:, :, self.a_pedal] = tq[:, None]
        rollout.rollout([self.m] * self.n, self.datas, self.state, self.ctrl,
                        control_spec=int(mujoco.mjtState.mjSTATE_CTRL), nstep=self.substeps, skip_checks=True,
                        state=self.out_state, sensordata=self.out_sens, persistent_pool=True)
        self.state = self.out_state[:, -1].copy()
        self.sens = self.out_sens[:, -1].copy()
        self.steps += 1
        da = a - self.prev
        k0 = self.k.copy()
        roll, yaw, g, speed, steer, xy = self._senses()
        self.k = self._progress(xy)
        self.prev = a.astype(np.float32)
        obs = self._observe_all(roll, yaw, g, speed, steer, xy)
        cross = np.abs(obs[:, 5])
        gain = self._gather(self.A, self.k) - self._gather(self.A, k0)
        rew = (2.0 * gain * np.exp(-(cross / 0.75) ** 2) - 0.1 * cross ** 2 - 0.02 * (da * da).sum(1)
               - 0.05 * np.maximum(0.0, obs[:, 3] - 3.5)).astype(np.float32)
        fell = (np.abs(roll) > 0.7) | (self.state[:, 3] < self.z0 - 0.12) | (cross > OFF_ROUTE_M) \
            | ~np.isfinite(self.state).all(1)
        finished = ~fell & (self.k >= self.L - 3)
        trunc = ~fell & ~finished & (self.steps >= self.max_steps)
        rew = rew - 5.0 * fell + 2.0 * finished
        done = fell | finished | trunc
        info = {"fell": fell, "finished": finished, "trunc": trunc}
        ids = np.nonzero(done)[0]
        if len(ids):
            info["final_obs"] = obs[ids].copy()
            info["done_ids"] = ids
            self.reset(ids)
            obs[ids] = self.observe()[ids]
        return obs, rew, done, info
