"""What every skill shares: targets, the skill context (one body in one scene), results and stall detection.

A skill never knows what kind of fixture it acts on. It is given a `Target`: a frame in the scene (a
MuJoCo site or body, or a fixed pose) with two directions in that frame, the part's `axis` (bar axis,
knob / key / keyhole axis) and its `approach` (the direction a hand moves to reach it). A handle, a
knob, a keyhole, a key and a loose block are all just targets; what a skill does to them is for the
agent to find out.

`SkillContext` wraps an env with the common embodiment interface (env.step(action) -> obs with palm_pos,
palm_rot, finger_force, ...; env.emb, env.model, env.data, env.control_dt) and keeps the state that
lives between skills: the grasp command (a closed hand stays closed from grasp() to release()) and the
object currently held. It also senses the net contact force on the hand (the vector sum over its
contacts with the world, so a squeeze cancels and a pull or push does not), which skills report and
push_pull limits.

A skill returns a `SkillResult`: its own success criterion, final errors, contact forces seen,
duration, and `stalled`: motion was commanded but did not happen (`StallMonitor`). A stalled pull on a
grasped handle is the observation that later tells the agent a door is locked.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from pai.envs.embodiments import _is_descendant, rot_error


# ---------------------------------------------------------------------------------------------- targets
@dataclass
class Target:
    """A frame to act on: a site (a part: handle, knob, keyhole, key tip), a body (a movable object) or,
    with neither, the fixed world pose (pos, rot). Directions are unit vectors in the target frame."""
    name: str
    site: str | None = None
    body: str | None = None
    pos: tuple | None = None
    rot: np.ndarray | None = None
    axis_local: tuple = (0.0, 0.0, 1.0)       # part axis (bar, knob, keyhole, key shaft)
    approach_local: tuple = (1.0, 0.0, 0.0)   # direction a hand moves to reach it (into the part)
    kind: str = "part"                        # "part" (on a fixture) | "object" (movable, can be held)

    def frame(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        if self.site is not None:
            i = model.site(self.site).id
            return data.site_xpos[i].copy(), data.site_xmat[i].reshape(3, 3).copy()
        if self.body is not None:
            i = model.body(self.body).id
            return data.xpos[i].copy(), data.xmat[i].reshape(3, 3).copy()
        R = np.eye(3) if self.rot is None else np.asarray(self.rot, float)
        return np.asarray(self.pos, float).copy(), R.copy()

    def world(self, model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(position, rotation, axis, approach) in the world frame."""
        p, R = self.frame(model, data)
        return p, R, _unit(R @ np.asarray(self.axis_local, float)), _unit(R @ np.asarray(self.approach_local, float))


def handle_target(info, name: str | None = None) -> Target:
    """Target for a fixture handle described by a HandleInfo-like record (handle_site, axis_local,
    approach_local), e.g. DoorSceneEnv.fixture. Only the geometry is used, never the fixture kind."""
    return Target(name or f"{info.name}:handle", site=info.handle_site, axis_local=tuple(info.axis_local),
                  approach_local=tuple(info.approach_local))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------------------------- results
@dataclass
class SkillResult:
    skill: str
    target: str | None
    success: bool                  # by the skill's own criterion
    stalled: bool                  # commanded motion, no progress (e.g. pulling a locked drawer)
    reason: str                    # "ok" | "stalled" | "timeout" | "force_limit" | "no_contact" | "unstable" | ...
    errors: dict = field(default_factory=dict)       # final errors / achieved quantities of this skill
    max_force: float = 0.0         # max |net contact force on the hand| (N)
    mean_force: float = 0.0
    max_finger_force: float = 0.0  # max summed finger normal force (N): grip strength seen
    steps: int = 0
    duration: float = 0.0          # simulated seconds
    wall_time: float = 0.0

    def row(self) -> str:
        err = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.errors.items())
        return (f"{self.skill:10s} {str(self.target):18s} ok={int(self.success)} stalled={int(self.stalled)} "
                f"{self.reason:11s} F={self.max_force:5.1f}N t={self.duration:4.2f}s {err}")


class StallMonitor:
    """Stalled = over the last `window` steps, more than `min_cmd` of motion was commanded but less than
    `ratio` of it happened. Units are the skill's (m along a line, m of error reduction, rad)."""

    def __init__(self, window: int = 25, min_cmd: float = 0.02, ratio: float = 0.25):
        self.cmd: deque = deque(maxlen=window)
        self.got: deque = deque(maxlen=window)
        self.min_cmd, self.ratio = min_cmd, ratio
        self.stalled = False

    def update(self, commanded: float, achieved: float) -> bool:
        self.cmd.append(abs(commanded))
        self.got.append(achieved)
        c = sum(self.cmd)
        self.stalled = len(self.cmd) == self.cmd.maxlen and c > self.min_cmd and sum(self.got) < self.ratio * c
        return self.stalled

    def progress_ratio(self, last: int | None = None) -> float:
        """Achieved / commanded over the window (or its `last` steps)."""
        n = len(self.cmd) if last is None else min(last, len(self.cmd))
        c = sum(list(self.cmd)[-n:]) if n else 0.0
        return float(sum(list(self.got)[-n:]) / c) if c > 1e-9 else 1.0

    def stopped(self, last: int = 5) -> bool:
        """No progress over the last few steps (used when a force limit ends a motion early)."""
        return self.progress_ratio(last) < self.ratio


# ---------------------------------------------------------------------------------------------- context
class SkillContext:
    """One body in one scene, driven only through the common action. Keeps the grasp command and the
    held object between skills; `on_step(env, obs)` callbacks see every control step (recording)."""

    kp = 4.0      # palm position gain (1/s), as in ScriptedReachGraspPull
    kr = 3.0      # palm orientation gain (1/s)

    def __init__(self, env, obs: dict | None = None, on_step: Callable | None = None):
        self.env, self.emb = env, env.emb
        self.model, self.data = env.model, env.data
        self.dt = float(env.control_dt)
        self.grip = 1.0                         # [-1 closed, 1 open]
        self.held: Target | None = None
        self.on_step = [on_step] if on_step else []
        self.obs = obs if obs is not None else env.step(np.r_[np.zeros(6), self.grip])
        m = self.model
        palm_body = m.body(self.emb.p(self.emb.sd.palm_body)).id
        self.hand_body = np.array([_is_descendant(m, b, palm_body) for b in range(m.nbody)])
        self.own_body = self.emb.own_body
        self._f6 = np.zeros(6)

    # ------------------------------------------------------------------ acting
    def step(self, v: np.ndarray, w: np.ndarray, grip: float | None = None) -> dict:
        if grip is not None:
            self.grip = float(grip)
        a = np.r_[np.asarray(v, float), np.asarray(w, float), self.grip]
        self.obs = self.env.step(a)
        for cb in self.on_step:
            cb(self.env, self.obs)
        return self.obs

    def servo(self, pos, R) -> tuple[np.ndarray, np.ndarray]:
        """P-command (v, w) towards a palm pose, v clipped to the body's speed limit."""
        o = self.obs
        v = self.kp * (np.asarray(pos, float) - o["palm_pos"])
        n = np.linalg.norm(v)
        if n > self.emb.max_lin_vel:
            v *= self.emb.max_lin_vel / n
        return v, self.kr * rot_error(o["palm_rot"], R)

    def stable(self) -> bool:
        return bool(self.env.stable()) if hasattr(self.env, "stable") else True

    # ------------------------------------------------------------------ sensing
    def hand_force(self) -> np.ndarray:
        """Net contact force on the hand (palm subtree) from everything that is not this body (N, world)."""
        m, d = self.model, self.data
        F = np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            h1, h2 = self.hand_body[b1] and not self.own_body[b2], self.hand_body[b2] and not self.own_body[b1]
            if not (h1 or h2):
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            f = c.frame.reshape(3, 3).T @ self._f6[:3]    # force of geom1 on geom2, world frame
            F += f if h2 else -f
        return F


class Recorder:
    """Per-skill bookkeeping: forces seen, steps, timing; builds the SkillResult."""

    def __init__(self, ctx: SkillContext, skill: str, target: Target | None):
        self.ctx, self.skill, self.target = ctx, skill, target
        self.forces: list[float] = []
        self.finger: list[float] = []
        self.steps = 0
        self.t0 = time.time()
        self.unstable = False

    def step(self, v, w, grip=None) -> dict:
        obs = self.ctx.step(v, w, grip)
        self.steps += 1
        self.forces.append(float(np.linalg.norm(self.ctx.hand_force())))
        self.finger.append(float(np.sum(obs["finger_force"])))
        if not self.ctx.stable():
            self.unstable = True
        return obs

    def result(self, success: bool, stalled: bool, reason: str, **errors) -> SkillResult:
        if self.unstable:
            success, reason = False, "unstable"
        f = np.asarray(self.forces or [0.0])
        return SkillResult(self.skill, self.target.name if self.target is not None else None, bool(success),
                           bool(stalled), reason, {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                                   for k, v in errors.items()},
                           float(f.max()), float(f.mean()), float(max(self.finger or [0.0])), self.steps,
                           self.steps * self.ctx.dt, time.time() - self.t0)
