"""Scripted opener for articulated fixtures with the Panda's end-effector action interface.

Not an agent: it uses the privileged obs["articulations"] entry of one part (handle pose, face
normal, pull direction) to check that the compliant arm can physically open each fixture family,
and to collect demonstrations. Phases: above -> pre-grasp -> reach -> close -> (turn the lever) ->
pull along the part's motion -> release -> retreat.

The hand always points down (yaw-only control), which fixes the grasp per handle shape:
- front  bar / knob on a vertical face: come in from the front, fingers close across the face
         (along the handle frame's y); the pull is held by friction, or by the fingers for slides
         along the closing axis (sliding doors)
- top    knob / tab on a lid (face normal up): descend onto it, fingers close along the hinge axis
- normal hbar / lever near the top of a face: descend from above with the fingers half open, close
         along the face normal so the back finger hooks the bar (the hand passes over the top)
- pinch  flush fin at the top edge: descend just in front of the face and pinch the fin
Pulling steers the commanded target to a point a little ahead of the handle along its current
motion direction, so hinged parts are followed along their arc and a locked part is pulled with a
bounded force (the commanded lead) until the pull stalls.
"""

from __future__ import annotations

import numpy as np

HOVER = 0.1          # approach height above the pre-grasp point (m)
FRONT_OFFSET = 0.09  # pre-grasp distance in front of a handle (m)
LEAD = 0.035         # how far ahead of the handle the pull target runs (m)
OPEN_GOAL = 0.9      # stop pulling at this opening fraction
LEVER_GOAL = 0.6     # turn the lever to this fraction of its travel first
STALL_STEPS = 50     # stop when the opening has not improved by STALL_DELTA for this many steps
STALL_DELTA = 0.01
PULL_TIMEOUT = 600
REACH_TIMEOUT = 250  # steps per approach waypoint
DOOR_SWEEP = np.pi / 2  # assumed door travel when choosing the start yaw (the range is not observed)
RETREAT = 0.15       # horizontal back-off after a front grasp (m)
POS_TOL = 0.008
YAW_LIMIT = np.pi / 2 - 0.02  # the controller clips its yaw target to +-pi/2
HALF_OPEN = -0.3     # grip action with fingers ~1.4 cm from the centre (hooking hbar / lever grasps)


def grasp_mode(a: dict) -> str | None:
    """How the down-pointing parallel gripper takes this handle, or None if it cannot."""
    if a["handle"] in ("hbar", "lever"):
        return "normal"
    if a["handle"] == "flush":
        return "pinch"
    if abs(a["normal"][2]) > 0.7:
        return "top" if a["handle"] in ("knob", "tab", "bar") else None
    return "front" if a["handle"] in ("bar", "knob") else None


def _horizontal(v: np.ndarray) -> np.ndarray:
    h = np.array([v[0], v[1], 0.0])
    n = np.linalg.norm(h)
    return h / n if n > 1e-6 else h


def _yaw_for_closing(c: np.ndarray) -> float:
    """EE yaw whose finger-closing axis is the horizontal direction c: at yaw g the fingers close
    along (sin g, -cos g, 0)."""
    return float(np.arctan2(c[0], -c[1]))


def _wrap_near(g: float, ref: float) -> float:
    """Equivalent yaw (mod pi: the gripper is symmetric) closest to ref."""
    return float(ref + (g - ref + np.pi / 2) % np.pi - np.pi / 2)


class OpenPart:
    """Opens one part (e.g. "drawer") of an ArticulatedEnv. act(obs) -> action; done() when finished."""

    def __init__(self, env, part: str, gain: float = 6.0, open_goal: float = OPEN_GOAL):
        self.env, self.part, self.gain, self.open_goal = env, part, gain, open_goal
        self.phase, self.timer = "plan", 0
        self.mode: str | None = None
        self.best, self.best_t, self.pull_t = 0.0, 0, 0
        self.result = ""  # how it ended: opened | stalled | timeout | unreachable | unsupported

    def done(self) -> bool:
        return self.phase == "done"

    # ------------------------------------------------------------------ geometry
    def _closing_axis(self, a: dict) -> np.ndarray:
        if self.mode == "normal":
            return _horizontal(a["normal"])
        return _horizontal(np.cross(a["handle_axis"], a["normal"]))  # the handle frame's y axis

    def _grasp_point(self, a: dict) -> np.ndarray:
        p = a["handle_pos"].copy()
        if self.mode == "pinch":  # keep the finger backs ~1 cm off the face
            p += _horizontal(a["normal"]) * 0.004
        return p

    def _plan(self, a: dict) -> None:
        self.mode = grasp_mode(a)
        if self.mode is None:
            self.phase, self.result = "done", "unsupported"
            return
        g = _yaw_for_closing(self._closing_axis(a))
        # A door on a vertical hinge turns the hand with it: pick the equivalent start yaw that keeps
        # the whole sweep inside the controller's +-pi/2 yaw range.
        sweep = 0.0
        if a["kind"] == "hinge" and abs(a["joint_axis"][2]) > 0.7:
            sweep = float(np.sign(a["joint_axis"][2]) * DOOR_SWEEP * self.open_goal)
        cands = [g + k * np.pi for k in (-2, -1, 0, 1, 2)]

        def overflow(c: float) -> float:
            lo, hi = min(c, c + sweep), max(c, c + sweep)
            return max(0.0, hi - YAW_LIMIT) + max(0.0, -YAW_LIMIT - lo)

        self._yaw_ref = min(cands, key=lambda c: (overflow(c), abs(c - self.env.ee_target_yaw)))
        n = _horizontal(a["normal"])
        self.approach_dir = n
        grasp = self._grasp_point(a)
        if self.mode == "front":
            pre = grasp + n * FRONT_OFFSET
        else:
            pre = grasp + np.array([0, 0, 0.05])
        self.waypoints = {"above": pre + np.array([0, 0, HOVER]), "pre": pre, "reach": grasp}
        self.phase = "above"

    # ------------------------------------------------------------------ control
    def _move(self, goal: np.ndarray, grip: float, yaw: float) -> np.ndarray:
        env = self.env
        vel = np.clip(self.gain * (goal - env.ee_target_pos), -env.cfg.max_ee_vel, env.cfg.max_ee_vel)
        yaw_err = yaw - env.ee_target_yaw
        return np.r_[vel, np.clip(3.0 * yaw_err, -1, 1), grip]

    def act(self, obs: dict) -> np.ndarray:
        a = obs["articulations"][self.part]
        env = self.env
        if self.phase == "plan":
            self._plan(a)
        if self.done():
            return np.array([0, 0, 0, 0, 1.0])
        open_grip = HALF_OPEN if self.mode == "normal" else 1.0
        close_grip = -1.0
        g = _wrap_near(_yaw_for_closing(self._closing_axis(a)), self._yaw_ref)
        self._yaw_ref = g  # unclipped, so a sweep past the limit does not flip to the other side
        yaw = float(np.clip(g, -YAW_LIMIT, YAW_LIMIT))
        ee = obs["ee_pos"]

        if self.phase in ("above", "pre", "reach"):
            goal = self.waypoints[self.phase]
            if self.phase == "reach":  # track the handle (it may be a little ajar)
                goal = self._grasp_point(a)
            self.timer += 1
            if np.linalg.norm(goal - ee) < POS_TOL and abs(yaw - env.ee_target_yaw) < 0.05:
                self.phase, self.timer = {"above": "pre", "pre": "reach", "reach": "close"}[self.phase], 0
            elif self.timer > REACH_TIMEOUT:  # blocked on the way (e.g. the wrist hits a handle above)
                self.phase, self.timer, self.result = "release", 0, "unreachable"
                self.release_pos = ee.copy()
            return self._move(goal, open_grip, yaw)

        if self.phase == "close":
            self.timer += 1
            if self.timer > 25:
                self.timer = 0
                self.phase = "turn" if "lever" in a else "pull"
                self.best, self.best_t, self.pull_t = a.get("lever", a)["turn" if "lever" in a else "opening"], 0, 0
            return self._move(self._grasp_point(a), close_grip, yaw)

        if self.phase == "turn":  # press / lift the lever along its arc until the latch is free
            lv = a["lever"]
            self._track(lv["turn"])
            if lv["turn"] >= LEVER_GOAL or self._stalled():
                self.phase = "pull"
                self.best, self.best_t, self.pull_t = a["opening"], 0, 0
            goal = a["handle_pos"] + LEAD * lv["pull_dir"]
            return self._move(goal, close_grip, yaw)

        if self.phase == "pull":
            self._track(a["opening"])
            if a["opening"] >= self.open_goal or self._stalled() or self.pull_t > PULL_TIMEOUT:
                self.result = ("opened" if a["opening"] >= self.open_goal else
                               "timeout" if self.pull_t > PULL_TIMEOUT else "stalled")
                self.phase, self.timer = "release", 0
                self.release_pos = ee.copy()
            goal = a["handle_pos"] + LEAD * a["pull_dir"]
            if "lever" in a and a["opening"] < 0.1 and a["lever"]["turn"] < LEVER_GOAL:  # keep it turned until ajar
                goal = goal + 0.5 * LEAD * a["lever"]["pull_dir"]
            return self._move(goal, close_grip, yaw)

        if self.phase == "release":
            self.timer += 1
            if self.timer > 20:
                self.phase = "back_off"
                # Front grasps back off along the approach direction first (the hand may be under an
                # open flap or beside a door), then everything goes up.
                back = self.approach_dir * RETREAT if self.mode == "front" else np.zeros(3)
                self.retreats = [self.release_pos + back, self.release_pos + back + np.array([0, 0, 0.08])]
            return np.r_[0.0, 0.0, 0.0, 0.0, open_grip]

        if self.phase in ("back_off", "retreat"):
            goal = self.retreats[self.phase == "retreat"]
            self.timer += 1
            if np.linalg.norm(goal - ee) < 3 * POS_TOL or self.timer > 150:
                self.phase, self.timer = ("retreat", 0) if self.phase == "back_off" else ("done", 0)
            return self._move(goal, open_grip, env.ee_target_yaw)
        raise RuntimeError(self.phase)

    def _track(self, value: float) -> None:
        self.pull_t += 1
        if value > self.best + STALL_DELTA:
            self.best, self.best_t = value, self.pull_t

    def _stalled(self) -> bool:
        return self.pull_t - self.best_t > STALL_STEPS


def run_open(env, part: str, max_steps: int = 1500, frames: list | None = None, render=None,
             frame_every: int = 2, hold_steps: int = 25) -> dict:
    """Runs OpenPart from the env's current (freshly reset) state and returns a summary; `opened` is
    whether an articulation_opened event was logged. frames/render optionally record images."""
    obs = env._observe()
    policy = OpenPart(env, part)
    peak = 0.0
    for t in range(max_steps + hold_steps):
        holding = policy.done() or t >= max_steps
        obs = env.step(np.array([0, 0, 0, 0, 1.0]) if holding else policy.act(obs))
        peak = max(peak, obs["articulations"][part]["opening"])
        if frames is not None and t % frame_every == 0:
            frames.append(render())
        if holding:  # after the release: let spring return / gravity act for hold_steps
            hold_steps -= 1
            if hold_steps <= 0:
                break
    opened = any(e["part"] == part for e in env.events.of_type("articulation_opened"))
    return {"opened": opened, "peak": peak, "final": obs["articulations"][part]["opening"],
            "result": policy.result, "steps": env.t}
