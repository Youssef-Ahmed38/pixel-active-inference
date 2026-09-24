"""Scripted reach -> grasp -> pull through the common embodiment interface (a sanity check per body).

Only the common observation and action are used, plus the privileged fixture state (where the handle is,
its bar axis, approach and opening direction), so one script serves every body:

1. pregrasp: palm to `standoff` in front of the handle, oriented for a bar grasp (embodiment.grasp_rot);
2. approach: straight in along the approach direction until the grasp centre is on the bar;
3. close: grasp command -1 for a fixed time;
4. pull: a constant velocity along the handle's opening direction (its Jacobian column; speed and duration
   sized by the fixture's travel, see _pull_budget), plus a
   P-correction across it that keeps the grasp where it was on the handle; the orientation follows the
   handle frame, so a door handle is carried round its arc. The constant velocity (rather than a fixed
   lead) keeps pulling through a sticky start: the embodiment's leash bounds the force it builds up.

Palm commands are a P-controller on the measured palm pose, clipped to the embodiment's speed limits.
"""

from __future__ import annotations

import numpy as np

from pai.envs.embodiments import rot_error

PULL_MAX_SPEED = 0.1    # m/s; faster pulls cost the parallel grippers their grip


class ScriptedReachGraspPull:
    def __init__(self, env, standoff: float = 0.1, kp: float = 4.0, kr: float = 3.0,
                 close_steps: int = 40, pull_speed: float | None = None, pull_steps: int | None = None,
                 open_frac: float = 0.6):
        """pull_speed / pull_steps default to a budget sized by the fixture: the handle's travel to
        open_frac of the joint range (arc length for hinges), at up to PULL_MAX_SPEED, so a 0.9 m room door
        opens as fully as a small cabinet door."""
        self.open_frac = open_frac
        self.env, self.emb = env, env.emb
        self.standoff, self.kp, self.kr = standoff, kp, kr
        self.close_steps, self._speed_arg, self._steps_arg = close_steps, pull_speed, pull_steps
        self.pull_speed, self.pull_steps = pull_speed or 0.06, pull_steps or 300
        self.reset()

    def _pull_budget(self) -> None:
        """Handle travel to open_frac of the range -> pull speed and number of steps."""
        env = self.env
        m, d = env.model, env.data
        jid = m.joint(env.fixture.joint).id
        lo, hi = env.fixture.open_range
        span = self.open_frac * (hi - lo)
        if m.jnt_type[jid] == 3:   # hinge: arc length at the handle's radius
            hp = d.site_xpos[env.handle_site]
            ax, an = d.xaxis[jid], d.xanchor[jid]
            rel = hp - an
            span *= float(np.linalg.norm(rel - np.dot(rel, ax) * ax))
        speed = self._speed_arg or float(np.clip(span / 10.0, 0.06, PULL_MAX_SPEED))
        self.pull_speed = speed
        self.pull_steps = self._steps_arg or int(span / (speed * env.control_dt)) + 100

    def reset(self):
        self.phase = "pregrasp"
        self.k = 0
        self.log: dict = {"reached": False, "reach_err": np.inf, "grasped": False, "contact_fingers": 0,
                          "lost_grip": False}
        self._grasp_R = None
        self._R_rel = None
        self._lost = 0

    # ------------------------------------------------------------------ helpers
    def _handle(self, obs):
        f = obs["fixture"]
        return f["handle_pos"], f["handle_axis"], f["handle_approach"], f["open_dir"]

    def _cmd(self, obs, pos, R, grasp):
        v = self.kp * (pos - obs["palm_pos"])
        w = self.kr * rot_error(obs["palm_rot"], R)
        g = np.atleast_1d(grasp).astype(float)
        return np.concatenate([v, w, g])

    def done(self) -> bool:
        return self.phase == "done"

    # ------------------------------------------------------------------ policy
    def act(self, obs) -> np.ndarray:
        hp, axis, appr, odir = self._handle(obs)
        self.k += 1
        if self._grasp_R is None:
            self._grasp_R = self.emb.grasp_rot(axis, appr, obs["palm_rot"])
        R = self._grasp_R
        z = R[:, 2]  # the (possibly rolled) approach direction
        if self.phase == "pregrasp":
            target = hp - self.standoff * z
            err = np.linalg.norm(target - obs["palm_pos"])
            rerr = np.linalg.norm(rot_error(obs["palm_rot"], R))
            if (err < 0.02 and rerr < 0.15) or self.k > 200:
                self.phase, self.k = "approach", 0
            return self._cmd(obs, target, R, 1.0)
        if self.phase == "approach":
            err = np.linalg.norm(hp - obs["palm_pos"])
            self.log["reach_err"] = float(err)
            if err < 0.006 or self.k > 120:
                self.log["reached"] = bool(err < 0.03)
                self.phase, self.k = "close", 0
            return self._cmd(obs, hp, R, 1.0)
        if self.phase == "close":
            if self.k == 1:
                self._hold = obs["palm_pos"].copy()   # close in place: do not push the hand on
            if self.k >= self.close_steps:
                self.log["contact_fingers"] = self.env.handle_contact_fingers()
                self.log["grasped"] = self.log["contact_fingers"] >= 2
                self.phase, self.k = "pull", 0
                self._pull_budget()
                # keep the palm-handle relative rotation fixed while pulling
                Rh = self._handle_rot(obs)
                self._R_rel = Rh.T @ obs["palm_rot"]
                self._p_rel = Rh.T @ (obs["palm_pos"] - hp)
            return self._cmd(obs, self._hold, R, -1.0)
        if self.phase == "pull":
            # stop when the pull is over, or when the hand has lost the handle for a while
            self._lost = self._lost + 1 if obs["finger_force"].sum() < 0.5 else 0
            if self.k >= self.pull_steps or self._lost > 15:
                self.log["lost_grip"] = self._lost > 15
                self.phase = "done"
            Rh = self._handle_rot(obs)
            a = self._cmd(obs, hp + Rh @ self._p_rel, Rh @ self._R_rel, -1.0)
            a[:3] = a[:3] - np.dot(a[:3], odir) * odir + self.pull_speed * odir
            return a
        return self._cmd(obs, obs["palm_pos"], obs["palm_rot"], -1.0)

    @staticmethod
    def _handle_rot(obs):
        return obs["fixture"]["handle_rot"]


def run_episode(env, seed: int = 0, max_steps: int = 2000, frames: list | None = None, camera: str = "front",
                frame_every: int = 4, frame_size: int = 320) -> dict:
    """Run the scripted behaviour once; returns reached / grasped / opening metrics."""
    obs = env.reset(seed=seed)
    policy = ScriptedReachGraspPull(env)
    stable = True
    for t in range(max_steps):
        obs = env.step(policy.act(obs))
        if frames is not None and t % frame_every == 0:
            frames.append(env.render(camera, frame_size))
        if not env.stable():
            stable = False
            break
        if policy.done():
            break
    f = obs["fixture"]
    return {**policy.log, "opening": f["opening"], "opening_frac": f["opening_frac"], "opened": bool(f["opened"]),
            "stable": stable, "steps": t + 1}
