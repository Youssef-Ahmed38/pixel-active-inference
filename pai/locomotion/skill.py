"""Locomotion skill API for upper levels: ``move(velocity, heading)``, ``turn``, ``stop``.

Upper levels (maze explorer, bike, HRM thinker) never see joints: they set a goal, advance time
and read back pose, events and surprise. The learned controller (``policy.pt``) turns the goal
into leg servo targets every 20 ms. One ``LocomotionSkill`` drives one body.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pai.locomotion.env import EnvConfig, LocomotionEnv
from pai.locomotion.falls import LocomotionEventLog
from pai.locomotion.policy import LocomotionController


class LocomotionSkill:
    def __init__(self, policy_path: Path, seed: int = 0, cfg: EnvConfig | None = None,
                 controller: LocomotionController | None = None):
        cfg = cfg or EnvConfig(randomize=False, nthread=1, episode_s=1e6)
        self.env = LocomotionEnv(1, cfg, seed=seed, stage=len(cfg.stages) - 1)
        self.env.pushes_enabled = False
        self.env.set_command([0], 0.0, 0.0, 0.0)
        self.ctrl = controller or LocomotionController(policy_path)
        self.obs = self.env.observe()
        self.events = LocomotionEventLog(0)
        self.t = 0.0
        self.falls = 0

    # ------------------------------------------------------------------ goals
    def move(self, velocity: float, heading: float | None = None, lateral: float = 0.0):
        """Walk (|v| < ~1 m/s) or run (faster) at ``velocity`` m/s, optionally holding ``heading``
        (rad, world frame). Walking and running are the same skill with a different goal."""
        self.env.set_command([0], velocity, lateral, 0.0, heading=heading)

    def turn(self, yaw_rate: float | None = None, heading: float | None = None, velocity: float = 0.0):
        if heading is not None:
            self.env.set_command([0], velocity, 0.0, 0.0, heading=heading)
        else:
            self.env.set_command([0], velocity, 0.0, float(yaw_rate or 0.0))

    def stop(self):
        self.env.set_command([0], 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ time
    def run(self, seconds: float) -> dict:
        """Advance ``seconds``; returns the pose and whether the body fell (it is then reset)."""
        fell_any = False
        for _ in range(int(round(seconds / self.env.cfg.ctrl_dt))):
            a = self.ctrl.act(self.obs)
            self.obs, _, _, info = self.env.step(a)
            self.t += self.env.cfg.ctrl_dt
            fell = bool(info["fell"][0])
            sensors = info["final_sensors"] if fell else None
            h, v = self.env.soles(sensors)
            self.events.observe(self.t, h[0], v[0], self.env.cfg.contact_height, fell=fell)
            if fell:
                self.falls += 1
                fell_any = True
        return {**self.pose(), "fell": fell_any}

    def pose(self) -> dict:
        qp = self.env.qpos()[0]
        k = self.env.kinematics()
        return {"x": float(qp[0]), "y": float(qp[1]), "heading": float(k["yaw"][0]),
                "speed": float(np.linalg.norm(k["v_head"][0, :2])), "t": self.t}
