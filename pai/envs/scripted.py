"""Scripted pick-and-place with end-effector control, using privileged object poses.

Not an agent: it checks that grasping physics and IK work, and it produces demonstrations
and known-good "success" episodes for the recipe and credit-assignment experiments.
"""

from __future__ import annotations

import numpy as np

from pai.envs.predicates import ObjectSpec

HOVER = 0.12        # height above the object for approach and retreat (m)
LIFT_Z = 0.2        # absolute carry height after grasping (m)
PLACE_GAP = 0.02    # release height above the target surface (m); lower and the fingertips pinch thin objects
GRASP_RAISE = 0.006 # grasp slightly above the block centre so the fingertips clear the table (m)
POS_TOL = 0.008     # waypoint tolerance (m)


def _yaw_of(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _grasp_yaw(block_yaw: float) -> float:
    """Closest yaw aligned with a cube face (cubes are symmetric every 90 degrees)."""
    return float((block_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)


class PickPlace:
    """Phases: above object -> descend -> close -> lift -> above target -> descend -> open -> retreat."""

    def __init__(self, env, obj: str, target: str, gain: float = 6.0):
        self.env, self.obj, self.target, self.gain = env, obj, target, gain
        self.phase, self.timer = 0, 0

    def _target_top(self, state) -> float:
        spec: ObjectSpec = self.env.objects[self.target]
        base = state[self.target]["pos"][2]
        if spec.kind == "bowl":
            return base + 0.01  # drop into the bowl just above its base
        return base + spec.height / 2

    def done(self) -> bool:
        return self.phase >= 8

    def act(self, obs: dict) -> np.ndarray:
        st, ee = obs["state"], obs["ee_pos"]
        env = self.env
        obj_pos = st[self.obj]["pos"]
        h_obj = env.objects[self.obj].height
        tgt_xy = st[self.target]["pos"][:2]
        place_z = self._target_top(st) + h_obj / 2 + PLACE_GAP

        waypoints = {
            0: (np.r_[obj_pos[:2], obj_pos[2] + HOVER], 1.0),
            1: (np.r_[obj_pos[:2], obj_pos[2] + GRASP_RAISE], 1.0),
            2: (None, -1.0),                                   # close
            3: (np.r_[ee[:2], LIFT_Z], -1.0),  # absolute: the block rises with the hand
            4: (np.r_[tgt_xy, place_z + HOVER], -1.0),
            5: (np.r_[tgt_xy, place_z], -1.0),
            6: (None, 1.0),                                    # open
            7: (np.r_[ee[:2], place_z + HOVER], 1.0),
        }
        if self.done():
            return np.array([0, 0, 0, 0, 1.0])
        goal, grip = waypoints[self.phase]
        yaw_err = 0.0
        if self.phase <= 1:
            yaw_err = _grasp_yaw(_yaw_of(st[self.obj]["quat"])) - env.ee_target_yaw
        if goal is None:  # wait for the gripper to finish moving
            self.timer += 1
            if self.timer > 25:
                self.phase, self.timer = self.phase + 1, 0
            return np.array([0, 0, 0, 0, grip])
        # Steer the commanded target, not the measured hand: with a compliant arm the hand lags the
        # command, and steering by the measured error winds the target up past the waypoint.
        if np.linalg.norm(goal - ee) < POS_TOL:
            self.phase += 1
        vel = np.clip(self.gain * (goal - env.ee_target_pos), -env.cfg.max_ee_vel, env.cfg.max_ee_vel)
        return np.r_[vel, np.clip(3.0 * yaw_err, -1, 1), grip]
