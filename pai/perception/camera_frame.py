"""Objects are seen through a camera whose pose the agent believes it knows.

Privileged object state is exact, so by itself it cannot be fooled by a bumped camera. A real agent
sees objects in the camera's frame and converts them to the world frame with the camera pose it was
calibrated with. This module applies exactly that conversion to the true object positions:

    seen_world = R_believed R_actual^T (x - p_actual) + p_believed

With an undisturbed camera the two poses coincide and nothing changes. After a camera shift, every
object seen appears displaced (by -offset for a pure translation), while the gripper, sensed through
proprioception, does not move. That mismatch is the signature the agent uses to infer the cause and
to recalibrate: it adds its estimated correction to everything it sees.
"""

from __future__ import annotations

import mujoco
import numpy as np

from pai.world.entities import POS, encode


def _rot(quat: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat, float))
    return m.reshape(3, 3)


class CameraFramePerception:
    """The `perceive` interface of the slice agent: privileged objects, but seen through the camera."""

    def __init__(self, env):
        self.env = env
        self.correction = np.zeros(3, np.float32)  # the agent's own recalibration, added to seen objects

    def reset(self) -> None:
        self.correction[:] = 0.0

    def __call__(self, obs: dict, prev_ee: np.ndarray | None = None) -> np.ndarray:
        env = self.env
        tokens = encode(env, obs, prev_ee)
        c = env.cam_id
        p_act, p_bel = env.model.cam_pos[c], env._nominal["cam_pos"][c]
        r_act, r_bel = _rot(env.model.cam_quat[c]), _rot(env._nominal["cam_quat"][c])
        x = tokens[1:, POS]
        tokens[1:, POS] = (x - p_act) @ r_act @ r_bel.T + p_bel + self.correction
        # velocities stay as measured: a static camera error does not change how fast things move
        return tokens
