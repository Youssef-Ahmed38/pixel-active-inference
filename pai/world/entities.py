"""Entity-token encoding of the scene: one token for the gripper, one per object.

This is the body-independent interface to the upper levels: the world model, goals and cause
inference only see entity tokens. In the vertical slice the tokens come from privileged
simulator state; later they come from object slots learned from pixels, with the same layout.

Token layout (TOKEN_DIM features):
    pos (3) | vel (3) | yaw sin, cos (2) | gripper opening (1) | kind one-hot (4) | colour (3) | force (3)

`force` is the wrist force sensor (world frame, N) on the gripper token and zero for objects. It is
a *sensation*: the world model predicts it from the state but never reads it as an input, so a
held object that is heavier than expected, or a push, shows up as a persistent force error
instead of being silently copied into the prediction.
"""

from __future__ import annotations

import numpy as np

KINDS = ("gripper", "block", "plate", "bowl")
POS, VEL, YAW, GRIP, KIND, COLOR, FORCE = (slice(0, 3), slice(3, 6), slice(6, 8), slice(8, 9), slice(9, 13),
                                           slice(13, 16), slice(16, 19))
TOKEN_DIM = 19
DYNAMIC = slice(0, 9)  # features the world model predicts as changes; kind and colour are constant


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def entity_names(env) -> list[str]:
    return ["gripper", *env.objects]


def encode(env, obs: dict, prev_ee: np.ndarray | None = None) -> np.ndarray:
    """(n_entities, TOKEN_DIM) float32 tokens from an observation of a TabletopEnv."""
    tokens = np.zeros((1 + len(env.objects), TOKEN_DIM), np.float32)
    g = tokens[0]
    g[POS] = obs["ee_pos"]
    if prev_ee is not None:
        g[VEL] = (obs["ee_pos"] - prev_ee) / env.control_dt
    g[YAW] = [np.sin(env.ee_target_yaw), np.cos(env.ee_target_yaw)]
    g[GRIP] = obs["gripper"]
    g[KIND] = [1, 0, 0, 0]
    g[FORCE] = obs["wrist_force"]
    rgba = {b.name: b.rgba for b in env.cfg.blocks}
    for i, (name, spec) in enumerate(env.objects.items(), start=1):
        st = obs["state"][name]
        t = tokens[i]
        t[POS] = st["pos"]
        t[VEL] = st["vel"]
        yaw = _yaw(st["quat"])
        t[YAW] = [np.sin(yaw), np.cos(yaw)]
        t[KIND] = np.eye(4)[KINDS.index(spec.kind)]
        colour = rgba[name] if name in rgba else env.cfg[spec.kind].rgba
        t[COLOR] = colour[:3]
    return tokens
