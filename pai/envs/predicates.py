"""Relation predicates computed from privileged simulator state.

These are the ground truth for relational goals ("red_block ON plate") and for scoring the
agent's own relation classifier later. Tolerances are deliberately loose enough to be
stable under contact jitter, and tight enough that a block resting beside a plate is not "on" it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

ON_Z_TOL = 0.015      # gap between a's bottom and b's top that still counts as resting on it (m)
NEAR_DIST = 0.10      # centre distance for near(a, b) (m)
LEFT_MARGIN = 0.03    # world-y margin for left_of(a, b) (m); +y is the robot's left
UPRIGHT_COS = 0.95    # cos of the tilt angle for upright(a)
LIFTED_Z = 0.03       # bottom height above the table for lifted(a) (m)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    kind: str        # block | plate | bowl
    radius: float    # half-extent in xy (block half-size, plate/bowl radius)
    height: float


def _up_cos(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(1 - 2 * (x * x + y * y))  # z-component of the body z-axis


def compute_predicates(env, state: dict[str, dict]) -> set[tuple]:
    """Returns the set of true relations, e.g. {("on", "red", "plate"), ("grasped", "red")}.
    Ground truth from simulator state: for goals' scoring and the event log, never the agent's belief
    (the agent's own relations come from `perceived_predicates`)."""
    grasped = env.events.grasped_objects() if hasattr(env, "events") else set()
    positions = {n: np.asarray(st["pos"]) for n, st in state.items()}
    upright = {n: _up_cos(st["quat"]) > UPRIGHT_COS for n, st in state.items()}
    return relations(env.objects, positions, grasped, upright)


HOLD_RADIUS = 0.04     # object centre this close to the hand, with the fingers closed on it: held
GRIP_CLOSED = 0.05     # finger opening below this (and above ~0, i.e. blocked by something): closed on it


def perceived_predicates(objects: dict, tokens: np.ndarray, names: list[str]) -> set[tuple]:
    """The agent's own relations, from its perceived entity tokens (gripper first). Same geometry
    as the ground truth, but from perceived positions; "grasped" from the body itself (fingers
    closed on something at the object); "upright" is not perceivable from the tokens (they carry
    the yaw only) and is left out."""
    from pai.world.entities import GRIP, POS

    positions = {n: np.asarray(tokens[i, POS], float) for i, n in enumerate(names) if i > 0}
    hand, opening = np.asarray(tokens[0, POS], float), float(tokens[0, GRIP][0])
    closed = 0.005 < opening < GRIP_CLOSED
    grasped = {n for n, p in positions.items() if closed and np.linalg.norm(p - hand) < HOLD_RADIUS}
    return relations(objects, positions, grasped, upright=None)


def relations(objs: dict, positions: dict[str, np.ndarray], grasped: set, upright: dict | None) -> set[tuple]:
    """Relations from object positions (shared by ground truth and perception)."""
    preds: set[tuple] = set()
    for name, pos in positions.items():
        spec = objs[name]
        bottom = pos[2] - (spec.height / 2 if spec.kind != "bowl" else 0.0)
        if upright is not None and spec.kind == "block" and upright[name]:
            preds.add(("upright", name))
        if bottom > LIFTED_Z:
            preds.add(("lifted", name))
        if name in grasped:
            preds.add(("grasped", name))

    for a, b in permutations(positions, 2):
        sa, sb = objs[a], objs[b]
        pa, pb = positions[a], positions[b]
        dxy = float(np.linalg.norm(pa[:2] - pb[:2]))
        if dxy < NEAR_DIST:
            preds.add(("near", a, b))
        if pa[1] > pb[1] + LEFT_MARGIN:
            preds.add(("left_of", a, b))
        if sa.kind != "block":
            continue
        a_bottom = pa[2] - sa.height / 2
        if sb.kind == "bowl":
            if dxy < sb.radius - sa.radius * 0.5 and a_bottom < pb[2] + sb.height:
                preds.add(("in", a, b))
        else:
            b_top = pb[2] + sb.height / 2
            if dxy < sb.radius and -ON_Z_TOL < a_bottom - b_top < ON_Z_TOL and a not in grasped:
                # a block held just above another object is not resting on it
                preds.add(("on", a, b))
    return preds

