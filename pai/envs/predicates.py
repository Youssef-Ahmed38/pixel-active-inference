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
    """Returns the set of true relations, e.g. {("on", "red", "plate"), ("grasped", "red")}."""
    objs = env.objects
    preds: set[tuple] = set()
    grasped = env.events.grasped_objects() if hasattr(env, "events") else set()

    for name, st in state.items():
        spec, pos = objs[name], st["pos"]
        bottom = pos[2] - (spec.height / 2 if spec.kind != "bowl" else 0.0)
        if spec.kind == "block" and _up_cos(st["quat"]) > UPRIGHT_COS:
            preds.add(("upright", name))
        if bottom > LIFTED_Z:
            preds.add(("lifted", name))
        if name in grasped:
            preds.add(("grasped", name))

    for a, b in permutations(state, 2):
        sa, sb = objs[a], objs[b]
        pa, pb = state[a]["pos"], state[b]["pos"]
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

