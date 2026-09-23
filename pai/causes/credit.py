"""Explaining the expected: why did a success happen? (Phase 3c, first version)

Counterfactual credit assignment in imagination. A stored successful episode (entity tokens and
actions) is replayed through the world model from a chosen step, once as it happened and once per
*intervention*, each changing one factor:

    skip_segment   the actions of one phase are replaced by "do nothing"
    shift_grasp    the approach is offset sideways before the grasp
    open_early     the gripper opens during the carry
    slow_down      the motion is halved

If the imagined outcome still satisfies the goal, the factor did not matter. If it no longer does,
the factor is a *cause of the success*, and its importance is how much the goal cost rises. The
necessary factors become the "why" of a recipe. The world model is the agent's own, so these are
the agent's beliefs about causation, which the simulator can later check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class Intervention:
    name: str
    description: str
    apply: Callable[[np.ndarray], np.ndarray]  # actions (T, A) -> changed actions


@dataclass
class CreditResult:
    name: str
    description: str
    goal_cost: float
    baseline_cost: float
    necessary: bool

    @property
    def importance(self) -> float:
        return self.goal_cost - self.baseline_cost


def default_interventions(phases: dict[str, tuple[int, int]]) -> list[Intervention]:
    """phases: subgoal name -> (first step, last step) of the actions executed for it."""
    out = []
    for name, (a, b) in phases.items():
        def skip(acts, a=a, b=b):
            acts = acts.copy()
            acts[a:b, :3] = 0.0
            return acts
        out.append(Intervention(f"skip_{name}", f"do nothing during '{name}'", skip))
    if "at_object" in phases:
        a, b = phases["at_object"]

        def shift(acts, a=a, b=b):
            acts = acts.copy()
            acts[a:b, 1] += 0.1  # 10 cm/s sideways during the final approach
            return acts
        out.append(Intervention("shift_grasp", "approach the object off-centre", shift))
    carry = [phases[k] for k in ("lifted", "over_target") if k in phases]
    if carry:
        a, b = carry[0][0], carry[-1][1]

        def open_early(acts, a=a, b=b):
            acts = acts.copy()
            acts[a:b, 3] = 1.0
            return acts
        out.append(Intervention("open_early", "open the gripper while carrying", open_early))

    def slow(acts):
        acts = acts.copy()
        acts[:, :3] *= 0.5
        return acts
    out.append(Intervention("slow_down", "move at half speed", slow))
    return out


@torch.no_grad()
def imagine(wm, start_tokens: np.ndarray, actions: np.ndarray, device="cpu") -> torch.Tensor:
    """Open-loop rollout of the ensemble mean; returns the final imagined tokens (N, D)."""
    x = torch.as_tensor(start_tokens, device=device)[None]
    for a in torch.as_tensor(actions, device=device):
        x = wm.predict(x, a[None]).next_tokens
    return x[0]


def assign_credit(wm, tokens: np.ndarray, actions: np.ndarray, goal_cost: Callable, phases: dict,
                  success_threshold: float, device="cpu") -> tuple[float, list[CreditResult]]:
    """tokens (T+1, N, D), actions (T, A) of a successful episode.

    goal_cost(final_tokens) -> scalar cost of the final goal (e.g. the 'released' subgoal cost).
    Returns the imagined baseline cost and one CreditResult per intervention, most important first.
    """
    base = float(goal_cost(imagine(wm, tokens[0], actions, device)))
    results = []
    for iv in default_interventions(phases):
        c = float(goal_cost(imagine(wm, tokens[0], iv.apply(actions), device)))
        results.append(CreditResult(iv.name, iv.description, c, base, necessary=c > success_threshold >= base))
    return base, sorted(results, key=lambda r: -r.importance)
