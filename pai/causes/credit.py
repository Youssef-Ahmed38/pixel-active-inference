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

Two versions:

    assign_credit        replays the WHOLE episode from its first state and judges the final goal.
                         Exact in a toy world, but in the learned model a ~70-step open-loop replay
                         drifts so far that even the unmodified replay misses the goal.
    assign_phase_credit  local counterfactuals: imagination starts at the RECORDED state at the start
                         of the phase an intervention touches, replays only that phase (plus the
                         phases up to the one that judges it) and a short hold-still tail, and judges
                         with that phase's own subgoal (cost and done test, pai.goals.relations).
                         Each factor is judged by the step it was meant to serve: skipping the lift
                         by "lifted", an off-centre approach by "grasped". Slowing down is tested as
                         half speed for twice as long (same path): did the speed matter, not the
                         distance.

Even one phase drifts in the learned model (the 17-37 step approach misses the 8 mm "at_object"
test by 1-4 cm), so the local replay is a counterfactual in Pearl's sense by default: abduction
infers the model's residual at every recorded step (what it did not predict), and the replay adds
it back at the same step. The unmodified replay then reproduces the episode exactly, and only the
change the intervention makes comes from the model.

A factor is necessary if the unmodified local replay passes the phase's done test and the intervened
one fails it; if the unmodified replay already fails (possible without abduction), the model cannot
judge that phase and the test is marked invalid rather than guessed. The residuals carry whatever
the model does not explain, so an effect the model does not predict cannot be credited: this model
hardly changes the predicted finger opening with the gripper command, so closing the fingers during
"grasped" comes out as not mattering (the recorded closing is in the residuals), while opening them
during the carry does break it (the model lets the object fall behind the hand).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from pai.world.entities import DYNAMIC


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
    phase: str = ""          # the subgoal that judged it (local counterfactuals); "" = the final goal
    valid: bool = True       # False if the unmodified replay already failed the judge's done test

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


@dataclass
class LocalIntervention:
    """An intervention on the actions of a window of phases, judged by the last phase's subgoal."""
    intervention: Intervention
    start: str   # the phase whose recorded start state imagination begins from
    judge: str   # the phase whose subgoal (cost, done) judges the end state
    stretch: int = 1  # every action is repeated this many times (slow_down: half speed, twice as long)


def phase_interventions(phases: dict[str, tuple[int, int]]) -> list[LocalIntervention]:
    """One local test per factor; phases: subgoal name -> (first, last) step, last exclusive."""
    out = []
    for name, (a, b) in phases.items():
        def skip(acts, a=a, b=b):
            acts = acts.copy()
            acts[a:b, :3] = 0.0
            if a > 0:
                acts[a:b, 3] = acts[a - 1, 3]  # "do nothing" includes the gripper: keep its last command
            return acts
        out.append(LocalIntervention(Intervention(f"skip_{name}", f"do nothing during '{name}'", skip),
                                     name, name))
    if "at_object" in phases:
        a, b = phases["at_object"]

        def shift(acts, a=a, b=b):
            acts = acts.copy()
            acts[a:b, 1] += 0.1  # 10 cm/s sideways during the final approach
            return acts
        judge = "grasped" if "grasped" in phases else "at_object"
        out.append(LocalIntervention(Intervention("shift_grasp", "approach the object off-centre", shift),
                                     "at_object", judge))
    carry = [k for k in ("lifted", "over_target") if k in phases]
    if carry:
        a, b = phases[carry[0]][0], phases[carry[-1]][1]

        def open_early(acts, a=a, b=b):
            acts = acts.copy()
            acts[a:b, 3] = 1.0
            return acts
        out.append(LocalIntervention(Intervention("open_early", "open the gripper while carrying", open_early),
                                     carry[0], carry[-1]))
    motion = [k for k in ("lifted", "over_target", "lowered") if k in phases]
    if motion:
        def half(acts):
            acts = acts.copy()
            acts[:, :3] *= 0.5
            return acts
        out.append(LocalIntervention(Intervention("slow_down", "carry at half speed (twice as long)", half),
                                     motion[0], motion[-1], stretch=2))
    return out


def _window(acts: np.ndarray, first: int, last: int, stretch: int, tail: int) -> np.ndarray:
    """actions[first:last], each repeated `stretch` times, then `tail` hold-still steps (same grip)."""
    w = np.repeat(acts[first:last], stretch, axis=0)
    hold = np.zeros((tail, acts.shape[1]), acts.dtype)
    if len(w):
        hold[:, 3] = w[-1, 3]
    return np.concatenate([w, hold])


@torch.no_grad()
def residuals(wm, tokens: np.ndarray, actions: np.ndarray, device="cpu") -> torch.Tensor:
    """Abduction: what the model did not explain at each recorded step, u_t = x_{t+1} - f(x_t, a_t).

    tokens (L+1, N, D), actions (L, A) -> (L, N, D). Only the dynamic features change in a
    prediction, so only they carry a residual."""
    x = torch.as_tensor(tokens, device=device)
    pred = wm.predict(x[:-1], torch.as_tensor(actions, device=device)).next_tokens
    u = torch.zeros_like(pred)
    u[..., DYNAMIC] = (x[1:] - pred)[..., DYNAMIC]
    return u


@torch.no_grad()
def imagine_path(wm, start_tokens: np.ndarray, actions: np.ndarray, noise: torch.Tensor | None = None,
                 device="cpu") -> torch.Tensor:
    """Rollout of the ensemble mean that keeps every state, (L+1, N, D). noise (L, N, D), if given,
    is added after each step: a counterfactual that reuses the factual episode's residuals."""
    x = torch.as_tensor(start_tokens, device=device)[None]
    path = [x[0]]
    for t, a in enumerate(torch.as_tensor(actions, device=device)):
        x = wm.predict(x, a[None]).next_tokens
        if noise is not None:
            x = x + noise[t]
        path.append(x[0])
    return torch.stack(path)


def assign_phase_credit(wm, tokens: np.ndarray, actions: np.ndarray, phases: dict, subgoals: list,
                        tail: int = 3, abduction: bool = True, device="cpu") -> list[CreditResult]:
    """Per-phase local counterfactuals. tokens (T+1, N, D), actions (T, A) of a successful episode;
    phases {subgoal: (first, last)}; subgoals: the goal's Subgoal list (name, cost, done).

    Each test replays the actions of its window of phases (changed or not) from the recorded state
    at the window's start, then holds still for `tail` steps. The judging subgoal counts as reached
    if its done test holds at the end of the window or during the tail (a grace period for motions
    that settle late); its cost is the lowest over the same steps.

    abduction: counterfactuals in Pearl's sense. The residual of every recorded step (what the model
    did not predict) is inferred from the factual episode and added back at the same step of the
    replay, so the unmodified replay reproduces what happened exactly and only the *change* the
    intervention makes is imagined. Without it, the plain open-loop replay drifts, and phases whose
    unmodified replay already fails are marked invalid.

    Returns one CreditResult per test: valid before invalid, necessary before not, then by
    importance (the rise of the judging subgoal's cost).
    """
    by_name = {g.name: g for g in subgoals}
    baselines: dict[tuple[str, str], tuple[float, bool]] = {}
    results = []
    for li in phase_interventions(phases):
        if li.judge not in by_name:
            continue
        first, last = phases[li.start][0], phases[li.judge][1]
        judge = by_name[li.judge]
        u = residuals(wm, tokens[first:last + 1], actions[first:last], device) if abduction else None

        def outcome(acts, stretch=1):
            window = _window(acts, first, last, stretch, tail)
            noise = None
            if u is not None:  # a stretched step does half of the recorded step, with half its residual
                noise = torch.zeros((len(window), *u.shape[1:]), dtype=u.dtype, device=u.device)
                noise[:len(u) * stretch] = u.repeat_interleave(stretch, 0) / stretch
            path = imagine_path(wm, tokens[first], window, noise, device)[-(tail + 1):]
            return min(float(judge.cost(x)) for x in path), any(bool(judge.done(x)) for x in path)

        key = (li.start, li.judge)
        if key not in baselines:
            baselines[key] = outcome(actions)
        base, base_done = baselines[key]
        c, done = outcome(li.intervention.apply(actions), li.stretch)
        iv = li.intervention
        results.append(CreditResult(iv.name, iv.description, c, base, necessary=base_done and not done,
                                    phase=li.judge, valid=base_done))
    return sorted(results, key=lambda r: (not r.valid, not r.necessary, -r.importance))
