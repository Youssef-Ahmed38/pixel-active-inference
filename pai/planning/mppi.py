"""MPPI planning in the learned world model, scored by expected free energy.

For a candidate action sequence, the imagined rollout is scored by

    G = sum_t [ cost_t(x_t)  -  lambda_epi * epistemic_t ]  +  action effort  +  smoothness

- cost_t = -log C(x_t): how far the imagined state is from the preferred one (pragmatic value)
- epistemic_t: ensemble disagreement about the predicted change, a standard tractable proxy for
  the expected information gain about the model (the epistemic value)

Smoothness penalises changes between consecutive actions (including from the last executed one):
jerky plans leave the world model's training distribution, where its predictions and its own
uncertainty estimates stop being trustworthy. Lower G is better. MPPI samples perturbations around the previous plan, weights them by
exp(-G / temperature) and averages. Only the first action is executed (receding horizon).
"""

from __future__ import annotations

import torch


class MPPIPlanner:
    def __init__(self, world_model, horizon: int = 10, samples: int = 256, iterations: int = 2,
                 temperature: float = 0.1, noise: tuple = (0.08, 0.08, 0.08, 0.3), epistemic_weight: float = 0.0,
                 effort_weight: float = 0.01, smooth_weight: float = 2.0, action_low=(-0.25, -0.25, -0.25, -1.0), action_high=(0.25, 0.25, 0.25, 1.0),
                 device="cpu"):
        self.wm = world_model
        self.H, self.K, self.iters = horizon, samples, iterations
        self.temperature = temperature
        self.epi_w, self.effort_w, self.smooth_w = epistemic_weight, effort_weight, smooth_weight
        self.device = torch.device(device)
        self.noise = torch.tensor(noise, device=self.device)
        self.low = torch.tensor(action_low, device=self.device)
        self.high = torch.tensor(action_high, device=self.device)
        self.plan = torch.zeros(horizon, len(noise), device=self.device)
        self.last_action = torch.zeros(len(noise), device=self.device)

    def reset(self) -> None:
        self.plan.zero_()
        self.last_action.zero_()

    @torch.no_grad()
    def rollout(self, tokens: torch.Tensor, actions: torch.Tensor, cost_fn):
        """tokens (N, D); actions (K, H, A) -> total cost (K,), pragmatic (K,), epistemic (K,)."""
        x = tokens.expand(actions.shape[0], *tokens.shape).clone()
        pragmatic = torch.zeros(actions.shape[0], device=self.device)
        epistemic = torch.zeros_like(pragmatic)
        # Without the epistemic term, one randomly chosen member per rollout is enough (5x cheaper,
        # and sampling a member is Thompson-style exploration); with it, all members are needed.
        member = None if self.epi_w > 0 else int(torch.randint(len(self.wm.members), (1,)))
        for t in range(actions.shape[1]):
            pred = self.wm.predict(x, actions[:, t], member=member)
            x, epi = pred.next_tokens, pred.epistemic
            w = 1.0 if t < actions.shape[1] - 1 else 3.0  # the end of the horizon matters most
            pragmatic = pragmatic + w * cost_fn(x)
            epistemic = epistemic + epi[..., :3].sum((-1, -2))
        effort = (actions[..., :3] ** 2).sum((-1, -2))
        prev = torch.cat([self.last_action.expand(actions.shape[0], 1, -1), actions[:, :-1]], 1)
        jerk = ((actions - prev) ** 2).sum((-1, -2))
        total = pragmatic - self.epi_w * epistemic + self.effort_w * effort + self.smooth_w * jerk
        return total, pragmatic, epistemic

    @torch.no_grad()
    def act(self, tokens: torch.Tensor, cost_fn, grip: float | None = None) -> tuple[torch.Tensor, dict]:
        """grip: gripper mode fixed by the goal level (L2); then only the motion is planned."""
        self.plan = torch.cat([self.plan[1:], self.plan[-1:]])  # warm start: shift the previous plan
        if grip is not None:
            self.plan[:, 3] = grip
        for _ in range(self.iters):
            eps = torch.randn(self.K, self.H, self.plan.shape[1], device=self.device) * self.noise
            eps[0] = 0  # always evaluate the current plan itself
            cand = torch.maximum(torch.minimum(self.plan + eps, self.high), self.low)
            if grip is not None:
                cand[..., 3] = grip
            total, prag, epi = self.rollout(tokens, cand, cost_fn)
            # Temperature relative to the spread of this batch's costs, so the planner behaves the
            # same whatever the absolute scale of the goal's cost function.
            beta = self.temperature * (total.std() + 1e-9)
            w = torch.softmax(-(total - total.min()) / beta, dim=0)
            self.plan = (w[:, None, None] * cand).sum(0)
            if grip is not None:
                self.plan[:, 3] = grip  # exact, not a float-rounded average
        best = int(torch.argmin(total))
        info = {"G": float(total[best]), "pragmatic": float(prag[best]), "epistemic": float(epi[best])}
        self.last_action = self.plan[0].clone()
        return self.plan[0].clone(), info
