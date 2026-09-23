"""PixelAI (Sancaktar et al., 2020): pixel-based active inference for body perception and action.

Belief over the active joints in generalised coordinates: mu (angles) and mu' (velocities).
Sensory model:  q = mu + noise (proprioception),  s_v = g(mu) + noise (vision, g = decoder).
Dynamics prior: mu' = f(mu), an attractor towards the goal.

Variational free energy (Laplace approximation, constants dropped):
    F = 1/2 pi_q |q - mu|^2  +  1/2 pi_v mean|s_v - g(mu)|^2  +  1/2 pi_mu |mu' - f(mu)|^2

Goal attractor f(mu):
    goal_mode="image":  f = -beta * d/dmu [1/2 mean|rho - g(mu)|^2]   (rho = goal image, as in PixelAI)
    goal_mode="joints": f =  beta * (q_goal - mu)                      (privileged, for debugging)

Perception (gradient descent in a moving frame; df/dmu is neglected as in PixelAI):
    mu_dot  = mu' - k_mu dF/dmu
    mu'_dot =     - k_mu dF/dmu'
Action (velocity commands). With ds/da approximated by the identity for proprioception and
by the decoder Jacobian dg/dmu for vision:
    dF/da = pi_q (q - mu) + (dg/dmu)^T pi_v (s_v - g(mu)) / P
    action_mode="integral":     a_dot = -k_a dF/da - damping * a
    action_mode="proportional": a     = -k_a dF/da
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pai.models.decoder import to_tensor_image


@dataclass
class StepInfo:
    free_energy: float
    e_q: float          # |q - mu|
    e_v: float          # mean squared visual prediction error (nan without vision)
    e_goal: float       # goal error in the goal's own space
    mu: np.ndarray
    prediction: np.ndarray | None  # g(mu) as (S, S, 3) uint8, only when requested


class PixelAIAgent:
    def __init__(self, cfg, decoder: torch.nn.Module | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.decoder = decoder.to(self.device).eval() if decoder is not None else None
        if self.decoder is not None:
            for p in self.decoder.parameters():
                p.requires_grad_(False)
        self.uses_vision = cfg.pi_v > 0 or cfg.goal_mode == "image"
        if self.uses_vision and self.decoder is None:
            raise ValueError("pi_v > 0 or goal_mode='image' needs a trained decoder")

    # ------------------------------------------------------------------ episode
    def reset(self, q_obs: np.ndarray, goal_image: np.ndarray | None = None, goal_q: np.ndarray | None = None,
              mu0: np.ndarray | None = None) -> None:
        """Start an episode. mu0 overrides the initial belief (default: first proprioceptive reading)."""
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)
        self.mu = t(q_obs if mu0 is None else mu0).clone()
        self.mu_dot = torch.zeros_like(self.mu)
        self.a = torch.zeros_like(self.mu)
        self.goal_q = t(goal_q) if goal_q is not None else None
        self.goal_img = None
        if self.cfg.goal_mode == "image":
            if goal_image is None:
                raise ValueError("goal_mode='image' needs goal_image")
            self.goal_img = to_tensor_image(goal_image, self.decoder.image_size, self.device)
        elif self.goal_q is None:
            raise ValueError("goal_mode='joints' needs goal_q")

    # ------------------------------------------------------------------ one control step
    def step(self, obs: dict, return_prediction: bool = False) -> tuple[np.ndarray, StepInfo]:
        c = self.cfg
        q = torch.as_tensor(obs["q"], dtype=torch.float32, device=self.device)
        s_v = to_tensor_image(obs["image"], self.decoder.image_size, self.device) if self.uses_vision else None

        for _ in range(int(c.n_iters)):
            grad_v, f, e_v, e_goal, pred = self._visual_terms(s_v)
            e_q = q - self.mu
            e_mu = self.mu_dot - f
            dF_dmu = grad_v - c.pi_q * e_q
            dF_dmu_dot = c.pi_mu * e_mu
            self.mu = self.mu + c.dt * (self.mu_dot - c.k_mu * dF_dmu)
            self.mu_dot = self.mu_dot + c.dt * (-c.k_mu * dF_dmu_dot)

        # (dg/dmu)^T pi_v e_v / P == -grad_v, see module docstring.
        dF_da = c.pi_q * e_q - (grad_v if c.visual_action else 0.0)
        if c.get("action_mode", "integral") == "proportional":
            self.a = -c.k_a * dF_da
        else:
            self.a = self.a + c.dt * (-c.k_a * dF_da - c.action_damping * self.a)

        free_energy = 0.5 * c.pi_q * float(e_q @ e_q) + 0.5 * c.pi_mu * float(e_mu @ e_mu)
        if s_v is not None:
            free_energy += 0.5 * c.pi_v * e_v
        info = StepInfo(
            free_energy=free_energy,
            e_q=float(e_q.norm()),
            e_v=e_v,
            e_goal=e_goal,
            mu=self.mu.cpu().numpy(),
            prediction=_to_uint8(pred) if (return_prediction and pred is not None) else None,
        )
        return self.a.cpu().numpy().astype(np.float64), info

    def _visual_terms(self, s_v):
        """Returns (dF_v/dmu, f(mu), visual mse, goal error, g(mu))."""
        c = self.cfg
        zero = torch.zeros_like(self.mu)
        if not self.uses_vision:
            e = self.goal_q - self.mu
            return zero, c.beta * e, float("nan"), float(e.norm()), None

        mu = self.mu.detach().requires_grad_(True)
        pred = self.decoder(mu[None])
        grads = []
        e_v = float("nan")
        if s_v is not None and c.pi_v > 0:
            mse = ((s_v - pred) ** 2).mean()
            grads.append(0.5 * c.pi_v * mse)
            e_v = float(mse.detach())
        if self.goal_img is not None:
            goal_mse = ((self.goal_img - pred) ** 2).mean()
            grads.append(0.5 * goal_mse)
            e_goal = float(goal_mse.detach())
        else:
            e_goal = float((self.goal_q - self.mu).norm())

        # Separate gradients for the sensory term and the goal term.
        grad_v = torch.autograd.grad(grads[0], mu, retain_graph=len(grads) > 1)[0] if c.pi_v > 0 else zero
        if self.goal_img is not None:
            grad_goal = torch.autograd.grad(grads[-1], mu)[0]
            f = -c.beta * grad_goal
        else:
            f = c.beta * (self.goal_q - self.mu)
        return grad_v.detach(), f.detach(), e_v, e_goal, pred.detach()


def _to_uint8(img: torch.Tensor) -> np.ndarray:
    return (img[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
