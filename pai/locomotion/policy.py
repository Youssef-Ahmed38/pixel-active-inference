"""Locomotion policy: a Gaussian actor-critic over proprioception + velocity goal.

The actor is the amortised controller (Stage 1's split: L0 motor skill trained by
reinforcement-style optimisation of the preference + epistemic objective, upper levels stay
active inference). ``LocomotionController`` is the frozen inference wrapper other stages reuse:
``act(obs) -> leg servo action`` with the command inside ``obs[9:12]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from pai.locomotion.env import ACT_DIM, CMD, OBS_DIM


class RunningNorm(nn.Module):
    """Running mean/variance input normaliser.

    The goal command (``CMD``) is passed through unnormalised: during the stand stage it is
    always zero, so its running variance would be zero and the first non-zero command would hit
    the clip. The sample count is capped so the statistics follow the curriculum's changing
    state distribution (a memory of ``max_count`` samples).
    """

    def __init__(self, dim: int, clip: float = 10.0, max_count: float = 5e6):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip
        self.max_count = max_count
        self.raw = list(range(CMD.start, CMD.stop)) if dim == OBS_DIM else []

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot.clamp(max=self.max_count)
        if self.raw:
            self.mean[self.raw] = 0.0
            self.var[self.raw] = 1.0

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(din, hidden, dout, out_scale=1.0):
    layers, d = [], din
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    last = nn.Linear(d, dout)
    if out_scale < 1:
        nn.init.uniform_(last.weight, -out_scale, out_scale)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden=(512, 256, 128), init_std: float = 0.8):
        super().__init__()
        self.obs_norm = RunningNorm(obs_dim)
        self.actor = mlp(obs_dim, hidden, act_dim, out_scale=0.01)
        self.critic = mlp(obs_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))
        self.hidden = tuple(hidden)

    def dist(self, obs_n):
        mu = self.actor(obs_n)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs_n):
        return self.critic(obs_n).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        on = self.obs_norm(obs)
        d = self.dist(on)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(on)


def save_policy(path: Path, ac: ActorCritic, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ac.state_dict(), "hidden": ac.hidden, **(extra or {})}, path)


def load_policy(path: Path, device="cpu") -> tuple[ActorCritic, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck.get("state_dict") or ck["policy"]
    ac = ActorCritic(hidden=ck.get("hidden", (512, 256, 128)))
    ac.load_state_dict(sd)
    ac.to(device).eval()
    return ac, ck


class LocomotionController:
    """Frozen learned controller: observation (with the goal command) -> leg action."""

    def __init__(self, path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.ac, self.meta = load_policy(path, self.device)

    def act(self, obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(np.asarray(obs, np.float32), device=self.device)
        a, _, _ = self.ac.act(o, deterministic=True)
        return a.cpu().numpy()
