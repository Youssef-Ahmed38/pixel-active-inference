"""Learned L0 body model: an ensemble that predicts the next proprioceptive state.

``f_k(s_t, a_t) -> s_{t+1} - s_t`` for k = 1..K, trained on the body's own experience (motor
babbling first, then practice). Three uses, all measured in docs/LOCOMOTION_STAGE2.md:

* surprise   = squared prediction error of the ensemble mean in normalised units
               (a Gaussian negative log-likelihood up to a constant): proprioceptive surprise;
* epistemic  = ensemble disagreement (variance of the member means): the curiosity term of the
               expected free energy, used as an exploration bonus;
* evidence   = surprise spikes are the evidence used by fall/push cause inference (falls.py).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pai.locomotion.env import ACT_DIM, PROPRIO

S_DIM = len(PROPRIO)


class Normalizer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, x: torch.Tensor):
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(1e-3))

    def forward(self, x):
        return (x - self.mean) / self.std


class EnsembleBodyModel(nn.Module):
    def __init__(self, members: int = 5, hidden: int = 256, s_dim: int = S_DIM, a_dim: int = ACT_DIM):
        super().__init__()
        self.members, self.s_dim, self.a_dim = members, s_dim, a_dim
        din = s_dim + a_dim
        # batched ensemble: weights (K, in, out)
        def p(*shape, scale):
            return nn.Parameter(torch.randn(*shape) * scale)
        self.w1 = p(members, din, hidden, scale=din ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(members, 1, hidden))
        self.w2 = p(members, hidden, hidden, scale=hidden ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(members, 1, hidden))
        self.w3 = p(members, hidden, s_dim, scale=0.1 * hidden ** -0.5)
        self.b3 = nn.Parameter(torch.zeros(members, 1, s_dim))
        self.in_norm = Normalizer(din)
        self.out_norm = Normalizer(s_dim)
        self.fitted = False

    def fit_normalizers(self, s, a, s_next):
        self.in_norm.fit(torch.cat([s, a], -1))
        self.out_norm.fit(s_next - s)
        self.fitted = True

    def forward(self, s, a):
        """Normalised predicted deltas for every member: (K, N, s_dim)."""
        x = self.in_norm(torch.cat([s, a], -1))
        x = x.unsqueeze(0).expand(self.members, -1, -1)
        h = torch.nn.functional.silu(torch.baddbmm(self.b1, x, self.w1))
        h = torch.nn.functional.silu(torch.baddbmm(self.b2, h, self.w2))
        return torch.baddbmm(self.b3, h, self.w3)

    def loss(self, s, a, s_next, member_mask=None):
        target = self.out_norm(s_next - s)
        pred = self(s, a)
        err = (pred - target.unsqueeze(0)) ** 2
        if member_mask is not None:            # bootstrap: each member sees its own subset
            err = err * member_mask.unsqueeze(-1)
            return err.sum() / (member_mask.sum() * self.s_dim).clamp_min(1)
        return err.mean()

    @torch.no_grad()
    def surprise_and_disagreement(self, s, a, s_next=None):
        pred = self(s, a)
        mean = pred.mean(0)
        dis = pred.var(0).mean(-1)
        if s_next is None:
            return None, dis
        target = self.out_norm(s_next - s)
        sur = ((mean - target) ** 2).mean(-1)
        return sur, dis

    @torch.no_grad()
    def predict(self, s, a):
        """Mean next state in physical units."""
        d = self(s, a).mean(0) * self.out_norm.std + self.out_norm.mean
        return s + d


class ReplayBuffer:
    def __init__(self, capacity: int, s_dim: int = S_DIM, a_dim: int = ACT_DIM):
        self.s = np.zeros((capacity, s_dim), np.float32)
        self.a = np.zeros((capacity, a_dim), np.float32)
        self.s2 = np.zeros((capacity, s_dim), np.float32)
        self.cap, self.n, self.i = capacity, 0, 0

    def add(self, s, a, s2):
        k = len(s)
        if k == 0:
            return
        idx = (self.i + np.arange(k)) % self.cap
        self.s[idx], self.a[idx], self.s2[idx] = s, a, s2
        self.i = (self.i + k) % self.cap
        self.n = min(self.cap, self.n + k)

    def sample(self, k, rng):
        idx = rng.integers(0, self.n, k)
        return self.s[idx], self.a[idx], self.s2[idx]


def train_steps(model: EnsembleBodyModel, opt, buf: ReplayBuffer, steps: int, batch: int,
                rng: np.random.Generator, device) -> float:
    losses = []
    for _ in range(steps):
        s, a, s2 = (torch.as_tensor(x, device=device) for x in buf.sample(batch, rng))
        mask = torch.as_tensor(rng.random((model.members, batch)) < 0.8, device=device, dtype=torch.float32)
        loss = model.loss(s, a, s2, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def heldout_error(model: EnsembleBodyModel, s, a, s2, device, chunk: int = 65536) -> float:
    errs = []
    for i in range(0, len(s), chunk):
        t = [torch.as_tensor(x[i:i + chunk], device=device) for x in (s, a, s2)]
        sur, _ = model.surprise_and_disagreement(*t)
        errs.append(sur.cpu().numpy())
    return float(np.mean(np.concatenate(errs)))
