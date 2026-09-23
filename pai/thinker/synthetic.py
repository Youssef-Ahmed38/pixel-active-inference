"""Synthetic hidden-cause inference with an exact Bayesian teacher.

Generative model (one example):
    cause k ~ Uniform(K)
    nuisance g = (onset tau, amplitude a, affected object j) ~ Uniform(grid)
    noise sigma ~ LogUniform(noise_range), evidence length L ~ Uniform{min_length..horizon}
    x[t, c] = a * sum_i gain_i * shape_i(t - tau) [channel_i] + sigma * eps,   observed for t < L

Channels are prediction-error signals: vision, proprioception, force, and one per object.
Each cause has a signature (which channels move, with what temporal shape). Two pairs are
deliberately similar (occlusion ~ camera_shift, heavier_object ~ low_friction); `overlap`
in [0, 1] pulls each pair's templates towards their mean (1 = indistinguishable). Short evidence
(L before the onset) is ambiguous with "none"; large sigma makes everything ambiguous.

Teacher: with a uniform prior over causes and nuisances, the log-likelihood of each
template is (x.m - |m|^2/2) / sigma^2 up to terms shared by all templates, so
    p(k | x) propto sum_g exp((x.m_kg - |m_kg|^2 / 2) / sigma^2),
exact because the nuisances are drawn from the same finite grid the teacher sums over.

A held-out cause (sensor_drift) can be mixed in with `heldout_prob`; its label is K and the
teacher has no hypothesis for it (for later "unknown cause" experiments).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

CAUSES = ("none", "occlusion", "push", "camera_shift", "heavier_object", "low_friction")
HELDOUT_CAUSE = "sensor_drift"
CHANNELS = ("vision", "proprio", "force", "obj_a", "obj_b")

# cause -> [(channel role, temporal shape, gain)]; "obj" = the affected object, "other" = the other one.
SIGNATURES: dict[str, list[tuple[str, str, float]]] = {
    "none": [],
    "occlusion": [("vision", "step", 0.5), ("obj", "step", 1.0)],
    "push": [("proprio", "pulse", 1.2), ("force", "pulse", 1.0), ("vision", "pulse", 0.6)],
    "camera_shift": [("vision", "step", 1.0), ("obj", "step", 0.7), ("other", "step", 0.7)],
    "heavier_object": [("force", "step", 1.0), ("proprio", "ramp", 0.6), ("obj", "ramp", 0.3)],
    "low_friction": [("force", "step", 0.5), ("proprio", "ramp", 0.2), ("obj", "ramp", 1.0)],
    HELDOUT_CAUSE: [("proprio", "slow", 0.8), ("vision", "slow", 0.4)],
}
SIMILAR_PAIRS = (("occlusion", "camera_shift"), ("heavier_object", "low_friction"))


def _shape(kind: str, t: np.ndarray, tau: int, horizon: int) -> np.ndarray:
    dt = t - tau
    if kind == "step":
        return (dt >= 0).astype(np.float64)
    if kind == "pulse":
        return np.where(dt >= 0, np.exp(-np.maximum(dt, 0) / 3.0), 0.0)
    if kind == "ramp":
        return np.clip((dt + 1) / 8.0, 0.0, 1.0)
    if kind == "slow":
        return np.clip((dt + 1) / horizon, 0.0, 1.0)
    raise ValueError(kind)


def _template(cause: str, tau: int, amp: float, obj: int, horizon: int) -> np.ndarray:
    t = np.arange(horizon)
    m = np.zeros((horizon, len(CHANNELS)))
    for role, kind, gain in SIGNATURES[cause]:
        ch = 3 + obj if role == "obj" else 4 - obj if role == "other" else CHANNELS.index(role)
        m[:, ch] += amp * gain * _shape(kind, t, tau, horizon)
    return m


@dataclass
class TaskConfig:
    horizon: int = 32
    min_length: int = 8
    onset_range: tuple[int, int] = (2, 20)  # inclusive
    amplitudes: tuple[float, ...] = (0.6, 1.0, 1.5)
    noise_range: tuple[float, float] = (0.25, 2.0)
    overlap: float = 0.5
    heldout_prob: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "TaskConfig":
        d = dict(d)
        for k in ("onset_range", "amplitudes", "noise_range"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


class CauseTask:
    """Sampler + exact teacher. Everything lives on `device`; sampling is batched."""

    n_channels = len(CHANNELS)
    ctx_dim = 2  # (log sigma, L / horizon)

    def __init__(self, cfg: TaskConfig, device: torch.device | str = "cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        grid = [(tau, a, j) for tau in range(cfg.onset_range[0], cfg.onset_range[1] + 1)
                for a in cfg.amplitudes for j in (0, 1)]
        bank = np.stack([[_template(c, *g, cfg.horizon) for g in grid] for c in CAUSES])  # [K, G, T, C]
        for a, b in SIMILAR_PAIRS:
            ia, ib = CAUSES.index(a), CAUSES.index(b)
            ma, mb = bank[ia].copy(), bank[ib].copy()
            bank[ia] = ma + 0.5 * cfg.overlap * (mb - ma)
            bank[ib] = mb + 0.5 * cfg.overlap * (ma - mb)
        heldout = np.stack([_template(HELDOUT_CAUSE, *g, cfg.horizon) for g in grid])
        self.grid = grid
        self.bank = torch.tensor(bank, dtype=torch.float32, device=self.device)
        self.heldout_bank = torch.tensor(heldout, dtype=torch.float32, device=self.device)
        k, g, t, c = self.bank.shape
        self._flat = self.bank.reshape(k * g, t * c)
        self._energy = self.bank.pow(2).sum(-1).cumsum(-1).reshape(k * g, t)  # |m|^2 over the first t+1 steps

    @property
    def n_causes(self) -> int:
        return len(CAUSES)

    def sample(self, n: int, gen: torch.Generator | None = None, heldout_prob: float | None = None) -> dict:
        cfg, dev = self.cfg, self.device
        k_all, g_all, t, c = self.bank.shape
        y = torch.randint(0, k_all, (n,), device=dev, generator=gen)
        g = torch.randint(0, g_all, (n,), device=dev, generator=gen)
        mean = self.bank[y, g]
        p_new = cfg.heldout_prob if heldout_prob is None else heldout_prob
        if p_new > 0:
            new = torch.rand(n, device=dev, generator=gen) < p_new
            mean = torch.where(new[:, None, None], self.heldout_bank[g], mean)
            y = torch.where(new, torch.full_like(y, k_all), y)
        lo, hi = map(math.log, cfg.noise_range)
        sigma = torch.exp(lo + (hi - lo) * torch.rand(n, device=dev, generator=gen))
        length = torch.randint(cfg.min_length, cfg.horizon + 1, (n,), device=dev, generator=gen)
        mask = (torch.arange(t, device=dev)[None] < length[:, None]).float()
        x = (mean + sigma[:, None, None] * torch.randn(n, t, c, device=dev, generator=gen)) * mask[..., None]
        ctx = torch.stack([sigma.log(), length.float() / cfg.horizon], -1)
        return {"x": x, "mask": mask, "ctx": ctx, "y": y, "sigma": sigma, "length": length,
                "teacher": self.posterior(x, length, sigma)}

    @torch.no_grad()
    def log_posterior(self, x: torch.Tensor, length: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Exact log p(cause | x) [B, K]; x must be zero beyond each example's length."""
        k, g = self.bank.shape[:2]
        dot = x.reshape(x.shape[0], -1).float() @ self._flat.T  # [B, K*G]
        energy = self._energy.T[length - 1]  # [B, K*G]
        ll = ((dot - 0.5 * energy) / sigma[:, None].pow(2)).view(-1, k, g)
        return torch.log_softmax(torch.logsumexp(ll, -1), -1)  # uniform priors cancel

    def posterior(self, x, length, sigma) -> torch.Tensor:
        return self.log_posterior(x, length, sigma).exp()
