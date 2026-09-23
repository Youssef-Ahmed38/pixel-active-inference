"""Synthetic multi-cause inference: 0-3 simultaneous causes from a library of 6, with an exact
Bayesian teacher over cause *subsets*. This is the "explaining away" task: two active causes can
each account for part of the same channel's error (occlusion and camera_shift both move vision;
heavier_object and low_friction both move force/proprio), so the correct posterior on one cause
depends on what the others already explain. A single feed-forward matched filter (which is exactly
what solved the single-cause task) is no longer enough on its own: crediting cause A less because
cause B already explains an overlapping channel is a *relative* judgement across hypotheses.

Reuses the single-cause signatures and shapes from `synthetic.py` (SIGNATURES, _template,
CHANNELS): the library is the 5 non-"none" causes there plus the held-out `sensor_drift`, now an
ordinary member, for 6 total.

Generative model (one example):
    size ~ Uniform{0, .., max_active}
    subset ~ Uniform over the C(6, size) subsets of that size (cause_active in {0,1}^6)
    obj ~ Uniform{0,1}, amplitude ~ Uniform(amplitude_grid)          (shared nuisances)
    tau_c ~ Uniform(onset_grid), independently for every active cause c
    x[t, ch] = sum_{c active} amplitude * template_c(t - tau_c)[ch] + sigma * eps, for t < L

Teacher: exact, because size, subset, obj, amplitude and every onset are drawn from the same
finite grids the teacher enumerates and marginalises over. Subsets larger than `max_active` have
zero prior mass under this generative model, so the teacher never builds their templates; their
posterior is exactly 0 by construction, and the returned subset-posterior vector (length 64, one
entry per subset of the 6-cause library) is zero there. Per-cause marginals are the sum of
subset-posterior over the subsets that contain that cause.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from pai.thinker.synthetic import CHANNELS, SIGNATURES, SIMILAR_PAIRS, _shape, _template

CAUSE_LIB = ("occlusion", "push", "camera_shift", "heavier_object", "low_friction", "sensor_drift")
N_LIB = len(CAUSE_LIB)
N_SUBSETS = 2 ** N_LIB  # 64
_PAIR_PARTNER: dict[str, str] = {}
for _a, _b in SIMILAR_PAIRS:
    if _a in CAUSE_LIB and _b in CAUSE_LIB:
        _PAIR_PARTNER[_a], _PAIR_PARTNER[_b] = _b, _a


def _shape_t(kind: str, dt: torch.Tensor, horizon: int) -> torch.Tensor:
    """Torch version of synthetic._shape, batched over dt of any shape (same formulas)."""
    if kind == "step":
        return (dt >= 0).to(dt.dtype)
    if kind == "pulse":
        return torch.where(dt >= 0, torch.exp(-dt.clamp_min(0) / 3.0), torch.zeros_like(dt))
    if kind == "ramp":
        return ((dt + 1) / 8.0).clamp(0.0, 1.0)
    if kind == "slow":
        return ((dt + 1) / horizon).clamp(0.0, 1.0)
    raise ValueError(kind)


def _cause_signal_batch(cause: str, tau: torch.Tensor, amp: torch.Tensor, obj: torch.Tensor,
                         horizon: int, n_channels: int) -> torch.Tensor:
    """[n] tau/amp/obj -> [n, horizon, n_channels] contribution of one (batched) cause instance."""
    n, dev, dt_type = tau.shape[0], tau.device, tau.dtype
    t = torch.arange(horizon, device=dev, dtype=dt_type)[None, :]
    dt = t - tau[:, None]
    out = torch.zeros(n, horizon, n_channels, device=dev, dtype=amp.dtype)
    for role, kind, gain in SIGNATURES[cause]:
        shape = _shape_t(kind, dt, horizon).to(amp.dtype)
        if role == "obj":
            ch = 3 + obj
        elif role == "other":
            ch = 4 - obj
        else:
            ch = torch.full_like(obj, CHANNELS.index(role))
        val = (amp * gain)[:, None] * shape  # [n, T]
        out.scatter_add_(2, ch[:, None, None].expand(-1, horizon, 1), val.unsqueeze(-1))
    return out


@dataclass
class MultiCauseConfig:
    horizon: int = 64
    min_length: int = 8
    max_active: int = 3           # 0..3 simultaneous causes
    onset_grid: tuple[int, ...] = (4, 20, 36, 52)   # per-cause nuisance, independent draw
    amplitude_grid: tuple[float, ...] = (0.7, 1.0, 1.3)  # shared nuisance
    noise_range: tuple[float, float] = (0.3, 2.0)
    overlap: float = 0.6

    @classmethod
    def from_dict(cls, d: dict) -> "MultiCauseConfig":
        d = dict(d)
        for k in ("onset_grid", "amplitude_grid", "noise_range"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


class MultiCauseTask:
    """Sampler + exact subset teacher for the multi-cause task. Batched, lives on `device`."""

    n_channels = len(CHANNELS)
    ctx_dim = 2  # (log sigma, L / horizon)
    n_causes = N_LIB
    n_subsets = N_SUBSETS

    def __init__(self, cfg: MultiCauseConfig, device: torch.device | str = "cpu"):
        self.cfg, self.device = cfg, torch.device(device)
        self._build_bank()

    # ---- exact teacher bank (built once) -------------------------------------------------
    def _build_bank(self) -> None:
        cfg, H, C = self.cfg, self.cfg.horizon, self.n_channels
        onset, amp_grid = cfg.onset_grid, cfg.amplitude_grid
        templates, masks = [], []
        for mask in range(N_SUBSETS):
            active = [i for i in range(N_LIB) if mask & (1 << i)]
            if len(active) > cfg.max_active:
                continue
            for obj in (0, 1):
                for amp in amp_grid:
                    for taus in itertools.product(onset, repeat=len(active)):
                        m = np.zeros((H, C))
                        for ci, tau in zip(active, taus):
                            cause = CAUSE_LIB[ci]
                            t1 = _template(cause, tau, amp, obj, H)
                            partner = _PAIR_PARTNER.get(cause)
                            if partner:
                                t2 = _template(partner, tau, amp, obj, H)
                                t1 = t1 + 0.5 * cfg.overlap * (t2 - t1)
                            m += t1
                        templates.append(m)
                        masks.append(mask)
        bank = np.stack(templates)  # [Ncombo, H, C]
        dev = self.device
        self.combo_bank = torch.tensor(bank, dtype=torch.float32, device=dev)
        self.combo_mask = torch.tensor(masks, dtype=torch.long, device=dev)
        self._combo_flat = self.combo_bank.reshape(len(masks), -1)
        self._combo_energy = self.combo_bank.pow(2).sum(-1).cumsum(-1)  # [Ncombo, H], cum over time
        group_size = torch.bincount(self.combo_mask, minlength=N_SUBSETS).clamp_min(1)
        self._combo_log_g = group_size.log()[self.combo_mask]  # [Ncombo]
        # prior: uniform over sizes 0..max_active, uniform over subsets of a given size
        prior = torch.full((N_SUBSETS,), float("-inf"), device=dev)
        for mask in range(N_SUBSETS):
            s = bin(mask).count("1")
            if s <= cfg.max_active:
                prior[mask] = -math.log(cfg.max_active + 1) - math.log(math.comb(N_LIB, s))
        self.log_prior = prior
        bits = torch.tensor([[1.0 if mask & (1 << i) else 0.0 for i in range(N_LIB)] for mask in range(N_SUBSETS)],
                             device=dev)
        self.cause_bits = bits  # [64, 6]
        self.n_combos = len(masks)

    @torch.no_grad()
    def log_subset_posterior(self, x: torch.Tensor, length: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Exact log p(subset | x), shape [B, 64]. x must be zero beyond each example's length."""
        b = x.shape[0]
        dot = x.reshape(b, -1).float() @ self._combo_flat.T  # [B, Ncombo]
        energy = self._combo_energy.T[length - 1]  # [B, Ncombo]
        ll = (dot - 0.5 * energy) / sigma[:, None].pow(2)
        ll = ll - self._combo_log_g[None, :]  # average, not sum, over each subset's nuisance grid
        idx = self.combo_mask[None, :].expand(b, -1)
        subset_max = torch.full((b, N_SUBSETS), -1e30, device=x.device)
        subset_max.scatter_reduce_(1, idx, ll, reduce="amax", include_self=True)
        shifted = (ll - subset_max.gather(1, idx)).exp()
        sumexp = torch.zeros((b, N_SUBSETS), device=x.device).scatter_add_(1, idx, shifted)
        loglik = subset_max + sumexp.clamp_min(1e-30).log()
        log_joint = loglik + self.log_prior[None, :]
        return torch.log_softmax(log_joint, dim=-1)

    # ---- sampling --------------------------------------------------------------------------
    def sample(self, n: int, gen: torch.Generator | None = None) -> dict:
        cfg, dev = self.cfg, self.device
        size = torch.randint(0, cfg.max_active + 1, (n,), device=dev, generator=gen)
        perm = torch.argsort(torch.rand(n, N_LIB, device=dev, generator=gen), dim=1)
        active_at_pos = torch.arange(N_LIB, device=dev)[None, :] < size[:, None]
        active = torch.zeros(n, N_LIB, dtype=torch.bool, device=dev).scatter_(1, perm, active_at_pos)

        obj = torch.randint(0, 2, (n,), device=dev, generator=gen)
        amp_idx = torch.randint(0, len(cfg.amplitude_grid), (n,), device=dev, generator=gen)
        amp = torch.tensor(cfg.amplitude_grid, device=dev)[amp_idx]
        onset_t = torch.tensor(cfg.onset_grid, device=dev, dtype=torch.float32)
        tau_idx = torch.randint(0, len(cfg.onset_grid), (n, N_LIB), device=dev, generator=gen)
        tau = onset_t[tau_idx]  # [n, N_LIB]

        mean = torch.zeros(n, cfg.horizon, self.n_channels, device=dev)
        for i, cause in enumerate(CAUSE_LIB):
            raw = _cause_signal_batch(cause, tau[:, i], amp, obj, cfg.horizon, self.n_channels)
            partner = _PAIR_PARTNER.get(cause)
            if partner:
                raw_p = _cause_signal_batch(partner, tau[:, i], amp, obj, cfg.horizon, self.n_channels)
                raw = raw + 0.5 * cfg.overlap * (raw_p - raw)
            mean = mean + active[:, i, None, None].to(raw.dtype) * raw

        lo, hi = math.log(cfg.noise_range[0]), math.log(cfg.noise_range[1])
        sigma = torch.exp(lo + (hi - lo) * torch.rand(n, device=dev, generator=gen))
        length = torch.randint(cfg.min_length, cfg.horizon + 1, (n,), device=dev, generator=gen)
        mask_t = (torch.arange(cfg.horizon, device=dev)[None] < length[:, None]).float()
        x = (mean + sigma[:, None, None] * torch.randn(n, cfg.horizon, self.n_channels, device=dev,
                                                        generator=gen)) * mask_t[..., None]
        ctx = torch.stack([sigma.log(), length.float() / cfg.horizon], -1)

        y_mask = (active.long() * (2 ** torch.arange(N_LIB, device=dev))).sum(-1)  # [n] bitmask label
        log_subset = self.log_subset_posterior(x, length, sigma)
        subset_post = log_subset.exp()
        marginal = subset_post @ self.cause_bits  # [n, 6]
        return {"x": x, "mask": mask_t, "ctx": ctx, "sigma": sigma, "length": length,
                "n_active": size, "y_multihot": active.float(), "y_subset": y_mask,
                "teacher_subset": subset_post, "teacher_marginal": marginal}
