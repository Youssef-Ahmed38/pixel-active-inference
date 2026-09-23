"""Train the entity world-model ensemble with a Gaussian negative log-likelihood.

Each member trains on its own bootstrap resample of the transitions, so members disagree where
data is thin. Entities that actually move are up-weighted: blocks sit still most of the time,
and without the weighting the model would learn "nothing ever moves".
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch

from pai.train.common import JsonlLogger, Progress, amp_dtype, autocast, git_commit, load_checkpoint, save_checkpoint
from pai.world.data import ACTION_DIM, load_transitions
from pai.world.entities import DYNAMIC, FORCE
from pai.world.model import EnsembleWorldModel

MOVING_THRESHOLD = 0.002  # m per planner step
MOVING_WEIGHT = 10.0


def _lr(step: int, total: int, peak: float, warmup_frac: float = 0.05) -> float:
    """Warm-up then cosine decay, computed from the step alone so resuming (even with a new
    total) needs no saved learning-rate state."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return peak * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def _gaussian_nll(mean, logvar, target, weight):
    per = 0.5 * (logvar + (target - mean) ** 2 / logvar.exp())
    return (per.mean(-1) * weight).sum() / weight.sum()


def train_world_model(cfg, device: str | None = None) -> Path:
    wc = cfg.world_model
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)
    x, a, y, ep = load_transitions(wc.data_dir)
    deltas = y[..., DYNAMIC] - x[..., DYNAMIC]
    forces = y[:, 0, FORCE]  # next wrist force, predicted as an absolute value
    n = len(x)
    rng = np.random.default_rng(cfg.seed)
    # Hold out whole episodes: transitions from one episode are strongly correlated, so a random
    # transition-level split would leak and make validation error look better than it is.
    n_ep = int(ep.max()) + 1
    val_eps = rng.choice(n_ep, max(1, int(0.05 * n_ep)), replace=False)
    is_val = np.isin(ep, val_eps)
    val_idx, train_idx = np.flatnonzero(is_val), np.flatnonzero(~is_val)

    X, A, D, Fz = (torch.as_tensor(v, device=device) for v in (x, a, deltas, forces))
    moving = (D[..., :3].norm(dim=-1) > MOVING_THRESHOLD).float()
    W = 1.0 + (MOVING_WEIGHT - 1.0) * moving

    model = EnsembleWorldModel(ACTION_DIM, n_members=wc.n_members, d_model=wc.d_model,
                               n_layers=wc.n_layers, n_heads=wc.n_heads).to(device)
    model.set_normalisation(X[train_idx], A[train_idx], D[train_idx], Fz[train_idx])
    opt = torch.optim.AdamW(model.parameters(), lr=wc.lr, weight_decay=1e-4)
    dtype = amp_dtype("auto", device)
    boot = [torch.as_tensor(rng.choice(train_idx, len(train_idx)), device=device) for _ in range(wc.n_members)]
    out_dir = Path(wc.out_dir)
    start = 1
    ckpt = load_checkpoint(out_dir)  # resume after an interruption instead of starting over
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["step"] + 1
    logger = JsonlLogger(out_dir / "log.jsonl")
    logger.log(event="start", step=start - 1, transitions=n, val_episodes=len(val_eps), members=wc.n_members,
               commit=git_commit(), params=sum(p.numel() for p in model.parameters()))
    dn = (D - model.delta_mean) / model.delta_std
    fn = (Fz - model.force_mean) / model.force_std
    t0 = time.time()
    progress = Progress(wc.steps - start + 1, "world model", every=100, unit=" steps")
    for step in range(start, wc.steps + 1):
        for g in opt.param_groups:
            g["lr"] = _lr(step, wc.steps, wc.lr)
        loss = 0.0
        for i in range(wc.n_members):
            b = boot[i][torch.randint(len(boot[i]), (wc.batch_size,), device=device)]
            with autocast(device, dtype):
                mean, logvar, fmean, flogvar = model.member_forward(i, X[b], A[b])
            loss = loss + _gaussian_nll(mean.float(), logvar.float(), dn[b], W[b])
            f_nll = 0.5 * (flogvar.float() + (fn[b] - fmean.float()) ** 2 / flogvar.float().exp())
            loss = loss + f_nll.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        progress.update(extra=f"loss {loss.item() / wc.n_members:.3f}")
        if step % wc.log_every == 0 or step == wc.steps:
            save_checkpoint(out_dir / "ckpt_last.pt", {"model": model.state_dict(), "opt": opt.state_dict(),
                                                       "step": step})
            logger.log(step=step, loss=loss.item() / wc.n_members, **evaluate_one_step(model, X, A, D, val_idx, Fz),
                       elapsed=round(time.time() - t0, 1))
    path = out_dir / "world_model.pt"
    save_checkpoint(path, {"model": model.state_dict(), "model_config": model.config(), "cfg": cfg.to_dict(),
                           "commit": git_commit()})
    return path


@torch.no_grad()
def evaluate_one_step(model, X, A, D, idx, F=None, batch: int = 4096) -> dict:
    """Validation error in raw units: position error (mm) for all and for moving entities, wrist force error (N)."""
    model.eval()
    errs, moving_errs, f_errs = [], [], []
    for s in range(0, len(idx), batch):
        b = torch.as_tensor(idx[s : s + batch], device=X.device)
        pred = model.predict(X[b], A[b])
        mean = pred.delta
        if F is not None:
            f_errs.append((pred.force - F[b]).norm(dim=-1))
        e = (mean[..., :3] - D[b][..., :3]).norm(dim=-1)
        mov = D[b][..., :3].norm(dim=-1) > MOVING_THRESHOLD
        errs.append(e.flatten())
        moving_errs.append(e[mov])
    model.train()
    e, m = torch.cat(errs), torch.cat(moving_errs)
    out = {"val_pos_err_mm": round(1000 * e.mean().item(), 3), "val_moving_pos_err_mm": round(1000 * m.mean().item(), 3)}
    if f_errs:
        out["val_force_err_N"] = round(torch.cat(f_errs).mean().item(), 3)
    return out
