"""Train a thinker (HRM or baseline) to reproduce the Bayesian teacher's cause posterior.

Loss per segment m (deep supervision):
    KL(teacher || p_m) + ce_weight * CE(p_m, true cause)
Halting (HRM only), Q-learning as in the paper with two choices of ours:
- reward for halting after segment m: r_m = 1 - TV(p_m, teacher), a soft version of the paper's
  0/1 "answer correct" reward (we supervise a distribution, not a single answer)
- ponder cost: continuing costs `ponder_cost` reward units per segment, so the Q-head only
  prefers to think on when that is expected to improve the answer by more than the cost.
Continue target: `q_target: oracle` uses the exact optimal-stopping value of the realised
trajectory, V_m = max(r_m, V_{m+1} - c) (the thinker is deterministic given its input, so this
is the true value, not a noisy one); `bootstrap` uses the paper's max(sigmoid(Q_{m+1})).

Deviation from the paper: we unroll all max_segments for every example inside one training
step and sum the segment losses before one optimizer step (the paper steps the optimizer per
segment and replaces halted examples). States are still detached between segments, so the
gradient approximation is the same; every segment's readout gets trained, which makes
epsilon-exploration of halting unnecessary (Q is supervised at every depth).
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from pai.thinker.baselines import FixedDepthTransformer, MLPThinker
from pai.thinker.hrm import HRM, SegmentOut
from pai.thinker.multicause import MultiCauseTask
from pai.thinker.synthetic import CauseTask
from pai.train.common import JsonlLogger, amp_dtype, autocast


def build_model(kind: str, task: CauseTask | MultiCauseTask, mcfg: dict) -> nn.Module:
    if isinstance(task, MultiCauseTask):
        heads = {"subset": task.n_subsets, "cause_marginal": task.n_causes}
    else:
        heads = {"cause": task.n_causes}
    common = dict(in_dim=task.n_channels, ctx_dim=task.ctx_dim, max_len=task.cfg.horizon, heads=heads)
    if kind == "hrm":
        return HRM(**common, **mcfg["hrm"])
    if kind == "transformer":
        return FixedDepthTransformer(**common, **mcfg["transformer"])
    if kind == "mlp":
        return MLPThinker(**common, **mcfg["mlp"])
    raise ValueError(f"unknown thinker kind {kind!r}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _halting_q_loss(outs: list[SegmentOut], rewards: list[torch.Tensor], ponder_cost: float,
                     q_target: str) -> tuple[torch.Tensor, dict] | tuple[None, dict]:
    """Shared Q-learning term (paper's ACT) for any per-segment reward sequence. See module
    docstring: r_m = 1 - TV(prediction_m, teacher), continue-target is the optimal-stopping value
    of the realised trajectory (oracle) or the paper's bootstrapped max(sigmoid(Q))."""
    if outs[0].q is None or len(outs) <= 1:
        return None, {}
    r = torch.stack(rewards, 1)  # [B, M]
    q = torch.stack([o.q for o in outs], 1)  # [B, M, 2]
    m = r.shape[1]
    cont = torch.empty_like(r[:, :-1])
    if q_target == "oracle":
        v = r[:, -1]
        for i in range(m - 2, -1, -1):
            cont[:, i] = v - ponder_cost
            v = torch.maximum(r[:, i], v - ponder_cost)
    else:  # bootstrap, as in the paper
        qs = torch.sigmoid(q.detach())
        nxt = torch.maximum(qs[:, 1:, 0], qs[:, 1:, 1])
        nxt[:, -1] = qs[:, -1, 0]  # the last segment must halt
        cont = nxt - ponder_cost
    q_loss = (F.binary_cross_entropy_with_logits(q[..., 0], r)
              + F.binary_cross_entropy_with_logits(q[:, :-1, 1], cont.clamp(0, 1)))
    stats = {"q_loss": q_loss.item(), "halt_rate_1": (q[:, 0, 0] > q[:, 0, 1]).float().mean().item()}
    return q_loss, stats


def thinker_loss(outs: list[SegmentOut], batch: dict, ce_weight: float = 0.1, q_weight: float = 0.5,
                 ponder_cost: float = 0.01, q_target: str = "oracle") -> tuple[torch.Tensor, dict]:
    teacher, y = batch["teacher"], batch["y"]
    log_t = teacher.clamp_min(1e-12).log()
    pred, rewards, kls = 0.0, [], []
    for o in outs:
        logp = F.log_softmax(o.logits["cause"], -1)
        kl = (teacher * (log_t - logp)).sum(-1)
        pred = pred + (kl + ce_weight * F.nll_loss(logp, y, reduction="none")).mean()
        kls.append(kl.mean().detach())
        rewards.append(1 - 0.5 * (logp.exp().detach() - teacher).abs().sum(-1))
    loss = pred / len(outs)
    stats = {"kl_first": kls[0].item(), "kl_last": kls[-1].item()}
    q_loss, q_stats = _halting_q_loss(outs, rewards, ponder_cost, q_target)
    if q_loss is not None:
        loss = loss + q_weight * q_loss
        stats.update(q_stats)
    return loss, stats


def multicause_loss(outs: list[SegmentOut], batch: dict, marginal_weight: float = 1.0, subset_weight: float = 1.0,
                    q_weight: float = 0.5, ponder_cost: float = 0.01, q_target: str = "oracle") -> tuple[torch.Tensor, dict]:
    """Same structure as thinker_loss (deep supervision + optional ACT halting), for the
    multi-cause task: KL to the teacher's subset posterior (the "explaining away" target) plus a
    BCE auxiliary on the per-cause marginals against the true multi-hot label. The halting reward
    is 1 - TV(subset prediction, teacher subset posterior), same recipe as the single-cause task."""
    teacher_subset, y_multihot = batch["teacher_subset"], batch["y_multihot"]
    log_ts = teacher_subset.clamp_min(1e-12).log()
    pred, rewards, kls, bces = 0.0, [], [], []
    for o in outs:
        logp_subset = F.log_softmax(o.logits["subset"], -1)
        kl = (teacher_subset * (log_ts - logp_subset)).sum(-1)
        bce = F.binary_cross_entropy_with_logits(o.logits["cause_marginal"], y_multihot, reduction="none").sum(-1)
        pred = pred + (subset_weight * kl + marginal_weight * bce).mean()
        kls.append(kl.mean().detach())
        bces.append(bce.mean().detach())
        rewards.append(1 - 0.5 * (logp_subset.exp().detach() - teacher_subset).abs().sum(-1))
    loss = pred / len(outs)
    stats = {"kl_subset_first": kls[0].item(), "kl_subset_last": kls[-1].item(),
             "bce_first": bces[0].item(), "bce_last": bces[-1].item()}
    q_loss, q_stats = _halting_q_loss(outs, rewards, ponder_cost, q_target)
    if q_loss is not None:
        loss = loss + q_weight * q_loss
        stats.update(q_stats)
    return loss, stats


def train_thinker(model: nn.Module, task: CauseTask | MultiCauseTask, tcfg: dict, device: torch.device,
                  logger: JsonlLogger | None = None, seed: int = 0, loss_fn=None) -> dict:
    """Train on freshly sampled batches (infinite data). Returns timing/throughput info.

    `loss_fn` defaults to `thinker_loss` (single-cause); pass `multicause_loss` for the
    multi-cause task. Its keyword args are taken from whichever of its own parameter names are
    present in `tcfg` (so the two losses' differently-named weights both work unmodified)."""
    if loss_fn is None:
        loss_fn = thinker_loss
    model.to(device).train()
    dtype = amp_dtype(tcfg.get("amp", "auto"), device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg["lr"], betas=(0.9, 0.95))
    steps, warmup = tcfg["steps"], tcfg["warmup_steps"]
    # Linear warm-up then cosine to 10%: converges within a short fixed budget (the paper uses a constant lr).
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * (
        0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    gen = torch.Generator(device).manual_seed(seed)
    loss_arg_names = set(loss_fn.__code__.co_varnames[:loss_fn.__code__.co_argcount])
    loss_kw = {k: v for k, v in tcfg.items() if k in loss_arg_names}
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = task.sample(tcfg["batch_size"], gen)
        with autocast(device, dtype):
            outs = model.forward_segments(batch["x"], batch["mask"], batch["ctx"])
        loss, stats = loss_fn(outs, batch, **loss_kw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if logger is not None and (step % tcfg["log_every"] == 0 or step == steps):
            logger.log(step=step, loss=loss.item(), lr=sched.get_last_lr()[0], **stats)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    return {"steps": steps, "wall_s": round(wall, 1), "examples_seen": steps * tcfg["batch_size"],
            "final_loss": loss.item()}
