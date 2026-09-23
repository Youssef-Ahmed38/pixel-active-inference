"""Evaluation of a trained thinker against the exact teacher and the true cause.

Key checks: student-teacher agreement, calibration, whether the HRM spends more thinking
segments on harder examples (difficulty = teacher posterior entropy, i.e. how ambiguous the
evidence really is, plus noise level and evidence length), accuracy vs thinking budget, speed.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from pai.thinker.multicause import MultiCauseConfig, MultiCauseTask
from pai.thinker.synthetic import CauseTask
from pai.train.common import amp_dtype, autocast


def ece(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 confidence."""
    conf, pred = probs.max(-1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            err += sel.float().mean().item() * abs(conf[sel].mean().item() - correct[sel].mean().item())
    return err


def score(probs: torch.Tensor, data: dict) -> dict:
    teacher, y = data["teacher"], data["y"]
    kl = (teacher * (teacher.clamp_min(1e-12).log() - probs.clamp_min(1e-12).log())).sum(-1)
    return {"acc": (probs.argmax(-1) == y).float().mean().item(),
            "teacher_agree": (probs.argmax(-1) == teacher.argmax(-1)).float().mean().item(),
            "kl_to_teacher": kl.mean().item(), "ece": ece(probs, y)}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else 0.0


def run_think(model, data: dict, batch: int, device, max_segments=None, halt=True):
    dtype = amp_dtype("auto", device)
    logits, segs = [], []
    for i in range(0, data["x"].shape[0], batch):
        sl = slice(i, i + batch)
        with autocast(device, dtype):
            res = model.think(data["x"][sl], data["mask"][sl], data["ctx"][sl], max_segments=max_segments, halt=halt)
        logits.append(res.logits["cause"].float())
        segs.append(res.segments)
    return torch.cat(logits).softmax(-1), torch.cat(segs), res.steps_per_segment


def _bins(values: torch.Tensor, n: int) -> tuple[torch.Tensor, list[float]]:
    edges = torch.quantile(values.float(), torch.linspace(0, 1, n + 1, device=values.device))
    idx = torch.bucketize(values.float(), edges[1:-1].contiguous())
    return idx, [round(e, 4) for e in edges.tolist()]


def difficulty_table(probs, segs, data: dict, n_bins: int = 5) -> dict:
    ent = -(data["teacher"] * data["teacher"].clamp_min(1e-12).log()).sum(-1)
    out = {}
    for name, v in [("teacher_entropy", ent), ("noise_sigma", data["sigma"]), ("evidence_length", data["length"].float())]:
        idx, edges = _bins(v, n_bins)
        rows = []
        for b in range(n_bins):
            sel = idx == b
            if not sel.any():
                continue
            rows.append({"range": [edges[b], edges[b + 1]], "n": int(sel.sum()),
                         "mean_segments": segs[sel].float().mean().item(),
                         "acc": (probs[sel].argmax(-1) == data["y"][sel]).float().mean().item(),
                         "teacher_acc": (data["teacher"][sel].argmax(-1) == data["y"][sel]).float().mean().item(),
                         "teacher_agree": (probs[sel].argmax(-1) == data["teacher"][sel].argmax(-1)).float().mean().item()})
        out[name] = {"bins": rows, "spearman_segments": _spearman(v.cpu().numpy(), segs.cpu().numpy())}
    return out


def _timed(fn, device, repeats: int = 3) -> float:
    fn()  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def speed(model, task: CauseTask, data: dict, device, batch: int) -> dict:
    """Examples/second at a large batch and at batch 1 (online use inside the agent)."""
    n = data["x"].shape[0]
    think = lambda b, m: [run_think(model, {k: v[:m] for k, v in data.items()}, b, device)]  # noqa: E731
    teach = lambda b, m: [task.posterior(data["x"][i:i + b], data["length"][i:i + b], data["sigma"][i:i + b])  # noqa: E731
                          for i in range(0, m, b)]
    m1 = 200
    return {"batched": {"batch": batch,
                        "thinker_ex_per_s": n / _timed(lambda: think(batch, n), device),
                        "teacher_ex_per_s": n / _timed(lambda: teach(batch, n), device)},
            "batch1": {"thinker_ex_per_s": m1 / _timed(lambda: think(1, m1), device, 1),
                       "teacher_ex_per_s": m1 / _timed(lambda: teach(1, m1), device, 1)}}


def evaluate_thinker(model, task: CauseTask, data: dict, device, ecfg: dict, heldout: dict | None = None) -> dict:
    model.eval()
    probs, segs, sps = run_think(model, data, ecfg["batch"], device)
    res = {"overall": score(probs, data),
           "mean_segments": segs.float().mean().item(),
           "mean_low_level_steps": (segs.float() * sps).mean().item(),
           "segment_histogram": torch.bincount(segs, minlength=int(segs.max()) + 1)[1:].tolist(),
           "difficulty": difficulty_table(probs, segs, data, ecfg["n_bins"])}
    max_seg = getattr(model, "max_segments", 1)
    if max_seg > 1:
        budget = []
        for m in range(1, max_seg + 1):
            p_h, s_h, _ = run_think(model, data, ecfg["batch"], device, max_segments=m, halt=True)
            p_f, _, _ = run_think(model, data, ecfg["batch"], device, max_segments=m, halt=False)
            budget.append({"max_segments": m, "halting": {**score(p_h, data), "mean_segments": s_h.float().mean().item()},
                           "forced": score(p_f, data)})
        res["budget"] = budget
    if heldout is not None:
        p_new, s_new, _ = run_think(model, heldout, ecfg["batch"], device)
        res["heldout_cause"] = {"mean_max_prob": p_new.max(-1).values.mean().item(),
                                "mean_max_prob_known": probs.max(-1).values.mean().item(),
                                "mean_segments": s_new.float().mean().item()}
    res["speed"] = speed(model, task, data, device, ecfg["speed_batch"])
    return res


def teacher_reference(task: CauseTask, data: dict, heldout: dict | None = None) -> dict:
    """The teacher scored like a student: the Bayes-optimal ceiling for accuracy and calibration."""
    out = {"overall": score(data["teacher"], data)}
    if heldout is not None:
        out["heldout_cause"] = {"mean_max_prob": heldout["teacher"].max(-1).values.mean().item()}
    return out


# ---------------------------------------------------------------------------------------------
# Multi-cause task: subset head (64-way, exact-subset accuracy / KL) + per-cause marginal head
# (6-way multi-label, BCE / per-cause accuracy). Mirrors the single-cause functions above; kept
# separate because the read-outs and metrics genuinely differ (multi-label vs single label).
# ---------------------------------------------------------------------------------------------

def run_think_multicause(model, data: dict, batch: int, device, max_segments=None, halt=True):
    dtype = amp_dtype("auto", device)
    subset_logits, marg_logits, segs = [], [], []
    for i in range(0, data["x"].shape[0], batch):
        sl = slice(i, i + batch)
        with autocast(device, dtype):
            res = model.think(data["x"][sl], data["mask"][sl], data["ctx"][sl], max_segments=max_segments, halt=halt)
        subset_logits.append(res.logits["subset"].float())
        marg_logits.append(res.logits["cause_marginal"].float())
        segs.append(res.segments)
    subset_p = torch.cat(subset_logits).softmax(-1)
    marg_p = torch.cat(marg_logits).sigmoid()
    return subset_p, marg_p, torch.cat(segs), res.steps_per_segment


def marginal_ece(marg_p: torch.Tensor, y_multihot: torch.Tensor, n_bins: int = 10) -> float:
    """Calibration of the per-cause sigmoid probabilities, pooled over all 6 causes."""
    p, y = marg_p.reshape(-1), y_multihot.reshape(-1)
    edges = torch.linspace(0, 1, n_bins + 1, device=p.device)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p > lo) & (p <= hi)
        if sel.any():
            err += sel.float().mean().item() * abs(p[sel].mean().item() - y[sel].mean().item())
    return err


def multicause_score(subset_p: torch.Tensor, marg_p: torch.Tensor, data: dict) -> dict:
    teacher_subset, teacher_marg = data["teacher_subset"], data["teacher_marginal"]
    y_subset, y_multihot = data["y_subset"], data["y_multihot"]
    kl_subset = (teacher_subset * (teacher_subset.clamp_min(1e-12).log() - subset_p.clamp_min(1e-12).log())).sum(-1)
    marg_pred = (marg_p > 0.5).float()
    teacher_marg_c = teacher_marg.clamp(0.0, 1.0)  # matmul of subset probs can round fractionally past 1
    bce_teacher = torch.nn.functional.binary_cross_entropy(marg_p.clamp(1e-6, 1 - 1e-6), teacher_marg_c, reduction="none")
    return {
        "exact_subset_acc": (subset_p.argmax(-1) == y_subset).float().mean().item(),
        "subset_teacher_agree": (subset_p.argmax(-1) == teacher_subset.argmax(-1)).float().mean().item(),
        "kl_subset_to_teacher": kl_subset.mean().item(),
        "subset_ece": ece(subset_p, y_subset, n_bins=10),
        "per_cause_acc": (marg_pred == y_multihot).float().mean().item(),
        "marginal_bce_to_teacher": bce_teacher.mean().item(),
        "marginal_bce_to_truth": torch.nn.functional.binary_cross_entropy(
            marg_p.clamp(1e-6, 1 - 1e-6), y_multihot, reduction="none").mean().item(),
        "marginal_ece": marginal_ece(marg_p, y_multihot),
    }


def multicause_difficulty_table(subset_p, marg_p, segs, data: dict, n_bins: int = 5) -> dict:
    ent = -(data["teacher_subset"] * data["teacher_subset"].clamp_min(1e-12).log()).sum(-1)
    out: dict = {}
    # discrete axis: number of simultaneous causes (0..max_active), exact groups, not quantile bins
    rows = []
    for k in sorted(data["n_active"].unique().tolist()):
        sel = data["n_active"] == k
        if not sel.any():
            continue
        rows.append({"n_active": int(k), "n": int(sel.sum()), "mean_segments": segs[sel].float().mean().item(),
                     "exact_subset_acc": (subset_p[sel].argmax(-1) == data["y_subset"][sel]).float().mean().item(),
                     "per_cause_acc": ((marg_p[sel] > 0.5).float() == data["y_multihot"][sel]).float().mean().item(),
                     "subset_teacher_agree": (subset_p[sel].argmax(-1) == data["teacher_subset"][sel].argmax(-1)).float().mean().item()})
    out["n_active_causes"] = {"bins": rows, "spearman_segments": _spearman(data["n_active"].cpu().numpy(), segs.cpu().numpy())}
    for name, v in [("teacher_subset_entropy", ent), ("noise_sigma", data["sigma"]), ("evidence_length", data["length"].float())]:
        idx, edges = _bins(v, n_bins)
        rows = []
        for b in range(n_bins):
            sel = idx == b
            if not sel.any():
                continue
            rows.append({"range": [edges[b], edges[b + 1]], "n": int(sel.sum()),
                         "mean_segments": segs[sel].float().mean().item(),
                         "exact_subset_acc": (subset_p[sel].argmax(-1) == data["y_subset"][sel]).float().mean().item(),
                         "per_cause_acc": ((marg_p[sel] > 0.5).float() == data["y_multihot"][sel]).float().mean().item(),
                         "subset_teacher_agree": (subset_p[sel].argmax(-1) == data["teacher_subset"][sel].argmax(-1)).float().mean().item()})
        out[name] = {"bins": rows, "spearman_segments": _spearman(v.cpu().numpy(), segs.cpu().numpy())}
    return out


def multicause_speed(model, task: MultiCauseTask, data: dict, device, batch: int) -> dict:
    n = data["x"].shape[0]
    think = lambda b, m: [run_think_multicause(model, {k: v[:m] for k, v in data.items()}, b, device)]  # noqa: E731
    teach = lambda b, m: [task.log_subset_posterior(data["x"][i:i + b], data["length"][i:i + b], data["sigma"][i:i + b])  # noqa: E731
                          for i in range(0, m, b)]
    m1 = 200
    return {"batched": {"batch": batch,
                        "thinker_ex_per_s": n / _timed(lambda: think(batch, n), device),
                        "teacher_ex_per_s": n / _timed(lambda: teach(batch, n), device)},
            "batch1": {"thinker_ex_per_s": m1 / _timed(lambda: think(1, m1), device, 1),
                       "teacher_ex_per_s": m1 / _timed(lambda: teach(1, m1), device, 1)}}


def evaluate_multicause(model, task: MultiCauseTask, data: dict, device, ecfg: dict) -> dict:
    model.eval()
    subset_p, marg_p, segs, sps = run_think_multicause(model, data, ecfg["batch"], device)
    res = {"overall": multicause_score(subset_p, marg_p, data),
           "mean_segments": segs.float().mean().item(),
           "mean_low_level_steps": (segs.float() * sps).mean().item(),
           "segment_histogram": torch.bincount(segs, minlength=int(segs.max()) + 1)[1:].tolist(),
           "difficulty": multicause_difficulty_table(subset_p, marg_p, segs, data, ecfg["n_bins"])}
    max_seg = getattr(model, "max_segments", 1)
    if max_seg > 1:
        budget = []
        for m in range(1, max_seg + 1):
            sp_h, mp_h, s_h, _ = run_think_multicause(model, data, ecfg["batch"], device, max_segments=m, halt=True)
            sp_f, mp_f, _, _ = run_think_multicause(model, data, ecfg["batch"], device, max_segments=m, halt=False)
            budget.append({"max_segments": m,
                           "halting": {**multicause_score(sp_h, mp_h, data), "mean_segments": s_h.float().mean().item()},
                           "forced": multicause_score(sp_f, mp_f, data)})
        res["budget"] = budget
    res["speed"] = multicause_speed(model, task, data, device, ecfg["speed_batch"])
    return res


def teacher_reference_multicause(data: dict) -> dict:
    """The exact teacher scored like a student (identity prediction): the Bayes-optimal ceiling."""
    return {"overall": multicause_score(data["teacher_subset"], data["teacher_marginal"], data)}


def overlap_sweep(model, base_cfg: MultiCauseConfig, device, n: int, batch: int, overlaps: list[float],
                  seed: int = 12345) -> list[dict]:
    """Evaluate an already-trained model on freshly generated data at several *other* overlap
    values (task-level knob, fixed within one training run) without retraining: overlap changes
    only the data generator, not the model's input shape, so this is a cheap way to see how much
    harder more-overlapping cause signatures make both the teacher's own task and the student."""
    rows = []
    for ov in overlaps:
        cfg = MultiCauseConfig(**{**base_cfg.__dict__, "overlap": ov})
        task = MultiCauseTask(cfg, device)
        data = task.sample(n, torch.Generator(device).manual_seed(seed))
        subset_p, marg_p, segs, _ = run_think_multicause(model, data, batch, device)
        rows.append({"overlap": ov, "mean_segments": segs.float().mean().item(),
                     "student": multicause_score(subset_p, marg_p, data),
                     "teacher": multicause_score(data["teacher_subset"], data["teacher_marginal"], data)})
    return rows
