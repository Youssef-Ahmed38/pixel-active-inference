"""Train one HRM on arm + maze causes and test arm-to-maze few-shot transfer.

No Kaggle Job A or C data is required. The arm input is the existing local v7
evidence; maze episodes are generated locally. Splits are by complete episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.cognition.thinker_data import CAUSES, FEATURES, WINDOW, EvidenceSet, arm_evidence, maze_evidence  # noqa: E402
from pai.thinker.hrm import HRM  # noqa: E402


def _model(seed: int) -> HRM:
    torch.manual_seed(seed)
    return HRM(len(FEATURES), 0, WINDOW, {"cause": len(CAUSES)}, d_model=32, n_heads=4,
               h_layers=1, l_layers=1, h_cycles=1, l_steps=1, max_segments=2)


def _split_by_label(y: np.ndarray, domain: np.ndarray, seed: int, train_per_label: int):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for d in sorted(set(domain.tolist())):
        for label in sorted(set(y[domain == d].tolist())):
            idx = rng.permutation(np.flatnonzero((domain == d) & (y == label)))
            train.extend(idx[:train_per_label])
            test.extend(idx[train_per_label:])
    return np.asarray(train, int), np.asarray(test, int)


def _train(model, data, indices, steps: int, seed: int, lr=0.001,
           optimizer=None, rng=None):
    if not len(indices):
        return optimizer, rng
    rng = rng if rng is not None else np.random.default_rng(seed)
    x = torch.tensor(data.x, dtype=torch.float32)
    y = torch.tensor(data.y, dtype=torch.long)
    teacher = torch.tensor(data.teacher, dtype=torch.float32)
    opt = optimizer if optimizer is not None else torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        sample = (rng.permutation(indices) if len(indices) <= 32
                  else rng.choice(indices, size=32, replace=False))
        batch = x[sample]
        logits = model.forward_segments(batch, torch.ones(batch.shape[:2]), n_segments=1)[-1].logits["cause"]
        loss = F.cross_entropy(logits, y[sample]) + 0.3 * F.kl_div(
            F.log_softmax(logits, -1), teacher[sample], reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return opt, rng


def _score(model, data, indices):
    if not len(indices):
        return {"n": 0, "accuracy": None}
    model.eval()
    with torch.no_grad():
        x = torch.tensor(data.x[indices], dtype=torch.float32)
        logits = model.forward_segments(x, torch.ones(x.shape[:2]), n_segments=1)[-1].logits["cause"]
        pred = logits.argmax(-1).numpy()
    truth = data.y[indices]
    return {"n": len(indices), "accuracy": float(np.mean(pred == truth)),
            "by_cause": {CAUSES[c]: {"n": int((truth == c).sum()),
                                     "correct": int(((truth == c) & (pred == c)).sum())}
                         for c in sorted(set(truth.tolist()))}}


def evaluate(arm_file, maze_per_cause=16, arm_steps=120, maze_steps=80, seed=0,
             checkpoint_dir: Path | None = None, maze_train_per_cause: int = 3):
    torch.set_num_threads(2)
    arm = arm_evidence(arm_file)
    cache = checkpoint_dir / f"maze_evidence_raw_v3_{maze_per_cause}.npz" if checkpoint_dir else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            maze = EvidenceSet(*(saved[field] for field in ("x", "y", "teacher", "domain", "source")))
    else:
        maze = maze_evidence(maze_per_cause)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, x=maze.x, y=maze.y, teacher=maze.teacher,
                                domain=maze.domain, source=maze.source)
    arm_train, arm_test = _split_by_label(arm.y, arm.domain, seed, 12)
    maze_train, maze_test = _split_by_label(maze.y, maze.domain, seed + 1, maze_train_per_cause)
    base = _model(seed)
    initial = {k: v.detach().clone() for k, v in base.state_dict().items()}
    pretrained = _model(seed)
    pretrained.load_state_dict(initial)
    _train(pretrained, arm, arm_train, arm_steps, seed)
    arm_result = _score(pretrained, arm, arm_test)
    transfer = _model(seed)
    transfer.load_state_dict(pretrained.state_dict())
    scratch = _model(seed)
    scratch.load_state_dict(initial)
    curve = []
    prior_step = 0
    transfer_opt = scratch_opt = transfer_rng = scratch_rng = None
    checkpoints = sorted({0, 1, 2, 5, 10, 20, 40, 80, 120, maze_steps})
    for step in checkpoints:
        if step > maze_steps:
            continue
        increment = step - prior_step
        if increment:
            transfer_opt, transfer_rng = _train(transfer, maze, maze_train, increment, seed + 2,
                                                optimizer=transfer_opt, rng=transfer_rng)
            scratch_opt, scratch_rng = _train(scratch, maze, maze_train, increment, seed + 2,
                                              optimizer=scratch_opt, rng=scratch_rng)
        curve.append({"maze_steps": step,
                      "transfer_accuracy": _score(transfer, maze, maze_test)["accuracy"],
                      "scratch_accuracy": _score(scratch, maze, maze_test)["accuracy"]})
        prior_step = step
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for name, model in (("arm_pretrained", pretrained), ("maze_transfer", transfer),
                            ("maze_scratch", scratch)):
            payload = {"model": model.state_dict(), "causes": CAUSES, "features": FEATURES,
                       "window": WINDOW, "seed": seed}
            torch.save(payload, checkpoint_dir / f"{name}_seed{seed}.pt")
            if seed == 0:
                torch.save(payload, checkpoint_dir / f"{name}.pt")
    return {"config": {"arm_file": str(arm_file), "maze_per_cause": maze_per_cause,
                       "arm_steps": arm_steps, "maze_steps": maze_steps, "seed": seed,
                       "maze_train_per_cause": maze_train_per_cause},
            "causes": list(CAUSES), "features": list(FEATURES),
            "splits": {"arm_train": len(arm_train), "arm_test": len(arm_test),
                       "maze_train": len(maze_train), "maze_test": len(maze_test)},
            "arm_pretrained": arm_result,
            "learning_curve": curve,
            "maze_transfer": _score(transfer, maze, maze_test),
            "maze_scratch": _score(scratch, maze, maze_test)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-file", type=Path, default=ROOT / "results" / "slice_v7" / "slice_evidence.npz")
    ap.add_argument("--maze-per-cause", type=int, default=16)
    ap.add_argument("--arm-steps", type=int, default=120)
    ap.add_argument("--maze-steps", type=int, default=80)
    ap.add_argument("--maze-train-per-cause", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "cross_domain_thinker.json")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "runs" / "cross_domain_thinker")
    args = ap.parse_args()
    report = evaluate(args.arm_file, args.maze_per_cause, args.arm_steps, args.maze_steps, args.seed,
                      args.checkpoint_dir, args.maze_train_per_cause)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("splits", "arm_pretrained", "learning_curve",
                                            "maze_transfer", "maze_scratch")}, indent=2))


if __name__ == "__main__":
    main()
