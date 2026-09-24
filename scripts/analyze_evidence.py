"""Offline look at stored cause-inference evidence: what the surprise looked like, and what the
explicit Bayesian teacher (pai.causes.inference) concludes from it.

slice_eval.py saves, per planner step, z = residual / sigma for the 12 evidence channels, the holding
flag, the task step index and the episode id (<tag>_evidence.npz). Since z is already standardised,
windows can be rebuilt with residual = z and sigma = 1 and fed to infer_cause again, so a change to
the hypotheses can be compared with the old ones on exactly the same evidence. This is an offline
proxy, not a new simulation: the agent's behaviour (and so the evidence) would change if the new
explanations were used online, and the trigger is replayed on raw z (the online monitor triggers on
the error left after the agent's own adaptations). Fitted parameters are in z units here (sigma = 1),
so their mm and N values are not physical.

    python scripts/analyze_evidence.py results/slice_v7            # confusion matrix + traces
    python scripts/analyze_evidence.py results/slice_v7 --cause slippery_object --show 8
    python scripts/analyze_evidence.py results/slice_v7 --before data/analysis/inference_before.py

--before replays a saved copy of an older inference.py too and prints both confusion matrices.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from pai.causes.inference import CAUSES, CHANNELS, StepEvidence, infer_cause

STEPS = ("above_object", "at_object", "grasped", "lifted", "over_target", "lowered", "opened", "released")
WINDOW = 20  # as in SurpriseMonitor


def load(folder: Path, tag: str = "slice"):
    ev = np.load(folder / f"{tag}_evidence.npz")
    calib = json.loads((folder / f"{tag}_calibration.json").read_text())
    rows = json.loads((folder / f"{tag}_eval.json").read_text())["episodes"]
    return ev, calib, rows


def episode_evidence(ev, e: int) -> list[StepEvidence]:
    m = ev["episode"] == e
    z, hold, phase = ev["z"][m].astype(np.float64), ev["holding"][m], ev["phase"][m]
    ones = np.ones(len(CHANNELS))
    return [StepEvidence(t, z[t], ones, np.zeros(4), bool(hold[t]),
                         STEPS[phase[t]] if phase[t] < len(STEPS) else "done") for t in range(len(z))]


def triggered(evs: list[StepEvidence], calib: dict) -> bool:
    """Replays SurpriseMonitor's trigger (3-step mean surprise above the per-step threshold)."""
    thr = calib["per_step_thresholds"]
    for k in range(len(evs)):
        level = np.mean([e.surprise for e in evs[max(0, k - 2): k + 1]])
        if level > thr.get(evs[k].step, calib["threshold"]):
            return True
    return False


def peak_window(evs: list[StepEvidence]) -> tuple[int, int]:
    """The window SurpriseMonitor.episode_verdict explains."""
    peak = max(range(len(evs)), key=lambda i: evs[i].surprise)
    lo = max(0, peak - WINDOW + 5)
    return lo, min(len(evs), lo + WINDOW)


def verdict(evs: list[StepEvidence], calib: dict, infer=infer_cause):
    if not evs or not triggered(evs, calib):
        return "none", None
    lo, hi = peak_window(evs)
    rep = infer(evs[lo:hi], 0.1)
    return rep.best, rep


def load_infer(path: Path):
    """infer_cause from another copy of inference.py (e.g. the version before a change)."""
    spec = importlib.util.spec_from_file_location("inference_before", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod.infer_cause


def confusion(true: list[str], pred: list[str]) -> str:
    cols = CAUSES
    lines = ["| true \\ inferred | " + " | ".join(cols) + " | accuracy |",
             "|---" * (len(cols) + 2) + "|"]
    for c in dict.fromkeys(true):
        cnt = Counter(p for t, p in zip(true, pred) if t == c)
        n = sum(cnt.values())
        lines.append(f"| {c} | " + " | ".join(str(cnt.get(k, 0)) for k in cols) + f" | {cnt.get(c, 0)}/{n} |")
    return "\n".join(lines)


def trace(evs: list[StepEvidence], lo: int, hi: int, width: int = 6) -> str:
    """z at the peak window on the channels that matter, the holding flag and the task step."""
    peak = max(range(lo, hi), key=lambda i: evs[i].surprise)
    rows = [f"    {'t':>4s} {'step':12s} hold " + " ".join(f"{c:>9s}" for c in CHANNELS[:9]) + "   surprise"]
    for t in range(max(lo, peak - width), min(hi, peak + width + 1)):
        e = evs[t]
        mark = "*" if t == peak else " "
        rows.append(f"   {mark}{t:4d} {e.step:12s} {'H' if e.holding else '.':>4s} "
                    + " ".join(f"{v:9.1f}" for v in e.residual[:9]) + f"   {e.surprise:8.0f}")
    return "\n".join(rows)


def summary_stats(evs: list[StepEvidence], lo: int, hi: int) -> dict:
    """Per-episode features of the peak window: which channels dominate, their sign, how long."""
    Z = np.stack([e.residual for e in evs[lo:hi]])
    hold = np.array([e.holding for e in evs[lo:hi]])
    big = (np.abs(Z) > 3).sum(0)  # steps with |z| > 3 per channel
    sig = Z.sum(0)
    peak = int(np.argmax(0.5 * (Z**2).sum(1)))
    after = hold[peak:]
    return {"gripper_steps": int(big[:3].max()), "force_z_sum": round(float(sig[5]), 1),
            "force_xy_steps": int(big[3:5].max()), "object_z_sum": round(float(sig[8]), 1),
            "object_xy_steps": int(big[6:8].max()), "hold_at_peak": bool(hold[peak]),
            "released_after_peak": bool(hold[peak] and not after.all()),
            "peak_step": evs[lo + peak].step}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--tag", default="slice")
    ap.add_argument("--cause", default="slippery_object", help="condition whose episodes to show in detail")
    ap.add_argument("--compare", default="push", help="condition shown next to it")
    ap.add_argument("--show", type=int, default=0, help="print z traces of this many episodes per group")
    ap.add_argument("--json", type=Path, help="write per-episode verdicts here")
    ap.add_argument("--before", type=Path, help="an older inference.py to replay on the same evidence")
    args = ap.parse_args()

    ev, calib, rows = load(args.folder, args.tag)
    labels = [str(x) for x in ev["labels"]]
    recorded = [r["inferred"] for r in rows]
    preds, per_ep = [], []
    for e in range(len(labels)):
        evs = episode_evidence(ev, e)
        best, rep = verdict(evs, calib)
        preds.append(best)
        lo, hi = peak_window(evs) if evs else (0, 0)
        per_ep.append({"episode": e, "true": labels[e], "recorded": recorded[e], "offline": best,
                       "params": rep.params.get(best) if rep else None,
                       **(summary_stats(evs, lo, hi) if evs else {})})

    print("## Recorded in simulation (the agent's own episode verdicts)\n")
    print(confusion(labels, recorded))
    if args.before:
        before = load_infer(args.before)
        old = [verdict(episode_evidence(ev, e), calib, before)[0] for e in range(len(labels))]
        print(f"\n## Offline replay with {args.before} (z as residual, sigma = 1)\n")
        print(confusion(labels, old))
        changed = [f"ep {e} ({labels[e]}) {a} -> {b}" for e, (a, b) in enumerate(zip(old, preds)) if a != b]
        print("\nchanged by the current version: " + (", ".join(changed) or "none"))
    print("\n## Offline replay with the current pai.causes.inference (z as residual, sigma = 1)\n")
    print(confusion(labels, preds))
    agree = np.mean([a == b for a, b in zip(recorded, preds)])
    print(f"\noffline replay agrees with the recorded verdict on {agree:.0%} of episodes")

    for cond in (args.cause, args.compare):
        eps = [p for p in per_ep if p["true"] == cond]
        print(f"\n## {cond}: peak-window features (offline verdict)\n")
        keys = ["episode", "offline", "peak_step", "hold_at_peak", "released_after_peak", "gripper_steps",
                "force_z_sum", "force_xy_steps", "object_z_sum", "object_xy_steps"]
        print(" ".join(f"{k[:14]:>14s}" for k in keys))
        for p in eps:
            print(" ".join(f"{str(p.get(k, '-'))[:14]:>14s}" for k in keys))
        if args.show:
            for group in sorted({p["offline"] for p in eps}):
                for p in [p for p in eps if p["offline"] == group][: args.show]:
                    evs = episode_evidence(ev, p["episode"])
                    lo, hi = peak_window(evs)
                    print(f"\n  episode {p['episode']} ({cond}, inferred {group}): {p['params']}")
                    print(trace(evs, lo, hi))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(per_ep, indent=1))


if __name__ == "__main__":
    main()
