"""Ablate physical causal probes after the visual maze map runs out of frontiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.advanced import run_advanced_episode  # noqa: E402
from pai.maze.env import MazeWorld  # noqa: E402


def evaluate(seeds=range(40000, 40020), budget=700):
    rows = []
    for seed in seeds:
        settings = dict(seed=seed, size=15 if seed % 2 == 0 else 21,
                        sensor_range=2 if seed % 2 == 0 else 3,
                        loop_probability=0.02 + 0.03 * (seed % 4),
                        appearance_jitter=12.0)
        for probes in (False, True):
            world = MazeWorld(**settings)
            result = run_advanced_episode(world, visual=True, mobile=True, budget=budget,
                                          block_at=20 + seed % 13, shift_at=38 + seed % 17,
                                          odom_noise=0.35, seed=seed, causal_probes=probes)
            rows.append({"seed": seed, "probes": probes,
                         "goal": result["goal_found"],
                         "full_coverage": result["visited"] == result["reachable"],
                         "coverage": result["coverage"],
                         "causal_probes": result["causal_probes"],
                         "steps": result["steps"]})
    summary = {}
    for condition in (False, True):
        selected = [r for r in rows if r["probes"] == condition]
        summary["with_probes" if condition else "without_probes"] = {
            "n": len(selected),
            "goal_rate": sum(r["goal"] for r in selected) / len(selected),
            "full_coverage_rate": sum(r["full_coverage"] for r in selected) / len(selected),
            "mean_coverage": sum(r["coverage"] for r in selected) / len(selected)}
    return {"config": {"seeds": list(seeds), "budget": budget},
            "summary": summary, "episodes": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=int, default=700)
    ap.add_argument("--out", type=Path, default=ROOT / "results/causal_probe_eval.json")
    args = ap.parse_args()
    result = evaluate(range(40000, 40000 + args.seeds), args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
