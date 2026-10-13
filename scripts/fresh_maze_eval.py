"""Paired online-thinker and baseline evaluation on unseen maze layouts and appearances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.cognition.thinker import OnlineMazeCauseThinker  # noqa: E402
from pai.maze.advanced import run_advanced_episode  # noqa: E402
from pai.maze.env import MazeWorld  # noqa: E402


def evaluate(seeds=range(40000, 40006), checkpoint=None, budget=700,
             appearance_jitter=12.0):
    checkpoint = Path(checkpoint or ROOT / "runs/cross_domain_thinker/maze_transfer.pt")
    rows = []
    for seed in seeds:
        size = (15, 21)[seed % 2]
        sensor_range = (2, 3)[seed % 2]
        block_at = 20 + seed % 13
        shift_at = 38 + seed % 17
        settings = dict(seed=seed, size=size, sensor_range=sensor_range,
                        loop_probability=0.02 + 0.03 * (seed % 4),
                        appearance_jitter=appearance_jitter)
        for condition in ("baseline", "online_thinker"):
            thinker = OnlineMazeCauseThinker(checkpoint) if condition == "online_thinker" else None
            world = MazeWorld(**settings)
            result = run_advanced_episode(world, visual=True, mobile=True, budget=budget,
                                          block_at=block_at, shift_at=shift_at, odom_noise=0.35,
                                          seed=seed, online_thinker=thinker)
            reported = {r["cause"] for r in result["thinker_reports"] if r["actionable"]}
            rows.append({"seed": seed, "condition": condition, "size": size,
                         "goal_found": result["goal_found"],
                         "full_coverage": result["visited"] == result["reachable"],
                         "coverage": result["coverage"], "steps": result["steps"],
                         "truth_causes": result["truth_causes"],
                         "thinker_recall": {c: c in reported for c in result["truth_causes"]},
                         "thinker_actions": result["thinker_actions"]})
    summary = {}
    for condition in ("baseline", "online_thinker"):
        selected = [r for r in rows if r["condition"] == condition]
        summary[condition] = {"n": len(selected),
                              "goal_rate": sum(r["goal_found"] for r in selected) / len(selected),
                              "full_coverage_rate": sum(r["full_coverage"] for r in selected) / len(selected),
                              "mean_coverage": sum(r["coverage"] for r in selected) / len(selected),
                              "mean_steps": sum(r["steps"] for r in selected) / len(selected)}
    return {"config": {"seeds": list(seeds), "budget": budget, "checkpoint": str(checkpoint),
                       "appearance_jitter": appearance_jitter, "odometry_noise": 0.35},
            "summary": summary, "episodes": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-start", type=int, default=40000)
    ap.add_argument("--budget", type=int, default=700)
    ap.add_argument("--appearance-jitter", type=float, default=12.0)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs/cross_domain_thinker/maze_transfer.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "results/fresh_maze_eval.json")
    args = ap.parse_args()
    result = evaluate(range(args.seed_start, args.seed_start + args.seeds), args.checkpoint,
                      args.budget, args.appearance_jitter)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
