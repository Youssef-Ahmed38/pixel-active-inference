"""Evaluate the maze upper levels with a stepping body and one forced fall."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.advanced import run_advanced_episode  # noqa: E402
from pai.maze.env import MazeWorld  # noqa: E402


def evaluate(seeds=range(40000, 40010), budget=750, learned_model=None):
    classifier = None
    if learned_model is not None:
        from pai.maze.learned_vision import TileColorModel
        classifier = TileColorModel.load(learned_model)
    rows = []
    for seed in seeds:
        world = MazeWorld(seed, size=15 if seed % 2 == 0 else 21,
                          sensor_range=2 if seed % 2 == 0 else 3,
                          appearance_jitter=22.0 if classifier is not None else 12.0,
                          appearance_bias=(30.0, -20.0, 15.0) if classifier is not None else (0.0, 0.0, 0.0))
        result = run_advanced_episode(world, visual=True, mobile=True, walk=True,
                                      fall_at=80, budget=budget, odom_noise=0.35,
                                      seed=seed, visual_classifier=classifier)
        rows.append({"seed": seed, "goal": result["goal_found"],
                     "full_coverage": result["visited"] == result["reachable"],
                     "falls": result["falls"], "recoveries": result["recoveries"],
                     "steps": result["steps"]})
    return {"config": {"seeds": list(seeds), "budget": budget, "fall_at": 80,
                       "learned_model": str(learned_model) if learned_model is not None else None},
            "summary": {"n": len(rows), "goal_rate": sum(r["goal"] for r in rows) / len(rows),
                        "full_coverage_rate": sum(r["full_coverage"] for r in rows) / len(rows),
                        "recovery_rate": sum(r["recoveries"] == 1 for r in rows) / len(rows)},
            "episodes": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=int, default=750)
    ap.add_argument("--learned-model", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results/walking_maze_eval.json")
    args = ap.parse_args()
    result = evaluate(range(40000, 40000 + args.seeds), args.budget, args.learned_model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
