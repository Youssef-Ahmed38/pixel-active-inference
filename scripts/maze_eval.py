"""Paired maze coverage and goal search versus a random walker.

    python scripts/maze_eval.py --episodes 30 --budget 600 --require-complete
    python scripts/maze_eval.py --episodes 30 --budget 160  # fixed-budget comparison
    python scripts/maze_eval.py --episodes 5 --budget 100 --out results/maze_smoke.json

Each policy sees only local observations. Truth is read after the episode for scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze import Explorer, MazeWorld, RandomWalker, run_episode  # noqa: E402


def evaluate(episodes: int = 30, budget: int = 600, size: int = 21, sensor_range: int = 3,
             seed: int = 0, stop_at_goal: bool = False) -> dict:
    rows = []
    for i in range(episodes):
        scene_seed = seed + i
        for name in ("explorer", "random"):
            world = MazeWorld(scene_seed, size=size, sensor_range=sensor_range)
            agent = (Explorer(sensor_range, size, seed=scene_seed) if name == "explorer"
                     else RandomWalker(seed=scene_seed))
            rows.append({"scene_seed": scene_seed, "agent": name,
                         **run_episode(world, agent, budget, stop_at_goal)})
    summary = {}
    for name in ("explorer", "random"):
        part = [r for r in rows if r["agent"] == name]
        summary[name] = {
            "mean_coverage": float(np.mean([r["coverage"] for r in part])),
            "full_coverage_rate": float(np.mean([r["visited"] == r["reachable"] for r in part])),
            "goal_found_rate": float(np.mean([r["found_goal"] for r in part])),
            "mean_goal_step_capped": float(np.mean([r["goal_step"] or budget for r in part])),
            "max_steps": max(r["steps"] for r in part),
            "mean_collisions": float(np.mean([r["collisions"] for r in part])),
        }
    return {"config": {"episodes": episodes, "budget": budget, "size": size,
                       "sensor_range": sensor_range, "seed": seed, "stop_at_goal": stop_at_goal},
            "summary": summary, "episodes": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--sensor-range", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop-at-goal", action="store_true")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail unless the explorer reaches the goal and visits every reachable cell in every maze")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "maze_eval.json")
    args = ap.parse_args()
    report = evaluate(args.episodes, args.budget, args.size, args.sensor_range, args.seed,
                      args.stop_at_goal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"wrote {args.out}")
    if args.require_complete and any(r["agent"] == "explorer" and
                                     (not r["found_goal"] or r["visited"] != r["reachable"])
                                     for r in report["episodes"]):
        raise SystemExit("maze completion gate failed")


if __name__ == "__main__":
    main()
