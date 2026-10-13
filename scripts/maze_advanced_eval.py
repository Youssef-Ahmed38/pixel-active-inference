"""Evaluate visual memory, odometry noise, mobile movement and disturbances."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.advanced import run_advanced_episode  # noqa: E402
from pai.maze.env import MazeWorld  # noqa: E402


def evaluate(seeds=5, sizes=(15, 21), sensor_ranges=(2, 3), budget=700, odom_noise=0.35):
    rows = []
    for size in sizes:
        for sensor_range in sensor_ranges:
            for seed in range(5000, 5000 + seeds):
                world = MazeWorld(seed, size=size, sensor_range=sensor_range)
                result = run_advanced_episode(world, visual=True, mobile=True, budget=budget,
                                              block_at=30, shift_at=50, odom_noise=odom_noise,
                                              seed=seed)
                rows.append({"seed": seed, "size": size, "sensor_range": sensor_range, **result})
    return {"config": {"seeds": seeds, "sizes": sizes, "sensor_ranges": sensor_ranges,
                       "budget": budget, "odom_noise": odom_noise,
                       "disturbances": ["passage_blocked", "camera_shift"]},
            "summary": {
                "goal_rate": float(np.mean([r["goal_found"] for r in rows])),
                "full_coverage_rate": float(np.mean([r["visited"] == r["reachable"] for r in rows])),
                "mean_coverage": float(np.mean([r["coverage"] for r in rows])),
                "mean_pose_error_cells": float(np.mean([r["mean_pose_error"] for r in rows])),
                "mean_raw_odom_error_cells": float(np.mean([r["mean_raw_odom_error"] for r in rows])),
                "block_detection_rate": float(np.mean([r["cause_recall"].get("passage_blocked", False)
                                                        for r in rows if "passage_blocked" in r["truth_causes"]]))
                if any("passage_blocked" in r["truth_causes"] for r in rows) else None,
                "shift_detection_rate": float(np.mean([r["cause_recall"].get("camera_shift", False)
                                                        for r in rows if "camera_shift" in r["truth_causes"]]))
                if any("camera_shift" in r["truth_causes"] for r in rows) else None,
            }, "episodes": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=700)
    ap.add_argument("--odom-noise", type=float, default=0.35)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "maze_advanced.json")
    args = ap.parse_args()
    result = evaluate(args.seeds, budget=args.budget, odom_noise=args.odom_noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
