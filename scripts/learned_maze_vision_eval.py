"""Train a local pixel classifier and score unseen palettes and maze rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.advanced import run_advanced_episode  # noqa: E402
from pai.maze.env import MazeWorld, VISUAL_COLORS  # noqa: E402
from pai.maze.learned_vision import TileColorModel, simulator_samples  # noqa: E402


def evaluate(train_seeds=range(100, 140), test_seeds=range(40000, 40010),
             appearance_jitter=22.0, appearance_bias=(30.0, -20.0, 15.0)):
    x_train, y_train = simulator_samples(train_seeds, appearance_jitter=appearance_jitter,
                                          appearance_bias=appearance_bias)
    model = TileColorModel.fit(x_train, y_train)
    x_test, y_test = simulator_samples(test_seeds, appearance_jitter=appearance_jitter,
                                       appearance_bias=appearance_bias)
    fixed_palette = np.asarray(list(VISUAL_COLORS.values()), dtype=float)
    fixed_prediction = ((x_test[:, None, :] - fixed_palette) ** 2).sum(-1).argmin(-1)
    learned_prediction = model.predict_rgb(x_test)
    rows = []
    for seed in test_seeds:
        settings = dict(seed=seed, size=15 if seed % 2 == 0 else 21,
                        sensor_range=2 if seed % 2 == 0 else 3,
                        appearance_jitter=appearance_jitter,
                        appearance_bias=appearance_bias)
        world = MazeWorld(**settings)
        observed_agents = []
        def capture(world, agent, observation, causes):
            observed_agents[:] = [agent]
        result = run_advanced_episode(world, visual=True, mobile=True, budget=700,
                                      block_at=20 + seed % 13, shift_at=38 + seed % 17,
                                      odom_noise=0.35, seed=seed, visual_classifier=model,
                                      frame_hook=capture)
        labels = observed_agents[-1].localizer.map_labels
        occupancy_checks = [(kind != "wall") == (world.grid[cell] == 0)
                            for cell, kind in labels.items()
                            if 0 <= cell[0] < world.size and 0 <= cell[1] < world.size]
        rows.append({"seed": seed, "goal": result["goal_found"],
                     "full_coverage": result["visited"] == result["reachable"],
                     "coverage": result["coverage"], "steps": result["steps"],
                     "map_occupancy_accuracy": float(np.mean(occupancy_checks))})
    return model, {"config": {"train_seeds": list(train_seeds), "test_seeds": list(test_seeds),
                              "appearance_jitter": appearance_jitter,
                              "appearance_bias": list(appearance_bias)},
                   "pixel_accuracy": {"fixed_palette": float(np.mean(fixed_prediction == y_test)),
                                      "learned": float(np.mean(learned_prediction == y_test))},
                   "summary": {"n": len(rows),
                               "goal_rate": sum(r["goal"] for r in rows) / len(rows),
                               "full_coverage_rate": sum(r["full_coverage"] for r in rows) / len(rows),
                               "mean_map_occupancy_accuracy": sum(r["map_occupancy_accuracy"] for r in rows)
                               / len(rows)},
                   "episodes": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-seeds", type=int, default=40)
    ap.add_argument("--test-seeds", type=int, default=10)
    ap.add_argument("--out", type=Path, default=ROOT / "results/learned_maze_vision_eval.json")
    ap.add_argument("--model", type=Path, default=ROOT / "runs/maze_tile_vision.npz")
    args = ap.parse_args()
    model, result = evaluate(range(100, 100 + args.train_seeds),
                             range(40000, 40000 + args.test_seeds))
    model.save(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"pixel_accuracy": result["pixel_accuracy"],
                      "summary": result["summary"]}, indent=2))
