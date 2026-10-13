"""Train a frontier preference on small mazes; test it on larger unseen mazes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze import Explorer, MazeWorld, RandomWalker, run_episode  # noqa: E402


def _score(seeds, size, budget, gain_weight):
    rows = []
    for seed in seeds:
        world = MazeWorld(seed, size=size)
        agent = (RandomWalker(seed) if gain_weight is None else
                 Explorer(size=size, seed=seed, gain_weight=gain_weight))
        rows.append(run_episode(world, agent, budget))
    return {"mean_coverage": float(np.mean([r["coverage"] for r in rows])),
            "goal_rate": float(np.mean([r["found_goal"] for r in rows])),
            "mean_goal_step_capped": float(np.mean([r["goal_step"] or budget for r in rows]))}


def evaluate(train_episodes=12, test_episodes=30, budget=160, train_size=15, test_size=21):
    train = list(range(2000, 2000 + train_episodes))
    test = list(range(4000, 4000 + test_episodes))
    candidates = (0.0, 0.01, 0.03, 0.1, 0.3)
    train_scores = {str(w): _score(train, train_size, budget, w) for w in candidates}
    # Tune only on train mazes, without looking at the held-out scenes.
    chosen = max(candidates, key=lambda w: (train_scores[str(w)]["mean_coverage"],
                                            train_scores[str(w)]["goal_rate"]))
    test_scores = {"transferred": _score(test, test_size, budget, chosen),
                   "fixed_default": _score(test, test_size, budget, 0.03),
                   "random": _score(test, test_size, budget, None)}
    return {"config": {"train_seeds": [train[0], train[-1]], "test_seeds": [test[0], test[-1]],
                       "train_size": train_size, "test_size": test_size, "budget": budget},
            "train_scores": train_scores, "selected_gain_weight": chosen,
            "test_scores": test_scores}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-episodes", type=int, default=12)
    ap.add_argument("--test-episodes", type=int, default=30)
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "maze_transfer.json")
    args = ap.parse_args()
    report = evaluate(args.train_episodes, args.test_episodes, args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_gain_weight": report["selected_gain_weight"],
                      "test_scores": report["test_scores"]}, indent=2))


if __name__ == "__main__":
    main()
