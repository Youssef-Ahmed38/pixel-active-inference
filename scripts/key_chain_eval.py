"""Evaluate the maze -> drawer -> carried key -> locked door task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.discovery.mock import MockWorld, Scenario  # noqa: E402
from pai.maze.env import MazeWorld  # noqa: E402
from pai.maze.key_chain import run_key_chain  # noqa: E402


def evaluate(seeds=range(6), *, physics=False, size=15,
             move_budget=500, fixture_budget=120, n_keys=1,
             type_name="hd_lever_pull_right"):
    rows = []
    for seed in seeds:
        maze = MazeWorld(seed=seed, size=size)
        if physics:
            from pai.discovery.physics import PhysicsWorld
            fixture = PhysicsWorld(type_name, "robotiq_2f85", seed=seed,
                                   lock_state="both", key_place="drawer", n_keys=n_keys)
        else:
            fixture = MockWorld(Scenario(type_name, "both", "drawer",
                                         n_keys=n_keys, seed=seed, p_fail=0.0, pen=False))
        result = run_key_chain(maze, fixture, seed=seed, move_budget=move_budget,
                               fixture_budget=fixture_budget)
        rows.append({"seed": seed, **vars(result)})
    return {"config": {"physics": physics, "size": size, "move_budget": move_budget,
                       "fixture_budget": fixture_budget, "keys": n_keys,
                       "type_name": type_name, "lock_state": "both", "key_place": "drawer"},
            "summary": {"n": len(rows), "passed": sum(row["success"] for row in rows),
                        "success_rate": sum(row["success"] for row in rows) / max(len(rows), 1)},
            "episodes": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--keys", type=int, default=1)
    parser.add_argument("--door-type", default="hd_lever_pull_right")
    parser.add_argument("--size", type=int, default=15)
    parser.add_argument("--move-budget", type=int, default=500)
    parser.add_argument("--fixture-budget", type=int, default=120)
    parser.add_argument("--out", type=Path, default=ROOT / "results/key_chain_eval.json")
    args = parser.parse_args()
    report = evaluate(range(args.seed_start, args.seed_start + args.seeds),
                      physics=args.physics, size=args.size,
                      move_budget=args.move_budget, fixture_budget=args.fixture_budget,
                      n_keys=args.keys, type_name=args.door_type)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
