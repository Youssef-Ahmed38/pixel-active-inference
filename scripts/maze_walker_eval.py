"""Does the maze agent still work when the learned G1 walker carries out its moves?

Same fresh mazes for the grid body (instant, exact moves) and the walker (``pai/maze/walker.py``),
for the route-EFE explorer and nearest-frontier exploration, exact sensing, 160 grid steps.

    python scripts/maze_walker_eval.py --test-start 75000 --num 30 --output results/maze_walker.json
    python scripts/maze_walker_eval.py --maze-style branchy --test-start 77000 --num 30 --output results/maze_walker_branchy.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.efe_agent import NearestFrontierExplorer, RouteEFEExplorer
from pai.maze.env import MazeWorld

AGENTS = ("efe_route", "nearest_frontier")
BODIES = ("grid", "walker")
MAZE_STYLES = {"dfs": {}, "branchy": dict(generator="prim", loop_probability=0.25, rooms=3)}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def run(job: tuple) -> dict:
    seed, agent_name, body, budget, style = job
    import torch
    torch.set_num_threads(1)
    world = MazeWorld(seed, **MAZE_STYLES[style])
    if agent_name == "efe_route":
        agent = RouteEFEExplorer(size=world.size, sensor_range=world.sensor_range, seed=seed,
                                 effort_weight=2.0, switch_margin=0.0)
    else:
        agent = NearestFrontierExplorer(size=world.size, sensor_range=world.sensor_range, seed=seed)
    env = world
    walker = None
    if body == "walker":
        from pai.maze.walker import WalkerMaze
        walker = env = WalkerMaze(world, seed=seed)
    obs = env.observe()
    goal_step = None
    for step in range(1, budget + 1):
        move = agent.act(obs)
        if move is None:
            break
        obs = env.step(move)
        if obs.reached_goal and goal_step is None:
            goal_step = step
            break
        if walker is not None and walker.fell:
            break
    out = {"seed": seed, "agent": agent_name, "body": body, "found_goal": goal_step is not None,
           "goal_step": goal_step, "visited": len(world.visited), "grid_steps": world.steps}
    if walker is not None:
        out.update({"fell": walker.fell, "bumps": walker.bumps, "walk_seconds": walker.t,
                    "failed_walks": sum(e["type"] == "walk_failed" for e in world.events)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", type=int, required=True)
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--maze-style", choices=sorted(MAZE_STYLES), default="dfs")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    seeds = list(range(args.test_start, args.test_start + args.num))
    jobs = [(s, a, b, args.budget, args.maze_style) for s in seeds for a in AGENTS for b in BODIES]
    with Pool(args.workers) as pool:
        episodes = pool.map(run, jobs, chunksize=1)
    summary = {}
    for a in AGENTS:
        for b in BODIES:
            eps = [e for e in episodes if e["agent"] == a and e["body"] == b]
            k = sum(e["found_goal"] for e in eps)
            row = {"episodes": len(eps), "goal_rate": k / len(eps), "goal_rate_ci95": wilson(k, len(eps)),
                   "mean_goal_step_capped": float(np.mean([e["goal_step"] or args.budget for e in eps]))}
            if b == "walker":
                row.update({"falls": sum(e["fell"] for e in eps), "bumps": sum(e["bumps"] for e in eps),
                            "failed_walks": sum(e["failed_walks"] for e in eps),
                            "mean_walk_seconds": float(np.mean([e["walk_seconds"] for e in eps]))})
            summary[f"{a}/{b}"] = row
    same = {a: sum(next(x for x in episodes if (x["seed"], x["agent"], x["body"]) == (s, a, "grid"))["found_goal"]
                   == next(x for x in episodes if (x["seed"], x["agent"], x["body"]) == (s, a, "walker"))["found_goal"]
                   for s in seeds) for a in AGENTS}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    report = {"git_commit": commit, "seeds": [seeds[0], seeds[-1]], "budget": args.budget, "maze_style": args.maze_style,
              "policy": "runs/locomotion/policy_v3/policy.pt", "summary": summary,
              "same_outcome_grid_vs_walker": same, "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    for k, v in summary.items():
        print(k, json.dumps(v))
    print("same outcome grid vs walker:", same, "of", len(seeds))


if __name__ == "__main__":
    main()
