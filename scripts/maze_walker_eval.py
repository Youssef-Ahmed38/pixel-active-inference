"""Does the maze agent still work when the learned G1 walker carries out its moves?

Same fresh mazes for the grid body (instant, exact moves) and the walker (``pai/maze/walker.py``),
for the route-EFE explorer and nearest-frontier exploration, exact sensing, 160 grid steps.

    python scripts/maze_walker_eval.py --test-start 75000 --num 30 --output results/maze_walker.json
    python scripts/maze_walker_eval.py --maze-style branchy --test-start 77000 --num 30 --output results/maze_walker_branchy.json
    python scripts/maze_walker_eval.py --visual-walls ...   # the first version (walls not colliders)
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
from pai.maze.map_model import ObservationModel
from pai.maze.noisy import NoisySensorMaze

AGENTS = ("efe_route", "nearest_frontier")
BODIES = ("grid", "walker")
MAZE_STYLES = {"dfs": {}, "branchy": dict(generator="prim", loop_probability=0.25, rooms=3)}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def run(job: tuple) -> dict:
    seed, agent_name, body, budget, style, solid, regime, matrix, repeat_weight, stall_s, policy = job
    import torch
    torch.set_num_threads(1)
    world = MazeWorld(seed, **MAZE_STYLES[style])
    obs_model = ObservationModel(matrix=np.array(matrix)) if matrix is not None else None
    if agent_name == "efe_route":
        agent = RouteEFEExplorer(size=world.size, sensor_range=world.sensor_range, seed=seed,
                                 effort_weight=2.0, switch_margin=0.0, obs_model=obs_model)
        agent.map.repeat_weight = repeat_weight
    else:
        agent = NearestFrontierExplorer(size=world.size, sensor_range=world.sensor_range, seed=seed,
                                        obs_model=obs_model)
    env = world
    walker = None
    if body == "walker":
        from pai.maze.walker import WalkerMaze
        walker = env = WalkerMaze(world, seed=seed, solid=solid, stall_s=stall_s, policy=Path(policy))
    if regime != "exact":
        # the noisy sensor sits on top of the body: the walker moves, the sensor reads and touches
        env = NoisySensorMaze(env, seed=seed, persistent=regime == "persistent")
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
           "goal_step": goal_step, "visited": len(world.visited), "grid_steps": world.steps,
           "regime": regime, "collisions": sum(e["type"] == "collision" for e in world.events)}
    if walker is not None:
        out.update({"fell": walker.fell, "bumps": walker.bumps, "walk_seconds": walker.t,
                    "failed_walks": sum(e["type"] == "walk_failed" for e in world.events),
                    "felt_blocked_open": sum(e["type"] == "felt_blocked" for e in world.events)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", type=int, required=True)
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--maze-style", choices=sorted(MAZE_STYLES), default="dfs")
    ap.add_argument("--regime", choices=("exact", "noisy", "persistent"), default="exact",
                    help="sensing; noisy/persistent use a sensor model the agent learns from its own practice")
    ap.add_argument("--stall-s", type=float, default=1.0, help="seconds without progress that count as blocked")
    ap.add_argument("--visual-walls", action="store_true",
                    help="walls the body passes through (the first version; results before solid walls)")
    ap.add_argument("--policy", default="runs/locomotion/policy_v3/policy.pt", help="walking policy")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    seeds = list(range(args.test_start, args.test_start + args.num))
    matrix, repeat_weight, sensor_info = None, 1.0, {"source": "none (exact sensing)"}
    if args.regime != "exact":
        from pai.maze.self_calibration import self_calibrated_sensor
        style = MAZE_STYLES[args.maze_style]
        # practice mazes 1000-1019 of the same style, walked by the grid body (the sensor model
        # describes the sensor, not the body), exactly as in scripts/maze_v3_eval.py
        sensor, sensor_info = self_calibrated_sensor(
            list(range(1000, 1020)), 21,
            make_env=lambda s: NoisySensorMaze(MazeWorld(s, **style), seed=s, persistent=args.regime == "persistent"),
            make_agent=lambda s: RouteEFEExplorer(size=21, sensor_range=3, seed=s, effort_weight=2.0,
                                                  switch_margin=0.0))
        matrix, repeat_weight = sensor.matrix.tolist(), sensor_info["repeat_weight"]
    jobs = [(s, a, b, args.budget, args.maze_style, not args.visual_walls, args.regime, matrix, repeat_weight, args.stall_s, args.policy) for s in seeds for a in AGENTS for b in BODIES]
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
              "walls": "visual" if args.visual_walls else "solid", "regime": args.regime, "stall_s": args.stall_s,
              "sensor": {"matrix": matrix, "repeat_weight": repeat_weight, "fit": sensor_info},
              "policy": args.policy, "summary": summary,
              "same_outcome_grid_vs_walker": same, "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    for k, v in summary.items():
        print(k, json.dumps(v))
    print("same outcome grid vs walker:", same, "of", len(seeds))


if __name__ == "__main__":
    main()
