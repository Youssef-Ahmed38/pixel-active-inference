"""Maze v3: does curiosity help when the map is genuinely uncertain?

Two sensing regimes on the same mazes:
- exact: ``MazeWorld.observe`` (one look settles a cell; curiosity has little to do);
- noisy: ``NoisySensorMaze`` (labels confused at every look, goal hard to recognise).

The sensor model every map-based agent uses in the noisy regime is fitted from training mazes
(seeds 1000-1019, random walk, true labels only for fitting), never the simulator's matrix. All map
agents share that model, so differences come from the decision rule only.

Usage:
    python scripts/maze_v3_eval.py --test-start 20000 --num-test-mazes 30 --output data/maze_dev/v3.json
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

from pai.maze.agent import RandomWalker
from pai.maze.efe_agent import (
    ActiveInferenceExplorer,
    InfoGainFrontierExplorer,
    NearestFrontierExplorer,
    RouteEFEExplorer,
)
from pai.maze.env import DIRECTIONS, MazeWorld
from pai.maze.map_model import ObservationModel
from pai.maze.noisy import NoisySensorMaze

AGENTS = ["random", "nearest_frontier", "infogain_frontier", "efe_v2", "efe_route",
          "efe_route_no_epistemic", "efe_route_no_goal_belief"]
REGIMES = ["exact", "noisy"]


def fit_sensor_model(train_seeds: list[int], size: int, steps: int = 200) -> ObservationModel:
    """Fit P(seen | true) from random walks in training mazes."""
    seen, actual = [], []
    for s in train_seeds:
        world = MazeWorld(s, size=size)
        noisy = NoisySensorMaze(world, seed=s)
        rng = np.random.default_rng(s)
        obs = noisy.observe()
        for _ in range(steps):
            for cell, kind in obs.visible.items():
                seen.append(kind)
                actual.append("wall" if world.grid[cell] else "goal" if cell == world.goal else "floor")
            obs = noisy.step(DIRECTIONS[int(rng.integers(4))])
    return ObservationModel.fit(seen, actual)


def make_agent(name: str, world: MazeWorld, seed: int, obs_model: ObservationModel | None, params: dict):
    common = dict(size=world.size, sensor_range=world.sensor_range, seed=seed)
    if name == "random":
        return RandomWalker(seed=seed)
    if name == "nearest_frontier":
        return NearestFrontierExplorer(obs_model=obs_model, **common)
    if name == "infogain_frontier":
        return InfoGainFrontierExplorer(obs_model=obs_model, **common)
    if name == "efe_v2":
        return ActiveInferenceExplorer(obs_model=obs_model, epistemic_weight=1.0, effort_weight=0.5,
                                       goal_weight=5.0, **common)
    route = dict(epistemic_weight=params["epistemic_weight"], effort_weight=params["effort_weight"],
                 goal_weight=params["goal_weight"], switch_margin=params["switch_margin"])
    if name == "efe_route_no_epistemic":
        route["epistemic_weight"] = 0.0
    if name == "efe_route_no_goal_belief":
        route["goal_weight"] = 0.0
    return RouteEFEExplorer(obs_model=obs_model, **route, **common)


def run_episode(job: tuple) -> dict:
    seed, regime, name, size, budget, matrix, params = job
    world = MazeWorld(seed, size=size)
    env = NoisySensorMaze(world, seed=seed) if regime == "noisy" else world
    obs_model = ObservationModel(matrix=np.array(matrix)) if regime == "noisy" else None
    agent = make_agent(name, world, seed, obs_model, params)
    obs = env.observe()
    seen = set(world.observe().visible)
    goal_step = 0 if obs.reached_goal else None
    for step in range(1, budget + 1):
        move = agent.act(obs)
        if move is None:
            break
        obs = env.step(move)
        seen |= set(world.observe().visible)
        if obs.reached_goal and goal_step is None:
            goal_step = step
    truth = world.truth()
    reachable = world.distances_from(world.start)
    return {"seen_coverage": len(seen & set(reachable)) / len(reachable),"seed": seed, "regime": regime, "agent": name, "coverage": float(truth["coverage"]),
            "found_goal": goal_step is not None, "goal_step": goal_step,
            "collisions": sum(e["type"] == "collision" for e in world.events)}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def summarise(episodes: list[dict], budget: int) -> dict:
    out = {}
    for regime in REGIMES:
        for name in AGENTS:
            eps = [e for e in episodes if e["regime"] == regime and e["agent"] == name]
            k = sum(e["found_goal"] for e in eps)
            out[f"{regime}/{name}"] = {
                "episodes": len(eps),
                "mean_coverage": float(np.mean([e["coverage"] for e in eps])),
                "mean_seen_coverage": float(np.mean([e["seen_coverage"] for e in eps])),
                "goal_rate": k / len(eps),
                "goal_rate_ci95": wilson(k, len(eps)),
                "mean_goal_step_capped": float(np.mean([e["goal_step"] or budget for e in eps])),
                "mean_collisions": float(np.mean([e["collisions"] for e in eps])),
            }
    return out


def paired_goal_comparison(episodes: list[dict], regime: str, a: str, b: str) -> dict:
    """Seeds where exactly one of two agents found the goal (sign test counts)."""
    found = {(e["agent"], e["seed"]): e["found_goal"] for e in episodes if e["regime"] == regime}
    seeds = sorted({e["seed"] for e in episodes})
    only_a = sum(found[(a, s)] and not found[(b, s)] for s in seeds)
    only_b = sum(found[(b, s)] and not found[(a, s)] for s in seeds)
    n = only_a + only_b
    # two-sided exact binomial sign test
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n if n else 1.0
    return {"a": a, "b": b, "only_a": only_a, "only_b": only_b, "sign_test_p": min(1.0, 2 * tail)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-start", type=int, required=True)
    parser.add_argument("--num-test-mazes", type=int, default=30)
    parser.add_argument("--size", type=int, default=21)
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--epistemic-weight", type=float, default=1.0)
    parser.add_argument("--effort-weight", type=float, default=0.5)
    parser.add_argument("--goal-weight", type=float, default=5.0)
    parser.add_argument("--switch-margin", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    params = {"epistemic_weight": args.epistemic_weight, "effort_weight": args.effort_weight,
              "goal_weight": args.goal_weight, "switch_margin": args.switch_margin}
    sensor = fit_sensor_model(list(range(1000, 1020)), args.size)
    seeds = list(range(args.test_start, args.test_start + args.num_test_mazes))
    jobs = [(s, r, a, args.size, args.budget, sensor.matrix.tolist(), params)
            for s in seeds for r in REGIMES for a in AGENTS]
    with Pool(args.workers) as pool:
        episodes = pool.map(run_episode, jobs, chunksize=4)

    summary = summarise(episodes, args.budget)
    comparisons = [paired_goal_comparison(episodes, r, "efe_route", b)
                   for r in REGIMES for b in ("nearest_frontier", "infogain_frontier",
                                              "efe_route_no_epistemic", "efe_route_no_goal_belief")]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    report = {"git_commit": commit, "seeds": [seeds[0], seeds[-1]], "budget": args.budget,
              "params": params, "fitted_sensor_model": sensor.matrix.round(4).tolist(),
              "summary": summary, "paired_goal_comparisons": comparisons, "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))

    for key, s in summary.items():
        lo, hi = s["goal_rate_ci95"]
        print(f"{key:34s} visited {s['mean_coverage']*100:5.1f}%  seen {s['mean_seen_coverage']*100:5.1f}%  goal {s['goal_rate']*100:5.1f}% "
              f"[{lo*100:4.1f}, {hi*100:4.1f}]  goal step {s['mean_goal_step_capped']:5.1f}  "
              f"collisions {s['mean_collisions']:4.1f}")
    for c in comparisons:
        print(f"{c}")


if __name__ == "__main__":
    main()
