"""The maze agent sees through the G1's head camera while the learned walker walks (Part C, step 1).

Practice (mazes 1000-1019): the agent walks with the camera and learns its camera's sensor model and
the repeat-look weight from its own looks and body labels (``pai/maze/self_calibration.py``), no
simulator labels. Evaluation on fresh mazes, the same mazes for every arm:

- ``grid``: the walker with the old grid sensor (exact labels), as in ``scripts/maze_walker_eval.py``;
- ``camera``: the walker with the head camera (``pai/maze/ego_camera.py``) and the learned sensor model.

    python scripts/maze_camera_eval.py --test-start 20000 --num 10 --output data/maze_dev/camera_dev.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import comb
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.efe_agent import NearestFrontierExplorer, RouteEFEExplorer
from pai.maze.env import MazeWorld
from pai.maze.map_model import ObservationModel

STYLES = {"dfs": {}, "branchy": dict(generator="prim", loop_probability=0.25, rooms=3)}
ARMS = [("grid", "efe_route"), ("grid", "nearest_frontier"), ("camera", "efe_route"),
        ("camera", "nearest_frontier"), ("camera", "efe_route_count_repeats")]


def make_world(seed, style):
    return MazeWorld(seed, **STYLES[style])


def make_body(world, sensor, seed, policy, odometry=False):
    from pai.maze.walker import WalkerMaze
    walker = WalkerMaze(world, policy=Path(policy), seed=seed, wall_height=1.4,
                        odometry=odometry and sensor == "camera")
    if sensor == "grid":
        return walker
    from pai.maze.ego_camera import EgoCameraMaze
    return EgoCameraMaze(walker)


def make_agent(name, world, seed, sensor_model, repeat_weight):
    common = dict(size=world.size, sensor_range=world.sensor_range, seed=seed, obs_model=sensor_model)
    if name == "nearest_frontier":
        agent = NearestFrontierExplorer(**common)
    else:
        # long-range camera: visit seen cells only when they may hold the goal (or bring information)
        agent = RouteEFEExplorer(effort_weight=2.0, switch_margin=0.0,
                                 min_goal_belief=0.005 if sensor_model is not None else None, **common)
    if name != "efe_route_count_repeats":
        agent.map.repeat_weight = repeat_weight
    return agent


def practice(seed, style, policy, odometry=False):
    """One practice maze for calibration, walked with the camera (run in a worker)."""
    import torch
    torch.set_num_threads(1)
    from pai.maze.self_calibration import collect_practice
    held = {}

    def make_env(s):
        held["env"] = make_body(make_world(s, style), "camera", s, policy, odometry)
        return held["env"]

    def make_agent(s):
        agent = RouteEFEExplorer(size=21, sensor_range=3, seed=s, effort_weight=2.0, switch_margin=0.0)
        held["env"].map_ref = agent.map            # localization against the agent's own map
        return agent

    return collect_practice([seed], make_env=make_env, make_agent=make_agent)


def run(job):
    seed, sensor, name, style, budget, matrix, repeat_weight, policy, odometry = job
    import torch
    torch.set_num_threads(1)
    world = make_world(seed, style)
    body = make_body(world, sensor, seed, policy, odometry)
    model = ObservationModel(matrix=np.array(matrix)) if (sensor == "camera" and matrix) else None
    agent = make_agent(name, world, seed, model, repeat_weight if sensor == "camera" else 1.0)
    if sensor == "camera":
        body.map_ref = agent.map
    walker = getattr(body, "walker", body)
    obs = body.observe()
    goal_step = None
    for step in range(1, budget + 1):
        move = agent.act(obs)
        if move is None:
            break
        obs = body.step(move)
        if obs.reached_goal:
            goal_step = step
            break
        if walker.fell:
            break
    if hasattr(body, "close"):
        body.close()
    drift = float(np.linalg.norm(walker.est_xy - walker.true_xy())) if walker.odometry else 0.0
    return {"seed": seed, "sensor": sensor, "agent": name, "found_goal": goal_step is not None, "final_drift_m": drift,
            "goal_step": goal_step, "visited": len(world.visited), "fell": walker.fell,
            "bumps": walker.bumps, "walk_seconds": walker.t}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def sign_test(a, b):
    n = a + b
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-start", type=int, required=True)
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--style", choices=list(STYLES), default="branchy")
    ap.add_argument("--policy", default="runs/locomotion/policy_v3/policy.pt")
    ap.add_argument("--practice", type=int, default=20, help="practice mazes 1000.. for calibration")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--odometry", action="store_true",
                    help="camera arms use the robot's own odometry (with map-based pose correction)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from pai.maze.self_calibration import dawid_skene, repeat_look_weight, weighted_counts

    with Pool(args.workers) as pool:
        parts = pool.starmap(practice, [(s, args.style, args.policy, args.odometry)
                                        for s in range(1000, 1000 + args.practice)])
    views, anchors = {}, {}
    for v, a in parts:
        views.update(v)
        anchors.update(a)
    weight, stats = repeat_look_weight(views)
    sensor, prior = dawid_skene(weighted_counts(views, weight), anchors)
    calibration = {"repeat_weight": weight, **stats, "matrix": sensor.matrix.round(4).tolist(),
                   "class_frequencies": prior.round(4).tolist(), "body_labels": len(anchors),
                   "looks": int(sum(len(x) for x in views.values()))}
    print("calibration:", json.dumps(calibration))

    seeds = list(range(args.test_start, args.test_start + args.num))
    jobs = [(s, sensor_kind, name, args.style, args.budget, sensor.matrix.tolist(), weight, args.policy,
             args.odometry)
            for s in seeds for sensor_kind, name in ARMS]
    with Pool(args.workers) as pool:
        episodes = pool.map(run, jobs, chunksize=1)
    found = {(e["sensor"], e["agent"], e["seed"]): e["found_goal"] for e in episodes}
    summary = {}
    for sensor_kind, name in ARMS:
        eps = [e for e in episodes if (e["sensor"], e["agent"]) == (sensor_kind, name)]
        k = sum(e["found_goal"] for e in eps)
        summary[f"{sensor_kind}/{name}"] = {
            "episodes": len(eps), "goals": k, "goal_rate_ci95": wilson(k, len(eps)),
            "falls": sum(e["fell"] for e in eps), "bumps": sum(e["bumps"] for e in eps),
            "mean_walk_seconds": float(np.mean([e["walk_seconds"] for e in eps])),
            "median_final_drift_m": float(np.median([e["final_drift_m"] for e in eps]))}
    paired = {}
    for a, b in ((("camera", "efe_route"), ("camera", "nearest_frontier")),
                 (("camera", "efe_route"), ("camera", "efe_route_count_repeats")),
                 (("camera", "efe_route"), ("grid", "efe_route"))):
        only_a = sum(found[(*a, s)] and not found[(*b, s)] for s in seeds)
        only_b = sum(found[(*b, s)] and not found[(*a, s)] for s in seeds)
        paired[f"{'/'.join(a)} vs {'/'.join(b)}"] = {"only_a": only_a, "only_b": only_b,
                                                     "sign_test_p": sign_test(only_a, only_b)}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip()
    report = {"git_commit": commit, "seeds": [seeds[0], seeds[-1]], "style": args.style, "odometry": args.odometry,
              "budget": args.budget, "policy": args.policy, "calibration": calibration,
              "summary": summary, "paired": paired, "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    for k, v in summary.items():
        print(k, json.dumps(v))
    print(json.dumps(paired))


if __name__ == "__main__":
    main()
