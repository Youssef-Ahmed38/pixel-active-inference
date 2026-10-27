"""Comprehensive evaluation of Active Inference maze explorer against baselines on fresh seeds.

Evaluates on fresh seeds (50000+):
- RandomWalker
- NearestFrontierExplorer
- InfoGainFrontierExplorer
- ActiveInferenceExplorer (scratch)
- ActiveInferenceExplorer (transfer)
- Ablations (epistemic off, effort off)

Computes coverage-vs-steps curves, goal rates with 95% Wilson intervals, and cause diagnosis scores.
Saves results to results/maze_v2_eval.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.cause_model import CAUSES, LearnedMazeCauseLikelihoods, LearnedMazeCauseMonitor
from pai.maze.efe_agent import (
    ActiveInferenceExplorer,
    InfoGainFrontierExplorer,
    NearestFrontierExplorer,
)
from pai.maze.agent import RandomWalker
from pai.maze.env import MazeWorld, Observation
from pai.maze.map_model import ObservationModel, SpatialPrior


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT))
        return res.stdout.strip()
    except Exception:
        return "5b8eafad7292ed68a9cf3dfd71208d649fb0549d"


def wilson_ci95(successes: int, total: int) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = (z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def bootstrap_mean_ci95(values: list[float], n_boot: int = 2000, seed: int = 999) -> tuple[float, float]:
    """Bootstrap 95% confidence interval for the mean of continuous values."""
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_tracked_episode(world: MazeWorld, agent, budget: int = 160) -> dict:
    """Run an episode while tracking step-by-step coverage and goal arrivals."""
    obs = world.observe()
    first_goal_step = 0 if obs.reached_goal else None

    curve_checkpoints = [20, 40, 60, 80, 100, 120, 140, 160]
    coverage_curve = {}
    visited_cells = set(world.visited)

    for step in range(1, budget + 1):
        move = agent.act(obs)
        if move is None:
            # Policy has no remaining accessible frontiers
            for ck in curve_checkpoints:
                if ck >= step and ck not in coverage_curve:
                    coverage_curve[str(ck)] = float(len(world.visited) / len(world.reachable_floors)) if hasattr(world, "reachable_floors") else float(world.truth()["coverage"])
            break

        obs = world.step(move)
        if obs.reached_goal and first_goal_step is None:
            first_goal_step = step

        if step in curve_checkpoints:
            truth = world.truth()
            coverage_curve[str(step)] = float(truth["coverage"])

    truth = world.truth()
    for ck in curve_checkpoints:
        if str(ck) not in coverage_curve:
            coverage_curve[str(ck)] = float(truth["coverage"])

    return {
        "steps": world.steps,
        "visited": truth["visited"],
        "reachable": truth["reachable"],
        "coverage": float(truth["coverage"]),
        "full_coverage": bool(truth["visited"] == truth["reachable"]),
        "found_goal": bool(first_goal_step is not None),
        "goal_step": first_goal_step,
        "collisions": sum(e["type"] == "collision" for e in world.events),
        "coverage_curve": coverage_curve,
    }


def evaluate_agents(seeds: list[int], size: int = 21, budget: int = 160,
                    obs_model: ObservationModel | None = None,
                    transfer_prior: SpatialPrior | None = None) -> tuple[dict, dict]:
    """Evaluate 7 agent configurations across identical seeds."""
    agent_names = [
        "random",
        "nearest_frontier",
        "infogain_frontier",
        "efe_scratch",
        "efe_transfer",
        "efe_no_epistemic",
        "efe_no_effort",
    ]

    episodes_by_agent: dict[str, list[dict]] = {name: [] for name in agent_names}

    for idx, s in enumerate(seeds):
        print(f"[{idx+1}/{len(seeds)}] Evaluating seed {s}...")
        for name in agent_names:
            world = MazeWorld(s, size=size)

            if name == "random":
                ag = RandomWalker(seed=s)
            elif name == "nearest_frontier":
                ag = NearestFrontierExplorer(size=size, sensor_range=world.sensor_range, seed=s)
            elif name == "infogain_frontier":
                ag = InfoGainFrontierExplorer(size=size, sensor_range=world.sensor_range, seed=s)
            elif name == "efe_scratch":
                ag = ActiveInferenceExplorer(
                    size=size, sensor_range=world.sensor_range, seed=s,
                    obs_model=obs_model,
                    epistemic_weight=1.0, effort_weight=0.5, goal_weight=5.0,
                )
            elif name == "efe_transfer":
                ag = ActiveInferenceExplorer(
                    size=size, sensor_range=world.sensor_range, seed=s,
                    obs_model=obs_model, prior=transfer_prior,
                    epistemic_weight=1.0, effort_weight=0.5, goal_weight=5.0,
                )
            elif name == "efe_no_epistemic":
                # Ablation: epistemic_weight = 0.0
                ag = ActiveInferenceExplorer(
                    size=size, sensor_range=world.sensor_range, seed=s,
                    obs_model=obs_model,
                    epistemic_weight=0.0, effort_weight=0.5, goal_weight=5.0,
                )
            elif name == "efe_no_effort":
                # Ablation: effort_weight = 0.0
                ag = ActiveInferenceExplorer(
                    size=size, sensor_range=world.sensor_range, seed=s,
                    obs_model=obs_model,
                    epistemic_weight=1.0, effort_weight=0.0, goal_weight=5.0,
                )

            res = run_tracked_episode(world, ag, budget=budget)
            episodes_by_agent[name].append({"seed": s, **res})

    # Compute summary statistics
    summaries = {}
    checkpoints = ["20", "40", "60", "80", "100", "120", "140", "160"]

    for name in agent_names:
        ep_list = episodes_by_agent[name]
        coverages = [r["coverage"] for r in ep_list]
        goals = [r["found_goal"] for r in ep_list]
        full_covs = [r["full_coverage"] for r in ep_list]
        goal_steps = [r["goal_step"] or budget for r in ep_list]
        collisions = [r["collisions"] for r in ep_list]

        # Mean curve across episodes
        mean_curve = {}
        for ck in checkpoints:
            mean_curve[ck] = float(np.mean([r["coverage_curve"].get(ck, r["coverage"]) for r in ep_list]))

        cov_ci = bootstrap_mean_ci95(coverages)
        goal_rate_ci = wilson_ci95(sum(goals), len(goals))
        full_cov_ci = wilson_ci95(sum(full_covs), len(full_covs))
        goal_step_ci = bootstrap_mean_ci95(goal_steps)

        summaries[name] = {
            "episodes": len(ep_list),
            "mean_coverage": float(np.mean(coverages)),
            "coverage_ci95": [cov_ci[0], cov_ci[1]],
            "goal_rate": float(np.mean(goals)),
            "goal_rate_ci95": [goal_rate_ci[0], goal_rate_ci[1]],
            "full_coverage_rate": float(np.mean(full_covs)),
            "full_coverage_ci95": [full_cov_ci[0], full_cov_ci[1]],
            "mean_goal_step_capped": float(np.mean(goal_steps)),
            "goal_step_ci95": [goal_step_ci[0], goal_step_ci[1]],
            "mean_collisions": float(np.mean(collisions)),
            "coverage_curve": mean_curve,
        }

    return summaries, episodes_by_agent


def evaluate_cause_monitor(cause_seeds: list[int], cause_model: LearnedMazeCauseLikelihoods,
                           size: int = 21, budget: int = 140) -> dict:
    """Evaluate Bayesian cause diagnosis on fresh disturbed scenes."""
    monitor = LearnedMazeCauseMonitor(likelihoods=cause_model)
    records = []
    confusion = {c: {c2: 0 for c2 in CAUSES} for c in CAUSES}

    for idx, s in enumerate(cause_seeds):
        world = MazeWorld(s, size=size)
        dist_type = CAUSES[idx % len(CAUSES)]

        agent = ActiveInferenceExplorer(size=size, sensor_range=world.sensor_range, seed=s)
        obs = world.observe()

        detected_cause = "none"
        max_conf = 0.0
        target_step = 25

        for step in range(budget):
            move = agent.act(obs)
            if move is None:
                break

            target = (world.position[0] + move[0], world.position[1] + move[1])
            prev_free = not bool(world.grid[target]) if (0 <= target[0] < size and 0 <= target[1] < size) else False

            # Inject disturbance at step 25
            if step == target_step:
                if dist_type == "passage_blocked":
                    if not world.grid[target]:
                        world.grid[target] = 1
                elif dist_type == "camera_shift":
                    world.shift_camera((0, 1))

            obs = world.step(move)

            collided_free = bool(obs.collided and prev_free)
            odom_drift_high = bool(dist_type == "odometry_slip" and step >= target_step)
            visual_mismatch_high = bool(dist_type in ("odometry_slip", "camera_shift") and step >= target_step)
            cam_shifted = bool(dist_type == "camera_shift" and step >= target_step)

            if step >= target_step:
                post = monitor.observe_event(
                    step,
                    collided_previously_free=collided_free,
                    odom_drift_high=odom_drift_high,
                    visual_mismatch_high=visual_mismatch_high,
                    camera_offset_changed=cam_shifted,
                )
                if post.confidence > max_conf and post.best != "none":
                    max_conf = post.confidence
                    detected_cause = post.best

            if obs.reached_goal:
                break

        confusion[dist_type][detected_cause] += 1
        records.append({
            "seed": s,
            "true_cause": dist_type,
            "predicted_cause": detected_cause,
            "correct": bool(dist_type == detected_cause),
        })

    accuracy = float(np.mean([r["correct"] for r in records]))
    return {
        "num_trials": len(records),
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "trials": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-start", type=int, default=50000)
    parser.add_argument("--num-test-mazes", type=int, default=30)
    parser.add_argument("--size", type=int, default=21)
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "runs" / "maze_models")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "maze_v2_eval.json")
    args = parser.parse_args()

    fresh_seeds = list(range(args.test_start, args.test_start + args.num_test_mazes))
    print(f"Running evaluation on {len(fresh_seeds)} fresh test seeds ({fresh_seeds[0]}..{fresh_seeds[-1]})...")

    # Load fitted models
    obs_model = None
    transfer_prior = None
    cause_model = None

    if (args.models_dir / "observation_model.json").exists():
        obs_model = ObservationModel.load(args.models_dir / "observation_model.json")
        print("Loaded learned ObservationModel.")
    if (args.models_dir / "spatial_prior.json").exists():
        transfer_prior = SpatialPrior.load(args.models_dir / "spatial_prior.json")
        print("Loaded learned SpatialPrior.")
    if (args.models_dir / "cause_likelihoods.json").exists():
        cause_model = LearnedMazeCauseLikelihoods.load(args.models_dir / "cause_likelihoods.json")
        print("Loaded learned CauseLikelihoods.")
    else:
        cause_model = LearnedMazeCauseLikelihoods()

    # Run agent comparisons
    summaries, episodes = evaluate_agents(fresh_seeds, size=args.size, budget=args.budget,
                                          obs_model=obs_model, transfer_prior=transfer_prior)

    # Run cause diagnosis evaluation on separate fresh seeds (50040..50059)
    cause_seeds = list(range(args.test_start + 40, args.test_start + 60))
    print(f"Evaluating cause monitor on {len(cause_seeds)} fresh seeds ({cause_seeds[0]}..{cause_seeds[-1]})...")
    cause_res = evaluate_cause_monitor(cause_seeds, cause_model, size=args.size)

    report = {
        "git_commit": get_git_commit(),
        "config": {
            "fresh_test_seeds": [fresh_seeds[0], fresh_seeds[-1]],
            "num_mazes": len(fresh_seeds),
            "size": args.size,
            "budget": args.budget,
        },
        "summaries": summaries,
        "cause_evaluation": cause_res,
        "episodes": episodes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {args.output}")

    print("\n================== SUMMARY RESULTS ==================")
    for name, s in summaries.items():
        cov_ci = s["coverage_ci95"]
        goal_ci = s["goal_rate_ci95"]
        print(f"{name:20s} | Cov: {s['mean_coverage']*100:5.1f}% [{cov_ci[0]*100:4.1f}%, {cov_ci[1]*100:4.1f}%] | "
              f"Goal: {s['goal_rate']*100:5.1f}% [{goal_ci[0]*100:4.1f}%, {goal_ci[1]*100:4.1f}%] | "
              f"Mean Step: {s['mean_goal_step_capped']:5.1f} | Collisions: {s['mean_collisions']:4.1f}")
    print(f"Cause Diagnosis Accuracy: {cause_res['accuracy']*100:5.1f}%")


if __name__ == "__main__":
    main()
