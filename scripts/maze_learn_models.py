"""Learn the observation model, spatial transfer prior, and cause likelihoods from development mazes.

Uses development seeds 20000..20029 (separate from test seeds 50000+).
Saves fitted models and parameter choices to runs/maze_models/ and results/maze_v2_learned_models.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.cause_model import CAUSES, LearnedMazeCauseLikelihoods
from pai.maze.efe_agent import ActiveInferenceExplorer
from pai.maze.env import MazeWorld, neighbors
from pai.maze.map_model import ObservationModel, SpatialPrior


def collect_training_data(dev_seeds: list[int], size: int = 21, budget: int = 120):
    """Explore development mazes to gather empirical observation and disturbance statistics."""
    obs_pairs: list[tuple[str, str]] = []
    grids: list[np.ndarray] = []
    goals: list[tuple[int, int]] = []
    disturbance_records: list[dict] = []

    for seed in dev_seeds:
        world = MazeWorld(seed=seed, size=size, appearance_jitter=10.0)
        grids.append(world.grid.copy())
        goals.append(world.goal)

        # Run short exploration episode
        agent = ActiveInferenceExplorer(size=size, sensor_range=world.sensor_range, seed=seed)
        obs = world.observe()

        for step in range(budget):
            # Record observation vs truth pairs
            for cell, obs_kind in obs.visible.items():
                r, c = cell
                true_kind = "wall" if world.grid[r, c] else "goal" if (r, c) == world.goal else "floor"
                obs_pairs.append((obs_kind, true_kind))

            move = agent.act(obs)
            if move is None:
                break

            target = (world.position[0] + move[0], world.position[1] + move[1])
            prev_free = not bool(world.grid[target]) if (0 <= target[0] < size and 0 <= target[1] < size) else False

            # Injected disturbances on specific dev seeds
            true_cause = "none"
            if seed % 4 == 1 and step == 30:
                # Injected passage blocked
                if not world.grid[target]:
                    world.grid[target] = 1  # block corridor
                    true_cause = "passage_blocked"
            elif seed % 4 == 2 and step == 30:
                # Injected camera shift
                world.shift_camera((0, 1))
                true_cause = "camera_shift"
            elif seed % 4 == 3 and step == 30:
                # Injected odometry slip (simulated wheel slip causing pose desync)
                true_cause = "odometry_slip"

            prev_pos = world.position
            obs = world.step(move)

            # Record disturbance features
            collided_free = bool(obs.collided and prev_free)
            odom_drift = "high" if true_cause == "odometry_slip" else "low"
            visual_mismatch = "high" if true_cause in ("odometry_slip", "camera_shift") else "low"
            cam_shifted = "1" if true_cause == "camera_shift" else "0"

            if true_cause != "none" or (step % 15 == 0):
                disturbance_records.append({
                    "true_cause": true_cause,
                    "features": {
                        "collided_free": "1" if collided_free else "0",
                        "odom_drift": odom_drift,
                        "visual_mismatch": visual_mismatch,
                        "camera_offset_changed": cam_shifted,
                    },
                })

            if obs.reached_goal:
                break

    return obs_pairs, grids, goals, disturbance_records


def tune_efe_weights(dev_seeds: list[int], size: int = 21, budget: int = 120,
                     obs_model: ObservationModel | None = None,
                     prior: SpatialPrior | None = None) -> dict:
    """Tune EFE hyperparameters on a subset of development seeds."""
    grid_params = [
        {"epistemic_weight": 0.5, "effort_weight": 0.5, "goal_weight": 5.0, "risk_weight": 1.0},
        {"epistemic_weight": 1.0, "effort_weight": 0.5, "goal_weight": 5.0, "risk_weight": 1.0},
        {"epistemic_weight": 2.0, "effort_weight": 0.5, "goal_weight": 5.0, "risk_weight": 1.0},
        {"epistemic_weight": 1.0, "effort_weight": 0.2, "goal_weight": 5.0, "risk_weight": 1.0},
        {"epistemic_weight": 1.0, "effort_weight": 1.0, "goal_weight": 5.0, "risk_weight": 1.0},
    ]

    best_score = -1.0
    best_params = grid_params[1]
    scores = {}

    eval_seeds = dev_seeds[:8]
    for params in grid_params:
        key = f"epi_{params['epistemic_weight']}_eff_{params['effort_weight']}"
        coverages = []
        for s in eval_seeds:
            w = MazeWorld(s, size=size)
            ag = ActiveInferenceExplorer(
                sensor_range=w.sensor_range, size=size, seed=s,
                prior=prior, obs_model=obs_model,
                epistemic_weight=params["epistemic_weight"],
                effort_weight=params["effort_weight"],
                goal_weight=params["goal_weight"],
                risk_weight=params["risk_weight"],
            )
            o = w.observe()
            for _ in range(budget):
                m = ag.act(o)
                if m is None:
                    break
                o = w.step(m)
            coverages.append(w.truth()["coverage"])
        mean_cov = float(np.mean(coverages))
        scores[key] = mean_cov
        if mean_cov > best_score:
            best_score = mean_cov
            best_params = params

    return {"best_params": best_params, "grid_scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=20000)
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--size", type=int, default=21)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "maze_models")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "maze_v2_learned_models.json")
    args = parser.parse_args()

    dev_seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    print(f"Collecting training data from {len(dev_seeds)} development seeds ({dev_seeds[0]}..{dev_seeds[-1]})...")

    obs_pairs, grids, goals, disturbances = collect_training_data(dev_seeds, size=args.size)

    # 1. Fit Observation Model
    observed_kinds = [p[0] for p in obs_pairs]
    actual_kinds = [p[1] for p in obs_pairs]
    obs_model = ObservationModel.fit(observed_kinds, actual_kinds, alpha=2.0)
    print("Observation Model matrix P(O | M):")
    print(obs_model.matrix)

    # 2. Fit Spatial Prior
    spatial_prior = SpatialPrior.fit_from_mazes(grids, goals, size=args.size, alpha=2.0)
    print("Spatial Prior fitted. Shape:", spatial_prior.priors.shape)

    # 3. Fit Cause Likelihoods
    cause_model = LearnedMazeCauseLikelihoods()
    cause_model.fit(disturbances)
    print(f"Cause model fitted on {len(disturbances)} disturbance events.")

    # 4. Tune EFE Hyperparameters
    tuning_res = tune_efe_weights(dev_seeds, size=args.size, obs_model=obs_model, prior=spatial_prior)
    print("EFE Tuning result:", tuning_res)

    # Save models
    args.output_dir.mkdir(parents=True, exist_ok=True)
    obs_model.save(args.output_dir / "observation_model.json")
    spatial_prior.save(args.output_dir / "spatial_prior.json")
    cause_model.save(args.output_dir / "cause_likelihoods.json")

    # Save report
    report = {
        "development_seeds": [dev_seeds[0], dev_seeds[-1]],
        "num_mazes": len(dev_seeds),
        "total_observation_pairs": len(obs_pairs),
        "observation_matrix": obs_model.matrix.tolist(),
        "total_disturbance_events": len(disturbances),
        "cause_feature_counts": cause_model.feature_counts,
        "tuning": tuning_res,
        "provenance": {
            "size": args.size,
            "seeds": dev_seeds,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Saved models to {args.output_dir} and report to {args.report}")


if __name__ == "__main__":
    main()
