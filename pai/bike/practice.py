"""CPU practice with measured, reversible policy updates and disjoint evaluation."""
from dataclasses import dataclass

import numpy as np

from pai.bike.env import BicycleEnv
from pai.bike.observation_learning import ObservedRidingPolicy
from pai.cognition.contracts import GoalRelation
from pai.cognition.controller import CognitiveController


def riding_trials(policy, seeds):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        speed, roll = float(rng.uniform(2.5, 4.5)), float(rng.uniform(-.1, .1))
        env = BicycleEnv()
        env.reset(seed, speed, roll)
        record = CognitiveController().run(env, GoalRelation("distance_reached", "rider"), policy, budget=500, episode=seed)
        rows.append({"seed": seed, "start_speed": speed, "start_roll": roll,
                     "success": record.success, "distance": env.travelled, "fallen": env.fallen,
                     "seconds": env.data.time})
    # Success dominates progress; early termination distance cannot earn a pass.
    score = (sum(row["success"] for row in rows),
             sum(float(np.clip(row["distance"], 0, 10)) for row in rows) / len(rows))
    return rows, score


@dataclass
class PracticeResult:
    policy: ObservedRidingPolicy
    trials: list
    first_all_successful_attempt: int | None


def practice(initial, seeds, updates=24, seed=0, progress=None):
    if not seeds or updates < 0:
        raise ValueError("nonempty practice seeds and nonnegative updates required")
    rng = np.random.default_rng(seed)
    provenance = f"random-search practice initialized from: {initial.training}"
    best = ObservedRidingPolicy(initial.weights.copy(), provenance)
    rows, best_score = riding_trials(best, seeds)
    trials = [{"attempt": 0, "accepted": True, "score": best_score, "episodes": rows}]
    first = 0 if best_score[0] == len(seeds) else None
    # Mutation scales reflect actuator units and ranges, not expert policy gains.
    scales = np.array([1.0, .10, .03])
    for attempt in range(1, updates + 1):
        candidate = ObservedRidingPolicy(best.weights + rng.normal(size=best.weights.shape) * scales, provenance)
        rows, score = riding_trials(candidate, seeds)
        accepted = score > best_score
        if accepted:
            best, best_score = candidate, score
        trials.append({"attempt": attempt, "accepted": accepted, "score": score, "episodes": rows})
        if first is None and score[0] == len(seeds):
            first = attempt
        if progress is not None:
            progress(attempt, accepted, score)
    return PracticeResult(best, trials, first)
