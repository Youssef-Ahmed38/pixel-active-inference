"""Compare bike practice from observed-motion initialization against scratch."""
import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from pai.bike.observation_learning import ObservedRidingPolicy
from pai.bike.practice import practice, riding_trials


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--updates", type=int, default=24)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/bike/observed_policy.json")
    p.add_argument("--artifacts", type=Path, default=ROOT / "runs/bike/practice")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--practice-seeds", nargs="+", type=int, default=[100, 101, 102])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(110, 120)))
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_practice.json")
    args = p.parse_args()
    if set(args.practice_seeds) & set(args.eval_seeds):
        p.error("practice and evaluation seeds must be disjoint")
    observed = ObservedRidingPolicy.load(args.checkpoint)
    report = {"scope": "random-search CPU practice; diverse moving starts; known model and motion observations",
              "updates_per_condition": args.updates, "mutation_seed": args.seed, "conditions": {},
              "source_checkpoint": str(args.checkpoint),
              "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "practice_seeds": args.practice_seeds, "evaluation_seeds": args.eval_seeds,
              "limitations": ["one optimizer seed is not a statistical transfer proof",
                  "no Stage 2 transfer comparison", "student does not infer motion from video pixels"]}
    for condition, initial in (("observed_motion", observed), ("scratch", ObservedRidingPolicy(np.zeros((7, 3)), "zero policy; no observation training"))):
        baseline, _ = riding_trials(initial, args.eval_seeds)
        result = practice(initial, args.practice_seeds, args.updates, args.seed,
            lambda attempt, accepted, score: print(f"{condition} attempt {attempt}: {score}, accepted={accepted}", flush=True))
        evaluation, _ = riding_trials(result.policy, args.eval_seeds)
        result.policy.save(args.artifacts / f"practiced_{condition}.json")
        report["conditions"][condition] = {"practice": result.trials,
            "first_all_successful_practice_attempt": result.first_all_successful_attempt,
            "attempt_definition": "one candidate evaluated on all practice seeds; attempt 0 is the initial policy",
            "baseline_evaluation": baseline, "practiced_evaluation": evaluation,
            "baseline_successes": sum(row["success"] for row in baseline),
            "practiced_successes": sum(row["success"] for row in evaluation)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: row[key] for key in ("first_all_successful_practice_attempt", "baseline_successes", "practiced_successes")}
                      for name, row in report["conditions"].items()}, indent=2))
