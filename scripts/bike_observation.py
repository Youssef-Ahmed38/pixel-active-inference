"""Record the bicycle expert, infer actions, fit a student and evaluate fresh seeds."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pai.bike import BicycleEnv, BalanceController
from pai.bike.observation_learning import collect_demonstrations, ObservedRidingPolicy
from pai.cognition.controller import CognitiveController
from pai.cognition.contracts import GoalRelation, SkillCommand


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--eval-seeds", nargs="+", type=int, default=list(range(20, 30)))
    p.add_argument("--video", type=Path)
    p.add_argument("--diverse-starts", action="store_true")
    p.add_argument("--artifacts", type=Path, default=ROOT / "runs/bike")
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_observation.json")
    args = p.parse_args()
    if set(args.train_seeds) & set(args.eval_seeds):
        p.error("training and evaluation seeds must be disjoint")
    dataset = args.artifacts / "expert_states.npz"
    demonstration = collect_demonstrations(args.train_seeds, dataset, args.video, args.diverse_starts)
    student = ObservedRidingPolicy.fit(dataset)
    student.save(args.artifacts / "observed_policy.json")
    rows = []
    for seed in args.eval_seeds:
        for condition in ("uncontrolled", "classical", "observed_student"):
            env = BicycleEnv()
            if args.diverse_starts:
                rng = np.random.default_rng(seed)
                env.reset(seed, speed=float(rng.uniform(2.5, 4.5)), roll=float(rng.uniform(-.1, .1)))
            else:
                env.reset(seed)
            policy = (student if condition == "observed_student" else BalanceController() if condition == "classical"
                else lambda state, adapter: None if adapter.done else SkillCommand("ride", parameters=(0, 0, 0)))
            record = CognitiveController().run(env, GoalRelation("distance_reached", "rider"), policy, budget=500, episode=seed)
            rows.append({"seed": seed, "condition": condition, "success": record.success,
                         "distance": env.travelled, "fallen": env.fallen, "seconds": env.data.time})
    report = {"scope": "motion-state observation with known MuJoCo inverse dynamics; not learned video perception",
              "demonstrations": demonstration, "episodes": rows,
              "diverse_starts": args.diverse_starts,
              "summary": {condition: {"episodes": len(args.eval_seeds),
                   "successes": sum(r["success"] for r in rows if r["condition"] == condition)}
                   for condition in ("uncontrolled", "classical", "observed_student")},
              "remaining": ["motion extraction from videos", "practice learning", "Stage 2 transfer comparison"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"inference": demonstration["inverse_action_mae"], "evaluation": report["summary"]}, indent=2))
