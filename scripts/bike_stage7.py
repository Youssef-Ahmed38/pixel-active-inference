"""Measure the initial bicycle model and classical balance baseline on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pai.bike import BicycleEnv, BalanceController
from pai.cognition.contracts import GoalRelation, SkillCommand
from pai.cognition.controller import CognitiveController


def evaluate(seeds, distance=10, gain=2, rate_gain=.8):
    rows = []
    for seed in seeds:
        for condition in ("uncontrolled", "classical"):
            env = BicycleEnv(distance=distance)
            env.reset(seed)
            policy = (BalanceController(gain, rate_gain) if condition == "classical" else
                      lambda state, adapter: None if adapter.done else SkillCommand("ride", parameters=(0, 0, 0)))
            record = CognitiveController().run(env, GoalRelation("distance_reached", "rider"), policy,
                                               budget=500, episode=seed)
            rows.append({"seed": seed, "condition": condition, "success": record.success,
                         "distance": env.travelled, "seconds": env.data.time, "fallen": env.fallen,
                         "final_roll": float(env.sensation()[0]), "events": env.events})
    return {"scope": "free MuJoCo bicycle, wheel drive, steering and simplified torso rider; privileged sensors",
            "gate": {"distance_m": distance, "fall_roll_radians": .7, "seeds": list(seeds)},
            "summary": {condition: {"episodes": sum(row["condition"] == condition for row in rows),
                "successes": sum(row["success"] for row in rows if row["condition"] == condition)}
                for condition in ("uncontrolled", "classical")},
            "episodes": rows, "remaining": ["video demonstrations", "inverse dynamics observation learning",
                    "practice learning", "Stage 2 transfer and no-observation comparisons"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--gain", type=float, default=2)
    p.add_argument("--rate-gain", type=float, default=.8)
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_stage7.json")
    args = p.parse_args()
    result = evaluate(args.seeds, gain=args.gain, rate_gain=args.rate_gain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
