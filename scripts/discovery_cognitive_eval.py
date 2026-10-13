"""Evaluate shared cognitive recording of door and drawer discovery skills."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.cognition.discovery import FeatureGoalWorld, run_discovery_cognitive  # noqa: E402
from pai.cognition.memory import CognitiveMemory  # noqa: E402
from pai.discovery.mock import MockWorld, sample_scenario  # noqa: E402


def evaluate(seeds=range(4), budget=300, physics=False,
             type_name="hd_knob_pull_left", embodiment="robotiq_2f85",
             lock_state="unlocked", key_place="table", tasks=("door", "drawer"),
             contact_repair=True, object_trials=None):
    rows = []
    if object_trials is None:
        object_trials = physics
    memory = CognitiveMemory()
    for seed in seeds:
        scenario = None if physics else sample_scenario(np.random.default_rng(seed), split="test",
                                                        lock_state=lock_state if lock_state != "unlocked" else "latch",
                                                        key_place=key_place, p_fail=0.0)
        for task in tasks:
            if physics:
                from pai.discovery.physics import PhysicsWorld
                base = PhysicsWorld(type_name, embodiment, seed=seed,
                                    lock_state=lock_state, key_place=key_place, n_keys=2,
                                    contact_repair=contact_repair)
            else:
                base = MockWorld(scenario)
            feature = f"part:{base.pid[task]}.open"
            world = FeatureGoalWorld(base, feature)
            record, log = run_discovery_cognitive(world, seed=seed, budget_actions=budget,
                                                   memory=memory, object_trials=object_trials)
            truth = base.truth() if physics else None
            rows.append({"seed": seed, "task": task, "success": record.success,
                         "actions": record.steps, "reason": log.stopped_reason,
                         "recorded_outcomes": len(record.outcomes),
                         "stalled_skills": sum(o.stalled for o in record.outcomes),
                         "action_history": [
                             {"step": i + 1, "action": list(action), "mode": log.modes[i],
                              "executed": bool(outcome.executed), "stalled": bool(outcome.stalled),
                              "notes": outcome.notes,
                              "changed": {key: list(pair) for key, pair in outcome.changed.items()},
                              "hand_before": log.states[i].get("body:hand")}
                             for i, (action, outcome) in enumerate(zip(log.actions, log.outcomes))
                         ] if physics else [],
                         "failed_inserts": [event for event in record.events
                                            if event.get("action", [None])[0] == "insert"
                                            and event.get("stalled")],
                         "physical_state": ({"key_locked": truth.get("key_locked"),
                                             "deadbolt_locked": truth.get("deadbolt_locked"),
                                             "opened": truth.get("opened")}
                                            if truth is not None else None)})
    return {"config": {"physics": physics, "type_name": type_name,
                       "embodiment": embodiment, "lock_state": lock_state,
                       "key_place": key_place, "tasks": list(tasks),
                       "contact_repair": contact_repair,
                       "object_trials": object_trials},
            "summary": {task: {"n": sum(r["task"] == task for r in rows),
                              "success_rate": sum(r["success"] for r in rows if r["task"] == task)
                              / max(1, sum(r["task"] == task for r in rows))}
                        for task in tasks}, "episodes": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-list", nargs="+", type=int,
                    help="explicit seed IDs for a paired held-out run")
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--type-name", default="hd_knob_pull_left")
    ap.add_argument("--embodiment", default="robotiq_2f85")
    ap.add_argument("--lock-state", choices=("unlocked", "deadbolt", "key", "both"),
                    default="unlocked")
    ap.add_argument("--key-place", choices=("table", "shelf", "drawer", "other_room"),
                    default="table")
    ap.add_argument("--tasks", nargs="+", choices=("door", "drawer"),
                    default=["door", "drawer"])
    ap.add_argument("--no-contact-repair", action="store_true")
    ap.add_argument("--no-object-trials", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/discovery_cognitive_eval.json")
    args = ap.parse_args()
    result = evaluate(args.seed_list if args.seed_list is not None else range(args.seeds),
                      args.budget, args.physics,
                      args.type_name, args.embodiment, args.lock_state,
                      args.key_place, tuple(args.tasks), not args.no_contact_repair,
                      False if args.no_object_trials else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
