"""Paired held-out ablation of arm stacking and physical key reseating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from discovery_cognitive_eval import evaluate as evaluate_discovery
from multistep_arm_eval import evaluate as evaluate_arm


def compare(arm_seeds=(126, 127, 128), key_seeds=(2, 3),
            budget_arm=180, budget_key=300, suites=("arm", "key")):
    result = {"config": {"arm_seeds": list(arm_seeds), "key_seeds": list(key_seeds),
                         "budget_arm": budget_arm, "budget_key": budget_key,
                         "suites": list(suites)}, "pairs": {}}
    if "arm" in suites:
        rows = {}
        for enabled in (False, True):
            run = evaluate_arm(arm_seeds, budget_per_goal=budget_arm, sequence="tower",
                               contact_repair=enabled)
            rows["with_repair" if enabled else "without_repair"] = run
        result["pairs"]["arm_tower"] = rows
    if "key" in suites:
        rows = {}
        for enabled in (False, True):
            run = evaluate_discovery(key_seeds, budget_key, physics=True,
                                     type_name="hd_lever_pull_right", lock_state="key",
                                     key_place="drawer", tasks=("door",),
                                     contact_repair=enabled, object_trials=enabled)
            rows["with_repair" if enabled else "without_repair"] = run
        result["pairs"]["key_in_drawer"] = rows
    result["summary"] = {}
    for name, pair in result["pairs"].items():
        before = pair["without_repair"]["episodes"]
        after = pair["with_repair"]["episodes"]
        result["summary"][name] = {
            "without_repair": sum(row["success"] for row in before) / len(before),
            "with_repair": sum(row["success"] for row in after) / len(after),
            "per_seed": [{"seed": a["seed"], "without_repair": bool(b["success"]),
                          "with_repair": bool(a["success"])} for b, a in zip(before, after)],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-seeds", nargs="*", type=int, default=[126, 127, 128])
    parser.add_argument("--key-seeds", nargs="*", type=int, default=[2, 3])
    parser.add_argument("--budget-arm", type=int, default=180)
    parser.add_argument("--budget-key", type=int, default=300)
    parser.add_argument("--suites", nargs="+", choices=("arm", "key"), default=["arm", "key"])
    parser.add_argument("--out", type=Path, default=Path("results/contact_repair_ablation.json"))
    args = parser.parse_args()
    report = compare(args.arm_seeds, args.key_seeds, args.budget_arm, args.budget_key,
                     tuple(args.suites))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
