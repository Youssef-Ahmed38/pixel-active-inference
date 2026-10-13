"""Held-out key-chain, wrong-key explanation, and recipe-transfer evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.discovery.mock import MockWorld, Scenario, available_states, scenario_types  # noqa: E402
from pai.discovery.recipes import RecipeBook  # noqa: E402
from pai.maze.env import MazeWorld  # noqa: E402
from pai.maze.key_chain import run_key_chain  # noqa: E402


def forced_decoy_scene(type_name: str, seed: int) -> MockWorld:
    """Place both keys in the drawer with the first selected one incompatible.

    This uses the simulator answer key only while generating the challenge.
    The agent receives generic object entities and contact outcomes.
    """
    fixture = MockWorld(Scenario(type_name, "both", "drawer", n_keys=2,
                                 seed=seed, p_fail=0.0, pen=False))
    keys = sorted((oid, obj) for oid, obj in fixture.objs.items()
                  if obj["name"].startswith("key"))
    for oid, obj in keys:
        obj["place"] = "drawer"
        obj["pos"] = fixture.places["drawer"]
        obj["moved"] = False
        obj["matches"] = oid == keys[-1][0]
    return fixture


def evaluate(seeds=range(20, 24), size=21, move_budget=1200, fixture_budget=180):
    book = RecipeBook()
    training = run_key_chain(
        MazeWorld(seed=0, size=15),
        MockWorld(Scenario("hd_lever_pull_right", "both", "drawer", n_keys=1,
                           seed=0, p_fail=0.0, pen=False)),
        seed=0, recipes=book, learn_recipe=True)
    if not training.success or len(book) != 1:
        raise RuntimeError("training episode did not produce one door recipe")
    types = [name for name in scenario_types("test") if "both" in available_states(name)]
    rows = []
    for type_name in types:
        for seed in seeds:
            paired = {}
            for label, recipes in (("cold", None), ("warm", book)):
                fixture = forced_decoy_scene(type_name, seed)
                result = run_key_chain(MazeWorld(seed=seed, size=size), fixture,
                                       seed=seed, move_budget=move_budget,
                                       fixture_budget=fixture_budget, recipes=recipes)
                reports = [event for event in result.events if event["stage"] == "diagnose"]
                correct = [event["cause"] == "incompatible_key" and
                           not fixture.objs[event["object"]]["matches"]
                           for event in reports if event["cause"] == "incompatible_key"]
                paired[label] = {"success": result.success, "stage": result.stage,
                                 "moves": result.moves, "fixture_actions": result.fixture_actions,
                                 "wrong_key_reports": len(correct),
                                 "correct_wrong_key_reports": sum(correct),
                                 "events": result.events}
            rows.append({"type": type_name, "seed": seed, **paired})
    n = len(rows)
    reports = sum(row["cold"]["wrong_key_reports"] for row in rows)
    correct = sum(row["cold"]["correct_wrong_key_reports"] for row in rows)
    return {"config": {"seed_ids": list(seeds), "size": size,
                       "move_budget": move_budget, "fixture_budget": fixture_budget,
                       "test_types": types, "forced_wrong_key_first": True,
                       "train_type": "hd_lever_pull_right", "train_seed": 0},
            "summary": {"n": n,
                        "cold_passed": sum(row["cold"]["success"] for row in rows),
                        "warm_passed": sum(row["warm"]["success"] for row in rows),
                        "wrong_key_explanation_accuracy": correct / max(reports, 1),
                        "wrong_key_reports": reports,
                        "cold_mean_fixture_actions": sum(row["cold"]["fixture_actions"] for row in rows) / n,
                        "warm_mean_fixture_actions": sum(row["warm"]["fixture_actions"] for row in rows) / n,
                        "warm_faster": sum(row["warm"]["fixture_actions"] <
                                           row["cold"]["fixture_actions"] for row in rows)},
            "episodes": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--size", type=int, default=21)
    parser.add_argument("--out", type=Path, default=ROOT / "results/key_chain_stage4_eval.json")
    args = parser.parse_args()
    report = evaluate(range(args.seed_start, args.seed_start + args.seeds), size=args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
