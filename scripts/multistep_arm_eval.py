"""Evaluate two ordered arm relations: put blue on the plate, then red on blue."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.cognition.adapters import ArmDomainAdapter  # noqa: E402
from pai.cognition.contracts import GoalRelation  # noqa: E402
from pai.cognition.controller import ArmRelationalPolicy, CognitiveController  # noqa: E402
from pai.cognition.contact_repair import stack_clearance_repairs  # noqa: E402
from pai.cognition.memory import CognitiveMemory  # noqa: E402
from pai.config import load_config  # noqa: E402
from pai.envs import TabletopEnv  # noqa: E402
from pai.world.model import load_world_model  # noqa: E402


NAMED_SEQUENCES = {
    "stack": ("on(blue,plate)", "on(red,blue)"),
    "tidy": ("on(blue,plate)", "in(red,bowl)"),
    "stack_tidy": ("on(blue,plate)", "on(red,blue)", "in(yellow,bowl)"),
    "tower": ("on(blue,plate)", "on(red,blue)", "on(green,red)"),
}


def parse_goal(text: str) -> GoalRelation:
    match = re.fullmatch(r"(on|in)\(([a-z]+),\s*([a-z]+)\)", text)
    if match is None:
        raise ValueError(f"invalid relation goal: {text}")
    return GoalRelation(*match.groups())


def evaluate(seeds, budget_per_goal=150, sequence="stack", goal_texts=None,
             repair=True, contact_repair=True):
    cfg = load_config(str(ROOT / "configs/tabletop.yaml"), [])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_world_model(cfg.slice.world_model, device)
    memory = CognitiveMemory()
    goals = [parse_goal(text) for text in (goal_texts or NAMED_SEQUENCES[sequence])]
    rows = []
    env_cfg = type(cfg.env)({**cfg.env, "render_images": False})
    for seed in seeds:
        torch.manual_seed(seed)
        env = TabletopEnv(env_cfg, disturbances=[])
        adapter = ArmDomainAdapter(env, seed=seed, action_repeat=cfg.slice.action_repeat,
                                   settle_steps=cfg.slice.settle_steps)
        controller = CognitiveController(memory)
        policies = []
        def policy_for(goal):
            policy = ArmRelationalPolicy(env, model, cfg.slice, goal.predicate,
                                         goal.subject, goal.object, device,
                                         contact_repair=contact_repair)
            policies.append(policy)
            return policy
        def repair_for(goal, state):
            if contact_repair:
                return stack_clearance_repairs(goal, state)
            if goal.predicate != "on" or goal.object == "bowl":
                return []
            blockers = [subject for predicate, subject, target in state.relations
                        if predicate == "on" and target == goal.subject and subject != goal.object]
            return [GoalRelation("in", blocker, "bowl") for blocker in sorted(blockers)]
        result = controller.run_sequence(adapter, goals, policy_for, budget_per_goal, seed,
                                         max_passes=3, repair_factory=repair_for if repair else None)
        state = adapter.observe()
        rows.append({"seed": seed, "success": result.success,
                     "goals": [{"goal": f"{r.goal.predicate}({r.goal.subject}, {r.goal.object})",
                                "success": r.success, "steps": r.steps,
                                "last_stage": policies[i].goal.current.name if policies[i].goal.current else "done",
                                "failure_cause": next((event["cause"] for event in reversed(r.events)
                                                       if event["type"] == "failure_trace"), None),
                                "contact_transitions": sum(bool(event.get("contact_events")) for event in r.events),
                                "failure_trace": next((event for event in reversed(r.events)
                                                       if event["type"] == "failure_trace"), None),
                                "relation_loss_events": [event for event in r.events
                                                         if event["type"] == "action_transition"
                                                         and any(relation[0] in ("on", "in")
                                                                 for relation in event["relations_lost"])],
                                "positions": {e.id: [e.features[k] for k in ("x", "y", "z")]
                                              for e in r.beliefs[-1].entities},
                                "relations": sorted([list(x) for x in r.beliefs[-1].relations
                                                     if x[0] in ("on", "in", "grasped")])}
                               for i, r in enumerate(result.episodes)],
                     "final_positions": {e.id: [e.features[k] for k in ("x", "y", "z")]
                                         for e in state.entities},
                     "final_relations": sorted([list(x) for x in state.relations
                                                if x[0] in ("on", "in")]),
                     "unmet_goals": [f"{g.predicate}({g.subject},{g.object})" for g in goals
                                     if not state.satisfies(g)]})
    return {"config": {"seeds": list(seeds), "budget_per_goal": budget_per_goal,
                       "sequence": sequence, "goals": [f"{g.predicate}({g.subject},{g.object})" for g in goals],
                       "repair": repair, "contact_repair": contact_repair},
            "summary": {"n": len(rows), "success_rate": sum(r["success"] for r in rows) / len(rows)},
            "episodes": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[123, 124, 125])
    ap.add_argument("--budget-per-goal", type=int, default=150)
    ap.add_argument("--sequence", choices=tuple(NAMED_SEQUENCES), default="stack")
    ap.add_argument("--goal", action="append", dest="goals",
                    help="ordered relation such as on(blue,plate); repeat for a custom task")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--no-contact-repair", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results/multistep_arm_eval.json")
    args = ap.parse_args()
    result = evaluate(args.seeds, args.budget_per_goal, args.sequence, args.goals,
                      repair=not args.no_repair, contact_repair=not args.no_contact_repair)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
