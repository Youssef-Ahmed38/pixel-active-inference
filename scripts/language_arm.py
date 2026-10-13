"""Execute held-out text instructions through the existing MuJoCo arm planner."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from pai.cognition.adapters import ArmDomainAdapter
from pai.cognition.controller import ArmRelationalPolicy
from pai.cognition.memory import CognitiveMemory
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.language import GroundedLanguage, LanguageSession, LearnedAttributes, GroundedArmAdapter, GroundedFrame
from pai.world.model import load_world_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123])
    parser.add_argument("--languages", nargs="+", choices=("english", "arabic"), default=["english", "arabic"])
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--relations", nargs="+", choices=("on", "in"), default=["on", "in"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/language_arm.json")
    parser.add_argument("--episodes", type=Path, default=ROOT / "runs/language/arm_episodes.jsonl")
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs/tabletop.yaml"), [])
    env_cfg = type(cfg.env)({**cfg.env, "render_images": False})
    # Controlled scene palette uses taught vocabulary; physical properties unchanged.
    env_cfg.plate.rgba = [0.15, 0.3, 0.85, 1]
    env_cfg.bowl.rgba = [0.15, 0.7, 0.2, 1]
    teacher_env = TabletopEnv(env_cfg, disturbances=[])
    teacher_adapter = ArmDomainAdapter(teacher_env, seed=0)
    teacher_adapter.observe()
    examples = []
    for name, token in zip(teacher_adapter.names, teacher_adapter.last_tokens):
        labels = {"kind": "gripper" if name == "gripper" else teacher_env.objects[name].kind}
        if name != "gripper":
            labels["color"] = "blue" if name == "plate" else "green" if name == "bowl" else name
        examples.append((token, labels))
    attributes = LearnedAttributes.fit(examples)
    attributes.save(ROOT / "runs/language/arm_attributes.json")
    world = load_world_model(cfg.slice.world_model, "cpu")
    rows = []
    memory = CognitiveMemory(args.episodes)
    for language in args.languages:
        model = GroundedLanguage.load(ROOT / f"runs/language/{language}.json")
        session = LanguageSession(model)
        frames = [GroundedFrame("instruction", "on", "red", "block", "blue", "plate") if predicate == "on"
                  else GroundedFrame("instruction", "in", "yellow", "block", "green", "bowl")
                  for predicate in args.relations]
        instructions = [model.describe(frame) for frame in frames]
        for seed in args.seeds:
            torch.manual_seed(seed)
            env = TabletopEnv(env_cfg, disturbances=[])
            arm = ArmDomainAdapter(env, seed=seed, action_repeat=cfg.slice.action_repeat,
                                   settle_steps=cfg.slice.settle_steps)
            adapter = GroundedArmAdapter(arm, attributes)
            def policy(goal):
                return ArmRelationalPolicy(env, world, cfg.slice, goal.predicate,
                                           goal.subject, goal.object, "cpu")
            result = session.execute_sequence(instructions, adapter, policy, budget_per_goal=args.budget,
                                              episode=seed, max_passes=2)
            for record in result.episodes:
                memory.add(record)
            narration = [r for record in result.episodes for r in session.narrate(record)]
            final = adapter.observe()
            matches = [any(row["frame"] == replace(frame, intent="description") for row in narration) for frame in frames]
            answers = []
            for frame in frames:
                try:
                    answers.append(session.answer(model.describe(replace(frame, intent="question")), final))
                except ValueError:
                    answers.append(None)
            row = {"language": language, "seed": seed, "instructions": instructions,
                   "goals": [asdict(goal) for goal in result.goals], "success": result.success,
                   "steps": sum(record.steps for record in result.episodes),
                   "goal_observed": [final.satisfies(goal) for goal in result.goals],
                   "success_narrated_from_observation": matches, "answers": answers,
                   "narration": [{**r, "frame": asdict(r["frame"])} for r in narration]}
            rows.append(row)
            print(f"{language} seed {seed}: success={result.success}, steps={row['steps']}, narration={matches}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"scope": "MuJoCo arm, CPU MPPI, privileged entity-token perception, controlled palette",
                "episodes": rows, "complete": len(rows) == len(args.languages) * len(args.seeds)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
