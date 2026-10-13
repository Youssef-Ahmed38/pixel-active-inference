"""Exercise the same cognitive goal/skill loop on the arm and maze domains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.cognition.adapters import ArmDomainAdapter, MazeDomainAdapter  # noqa: E402
from pai.cognition.contracts import GoalRelation  # noqa: E402
from pai.cognition.controller import ArmRelationalPolicy, CognitiveController, MazeFrontierPolicy  # noqa: E402
from pai.cognition.memory import CognitiveMemory  # noqa: E402
from pai.config import load_config  # noqa: E402
from pai.envs import TabletopEnv  # noqa: E402
from pai.maze.env import MazeWorld  # noqa: E402
from pai.world.model import load_world_model  # noqa: E402


def evaluate(arm_seeds: list[int], maze_seeds: list[int], maze_size=11, maze_budget=300):
    cfg = load_config(str(ROOT / "configs" / "tabletop.yaml"), [])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_world_model(cfg.slice.world_model, device)
    memory = CognitiveMemory()
    controller = CognitiveController(memory)
    rows = []
    env_cfg = type(cfg.env)({**cfg.env, "render_images": False})
    for seed in arm_seeds:
        torch.manual_seed(seed)
        env = TabletopEnv(env_cfg, disturbances=[])
        adapter = ArmDomainAdapter(env, seed=seed, action_repeat=cfg.slice.action_repeat,
                                   settle_steps=cfg.slice.settle_steps)
        policy = ArmRelationalPolicy(env, model, cfg.slice, device=device)
        record = controller.run(adapter, GoalRelation("on", "red", "plate"), policy,
                                budget=cfg.slice.max_plan_steps, episode=seed)
        rows.append({"domain": "arm", "seed": seed, "success": record.success,
                     "steps": record.steps, "commands": len(record.commands)})
    for seed in maze_seeds:
        world = MazeWorld(seed, size=maze_size, sensor_range=2)
        adapter = MazeDomainAdapter(world)
        policy = MazeFrontierPolicy(world.size, world.sensor_range, seed)
        record = controller.run(adapter, GoalRelation("goal_reached", "body"), policy,
                                budget=maze_budget, episode=seed)
        rows.append({"domain": "maze", "seed": seed, "success": record.success,
                     "steps": record.steps, "commands": len(record.commands)})
    return {"device": device, "episodes": rows,
            "summary": {domain: {"n": sum(r["domain"] == domain for r in rows),
                                 "success_rate": sum(r["success"] for r in rows if r["domain"] == domain)
                                 / max(1, sum(r["domain"] == domain for r in rows))}
                        for domain in ("arm", "maze")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-seeds", type=int, nargs="*", default=[123])
    ap.add_argument("--maze-seeds", type=int, nargs="*", default=[7, 8, 9])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "alignment_eval.json")
    args = ap.parse_args()
    result = evaluate(args.arm_seeds, args.maze_seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
