"""One-shot recipe reuse: learn a recipe from ONE success, reuse it in new layouts with less thinking.

    python scripts/recipe_eval.py --config configs/tabletop.yaml --episodes 20

1. Source: one episode of on(red, plate) with the full planner. If it succeeds, its recipe is
   extracted, and counterfactual replay in the world model says which steps the success needed.
2. Test on new layouts (seeds never used before), three agents:
     full planner        the reference (samples, iterations from the config)
     small planner       a planner with a fraction of the compute (--samples, --iterations)
     small + recipe      the same small planner, with the recalled recipe as an extra candidate
   A recipe is useful if "small + recipe" succeeds where "small planner" fails, i.e. a solved task
   needs less thinking the second time.
Writes results/recipe_eval.{json,md} and results/recipe_library.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from _cli import parse

from pai.agents.slice_agent import run_episode
from pai.causes.credit import assign_credit
from pai.envs import TabletopEnv
from pai.goals.relations import RelationalGoal
from pai.memory.recipes import RecipeLibrary, check_preconditions, extract_recipe
from pai.train.common import Progress
from pai.world.entities import POS, encode, entity_names
from pai.world.model import load_world_model


def extra(ap):
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--samples", type=int, default=16, help="samples of the small planner")
    ap.add_argument("--iterations", type=int, default=1, help="MPPI iterations of the small planner")
    ap.add_argument("--source-seed", type=int, default=2000)
    ap.add_argument("--objects", default="red", help="comma-separated blocks to test on (the recipe is learned on red)")


def with_planner(cfg, samples, iterations):
    sc = type(cfg.slice)({**cfg.slice, "samples": samples, "iterations": iterations})
    return type(cfg)({**cfg, "slice": sc})


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = load_world_model(cfg.slice.world_model, device)
    env = TabletopEnv(type(cfg.env)({**cfg.env, "render_images": False}), disturbances=[])
    out = Path("results")
    (out / "recipe_library.json").unlink(missing_ok=True)
    library = RecipeLibrary(out / "recipe_library.json")
    names = entity_names(env)

    # 1. one success -> one recipe
    seed = args.source_seed
    while True:
        rec = run_episode(env, wm, cfg, -1, seed=seed, device=device)
        if rec.success:
            break
        print(f"source episode (seed {seed}) failed, trying the next layout", flush=True)
        seed += 1
    goal = RelationalGoal("on", "red", "plate", names, env.objects)
    order = [g.name for g in goal.subgoals]
    rest_z = goal.t_top + goal.h_o / 2

    def on_target(x):  # final imagined state: object resting on the target
        o, t = x[goal.i_o, POS], x[goal.i_t, POS]
        return float(((o[:2] - t[:2]) ** 2).sum() + (o[2] - rest_z) ** 2)

    base, credit = assign_credit(wm, rec.tokens, rec.actions, on_target, rec.phases, success_threshold=0.03 ** 2,
                                 device=device)
    why = [{"name": c.name, "description": c.description, "necessary": bool(c.necessary),
            "importance": round(c.importance, 5)} for c in credit[:5]] if base < 0.03 ** 2 else []
    recipe = extract_recipe(rec, "on", "red", "plate", names, env.objects, order, why)
    library.add(recipe)
    print(recipe.describe(), flush=True)
    if not why:
        print(f"(counterfactual replay: the imagined replay itself misses the goal, cost {base:.4f}; "
              "no 'why' recorded)", flush=True)

    # 2. reuse in new layouts
    small = with_planner(cfg, args.samples, args.iterations)
    agents = {"full planner": (cfg, False), "small planner": (small, False), "small + recipe": (small, True)}
    objects = args.objects.split(",")
    rows = []
    progress = Progress(args.episodes * len(agents) * len(objects), "recipe eval", every=len(agents), unit=" episodes")
    for obj in objects:
        for i in range(args.episodes):
            test_seed = 3000 + i
            for agent, (c, use) in agents.items():
                r = None
                if use:
                    r = library.recall("on", env.objects[obj].kind, env.objects["plate"].kind)
                    obs0 = env.reset(seed=test_seed)
                    t0 = encode(env, obs0)
                    missing = check_preconditions(r, t0, names.index(obj), names.index("plate"))
                    if missing:
                        r = None
                res = run_episode(env, wm, c, i, seed=test_seed, obj=obj, device=device, recipe=r)
                if use and r is not None:
                    library.record_use(r, res.success)
                rows.append({"object": obj, "seed": test_seed, "agent": agent, "success": res.success,
                             "steps": res.steps, "recipe_used": r is not None})
                progress.update()
                print(f"[{obj} {i:2d}] {agent:15s} success={res.success!s:5s} steps={res.steps}", flush=True)

    summary = {}
    for obj in objects:
        for agent, (c, _) in agents.items():
            rs = [r for r in rows if r["object"] == obj and r["agent"] == agent]
            ok = [r["steps"] for r in rs if r["success"]]
            summary[f"{obj}/{agent}"] = {"success": float(np.mean([r["success"] for r in rs])),
                                         "mean_steps_when_successful": float(np.mean(ok)) if ok else None,
                                         "rollouts_per_step": int(c.slice.samples) * int(c.slice.iterations)}
    (out / "recipe_eval.json").write_text(json.dumps({"summary": summary, "recipe": recipe.describe(),
                                                      "episodes": rows}, indent=1))
    lines = ["# One-shot recipe reuse", "",
             f"Recipe learned from one successful episode (seed {seed}); tested on {args.episodes} new layouts per object.", "",
             "| object | agent | rollouts per step | task success | steps (successful) |", "|---|---|---|---|---|"]
    for key, s in summary.items():
        obj, agent = key.split("/")
        steps = f"{s['mean_steps_when_successful']:.0f}" if s["mean_steps_when_successful"] else "-"
        lines.append(f"| {obj} | {agent} | {s['rollouts_per_step']} | {s['success']:.0%} | {steps} |")
    lines += ["", "```", recipe.describe(), "```"]
    (out / "recipe_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
