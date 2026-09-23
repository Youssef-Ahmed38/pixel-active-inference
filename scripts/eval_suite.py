"""Run the standard evaluation suite and write a Markdown summary table.

    python scripts/eval_suite.py --decoder runs/pixelai_decoder/decoder.pt --name pixelai

Runs: goal-distance curve (clean), reaching under every disturbance preset at a fixed goal
distance, and the perception task. Output: results/<name>_suite.md and .json.
"""

import json
from pathlib import Path

import torch
import yaml
from _cli import parse

from pai.config import REPO_ROOT, Config, apply_overrides
from pai.eval.reach import evaluate
from pai.models import load_decoder

DIST_SCALES = [0.1, 0.2, 0.35, 0.5, 1.0]


def extra(ap):
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--name", default="pixelai")
    ap.add_argument("--disturbance-goal-scale", type=float, default=0.35)


def run(base, overrides, decoder, device, task, tag):
    cfg = Config(apply_overrides(base.to_dict(), overrides))
    return evaluate(cfg, decoder, device, task, tag)


def fmt(x, digits=3):
    return "–" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    args, base = parse(__doc__, extra)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = load_decoder(args.decoder, device) if args.decoder else None
    presets = yaml.safe_load((REPO_ROOT / "configs" / "disturbances.yaml").read_text())
    n = args.name
    results = {"distance": [], "disturbance": [], "perception": None}

    for s in DIST_SCALES:
        r = run(base, [f"eval.goal_scale={s}", f"eval.save_gif={s == 0.35}"], decoder, device, "reach", f"{n}_dist{s}")
        results["distance"].append({"goal_scale": s, **r})
        print(f"distance {s}: success {r['success_rate']}", flush=True)

    gs = args.disturbance_goal_scale
    for name in ["none", *presets]:
        dist = [] if name == "none" else presets[name]
        ov = [f"eval.goal_scale={gs}", f"env.disturbances={json.dumps(dist)}", f"eval.save_gif={name != 'none'}"]
        r = run(base, ov, decoder, device, "reach", f"{n}_{name}")
        results["disturbance"].append({"disturbance": name, **r})
        print(f"disturbance {name}: success {r['success_rate']}", flush=True)

    results["perception"] = run(base, ["eval.steps=150"], decoder, device, "perception", f"{n}_perception")

    lines = [f"# Evaluation suite: {n}", "", "## Success vs goal distance (clean)", "",
             "| goal scale | start dist (m) | success | final dist (m) | steps to 2 cm |", "|---|---|---|---|---|"]
    for r in results["distance"]:
        lines.append(f"| {r['goal_scale']} | {fmt(r['start_ee_dist_median'])} | {fmt(r['success_rate'], 2)} | "
                     f"{fmt(r['final_ee_dist_median'])} | {fmt(r['steps_to_success_median'], 0)} |")
    lines += ["", f"## Robustness (goal scale {gs})", "",
              "| disturbance | success | final dist (m) | worst deviation after reaching (m) | steps outside after reaching |",
              "|---|---|---|---|---|"]
    for r in results["disturbance"]:
        lines.append(f"| {r['disturbance']} | {fmt(r['success_rate'], 2)} | {fmt(r['final_ee_dist_median'])} | "
                     f"{fmt(r['max_dev_after_success_median'])} | {fmt(r['steps_outside_after_success_median'], 0)} |")
    p = results["perception"]
    lines += ["", "## Perception (static arm, belief starts 0.3 rad off per joint)", "",
              f"Final belief error: {fmt(p['final_belief_err_mean'], 4)} rad (norm over joints), {p['episodes']} episodes."]
    out = Path("results")
    (out / f"{n}_suite.md").write_text("\n".join(lines) + "\n")
    (out / f"{n}_suite.json").write_text(json.dumps(results, indent=1, default=str))
    print("\n".join(lines))
