"""Evaluate the vertical slice: task success and cause identification against the event log.

    python scripts/slice_eval.py --config configs/tabletop.yaml slice.episodes=20

Conditions: none, push (random 30-50 N sideways shove at a random time), heavier_object (the
object is 0.3-0.6 kg heavier than it looks). Writes results/slice_eval.{json,md}, the episodic
memory to results/slice_memory.jsonl, and one GIF per condition.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from _cli import parse

from pai.agents.slice_agent import run_episode
from pai.causes.inference import CAUSES, calibrate
from pai.envs import TabletopEnv
from pai.memory.episodic import EpisodicMemory
from pai.memory.report import episode_report
from pai.world.model import load_world_model

CONDITIONS = ("none", "push", "heavier_object")


def disturbance_for(condition: str, rng) -> list[dict]:
    if condition == "push":
        ang = rng.uniform(0, 2 * np.pi)
        f = rng.uniform(30, 50)
        return [{"type": "push", "start": int(rng.integers(120, 320)), "duration": 15, "body": "hand",
                 "force": [float(f * np.cos(ang)), float(f * np.sin(ang)), 0.0]}]
    if condition == "heavier_object":
        return [{"type": "payload", "start": 0, "duration": -1, "body": "red", "mass": float(rng.uniform(0.3, 0.6))}]
    return []


def extra(ap):
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--calibration-episodes", type=int, default=8)


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = load_world_model(cfg.slice.world_model, device)
    env_cfg = type(cfg.env)({**cfg.env, "render_images": not args.no_gif})
    env = TabletopEnv(env_cfg, disturbances=[])
    out = Path("results")
    (out / "slice_memory.jsonl").unlink(missing_ok=True)
    memory = EpisodicMemory(out / "slice_memory.jsonl")
    rng = np.random.default_rng(cfg.seed)

    # Calibrate surprise on clean episodes with their own seeds, never the evaluation seeds.
    raw = []
    for i in range(args.calibration_episodes):
        run_episode(env, wm, cfg, -1, seed=5000 + i, device=device, raw=raw)
    calibration = calibrate(np.stack([r for r, _ in raw]), np.stack([s for _, s in raw]))
    floor_mm, floor_n = (1000 * calibration.floor[:3]).round(2).tolist(), calibration.floor[3:].round(2).tolist()
    (out / "slice_calibration.json").write_text(json.dumps(
        {"position_floor_mm": floor_mm, "force_floor_N": floor_n, "threshold": round(calibration.threshold, 2),
         "steps": len(raw)}, indent=1))
    print(f"calibration: floor {floor_mm} mm and {floor_n} N, threshold {calibration.threshold:.1f} "
          f"({len(raw)} clean steps)", flush=True)

    rows, ep = [], 0
    for cond in args.conditions:
        for i in range(int(cfg.slice.episodes)):
            frames = [] if (i == 0 and not args.no_gif) else None
            rec = run_episode(env, wm, cfg, ep, seed=1000 + i, disturbances=disturbance_for(cond, rng),
                              device=device, frames=frames, calibration=calibration)
            memory.add(rec)
            if frames:
                import imageio
                imageio.mimsave(out / f"slice_{cond}.gif", frames, duration=0.08, loop=0)
            rows.append({"condition": cond, "success": rec.success, "inferred": rec.inferred_cause,
                         "fallbacks": rec.fallbacks,
                         "posterior": rec.cause_posterior, "steps": rec.steps, "subgoals": rec.subgoal_times})
            print(f"[{cond:14s} {i:2d}] success={rec.success!s:5s} inferred={rec.inferred_cause:14s} regrasps={rec.fallbacks} "
                  f"steps={rec.steps:3d} reached={list(rec.subgoal_times)[-1:] or ['-']}", flush=True)
            ep += 1

    summary = {"episodes": len(rows)}
    for cond in args.conditions:
        rs = [r for r in rows if r["condition"] == cond]
        summary[cond] = {"success": float(np.mean([r["success"] for r in rs])),
                         "cause_accuracy": float(np.mean([r["inferred"] == cond for r in rs])),
                         "inferred": dict(Counter(r["inferred"] for r in rs))}
    summary["task_success_overall"] = float(np.mean([r["success"] for r in rows]))
    summary["cause_accuracy_overall"] = float(np.mean([r["inferred"] == r["condition"] for r in rows]))
    disturbed = [r for r in rows if r["condition"] != "none"]
    if disturbed:
        summary["cause_accuracy_disturbed"] = float(np.mean([r["inferred"] == r["condition"] for r in disturbed]))
    (out / "slice_eval.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))

    lines = ["# Vertical slice: on(red, plate)", "",
             f"{len(rows)} episodes, privileged object state, learned entity world model, MPPI planning.", "",
             "| condition | task success | cause accuracy | inferred causes |", "|---|---|---|---|"]
    for cond in args.conditions:
        s = summary[cond]
        lines.append(f"| {cond} | {s['success']:.0%} | {s['cause_accuracy']:.0%} | {s['inferred']} |")
    lines += ["", "Confusion (rows: true cause, columns: inferred):", "",
              "| true \\ inferred | " + " | ".join(CAUSES) + " |", "|---" * (len(CAUSES) + 1) + "|"]
    for cond in args.conditions:
        c = Counter(r["inferred"] for r in rows if r["condition"] == cond)
        lines.append(f"| {cond} | " + " | ".join(str(c.get(k, 0)) for k in CAUSES) + " |")
    (out / "slice_eval.md").write_text("\n".join(lines) + "\n")
    dt = cfg.env.control_dt * cfg.slice.action_repeat
    reports = "\n\n".join(episode_report(r, dt) for r in memory.records)
    (out / "slice_reports.md").write_text("# Episode reports (the agent's own account vs ground truth)\n\n```\n"
                                          + reports + "\n```\n")
    print("\n".join(lines))
