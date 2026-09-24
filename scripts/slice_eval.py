"""Evaluate the vertical slice: task success and cause identification against the event log.

    python scripts/slice_eval.py --config configs/tabletop.yaml slice.episodes=20

Conditions: none, push (random 30-50 N sideways shove at a random time), heavier_object (the
object is 0.3-0.6 kg heavier than it looks), slippery_object (the object's friction is 5% of
normal), camera_shift (the camera is bumped 3-6 cm at a random time and stays there). Writes
results/<tag>_eval.{json,md}, the episodic memory, the per-step evidence for the thinker
(<tag>_evidence.npz) and one GIF per condition. --no-adapt runs the ablation in which explanations
do not change behaviour.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from _cli import parse

from pai.agents.slice_agent import run_episode
from pai.causes.inference import CAUSES, CHANNELS, calibrate
from pai.envs import TabletopEnv
from pai.memory.episodic import EpisodicMemory
from pai.memory.report import episode_report
from pai.world.model import load_world_model

CONDITIONS = ("none", "push", "heavier_object", "slippery_object", "camera_shift")


def disturbance_for(condition: str, rng) -> list[dict]:
    if condition == "push":
        ang = rng.uniform(0, 2 * np.pi)
        f = rng.uniform(30, 50)
        return [{"type": "push", "start": int(rng.integers(120, 320)), "duration": 15, "body": "hand",
                 "force": [float(f * np.cos(ang)), float(f * np.sin(ang)), 0.0]}]
    if condition == "heavier_object":
        return [{"type": "payload", "start": 0, "duration": -1, "body": "red", "mass": float(rng.uniform(0.3, 0.6))}]
    if condition == "slippery_object":
        return [{"type": "friction", "start": 0, "duration": -1, "body": "red", "scale": 0.05}]
    if condition == "camera_shift":
        ang, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.03, 0.06)
        return [{"type": "camera_shift", "start": int(rng.integers(40, 250)), "duration": -1,
                 "offset": [float(r * np.cos(ang)), float(r * np.sin(ang)), float(rng.uniform(-0.01, 0.01))]}]
    return []


def truth_estimate_error(cond: str, rec) -> dict:
    """How close the agent's estimate of the cause's parameters is to the truth (when it named it)."""
    if rec.inferred_cause != cond or not rec.true_disturbances or not rec.cause_params:
        return {}
    truth, est = rec.true_disturbances[0]["params"], rec.cause_params.get(cond, {})
    if cond == "heavier_object":
        return {"mass_true_kg": truth["mass"], "mass_est_kg": est["extra_mass_kg"]}
    if cond == "camera_shift":
        err = np.linalg.norm(np.array(est["camera_offset_mm"]) - 1000 * np.array(truth["offset"]))
        return {"offset_true_mm": (1000 * np.array(truth["offset"])).round(1).tolist(),
                "offset_est_mm": est["camera_offset_mm"], "offset_error_mm": round(float(err), 1)}
    return {}


def extra(ap):
    ap.add_argument("--conditions", type=lambda s: s.split(","), default=list(CONDITIONS),
                    help="comma-separated, e.g. none,push (a list would swallow the config overrides)")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--calibration-episodes", type=int, default=8)
    ap.add_argument("--slots", default=None,
                    help="slot checkpoint: perceive objects from the camera instead of exact simulator state")
    ap.add_argument("--tag", default="slice", help="prefix of the result files")
    ap.add_argument("--no-adapt", action="store_true", help="ablation: explanations do not change behaviour")


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = load_world_model(cfg.slice.world_model, device)
    pixels = args.slots is not None
    # From pixels the camera image is needed every step, at the DINOv2 size (224 = 16 x 14 patches).
    env_cfg = type(cfg.env)({**cfg.env, "render_images": pixels or not args.no_gif,
                             **({"image_size": 224} if pixels else {})})
    env = TabletopEnv(env_cfg, disturbances=[])
    perceive = None
    if pixels:
        from pai.perception.slot_tokens import SlotPerception
        perceive = SlotPerception(env, args.slots, device=device, camera=cfg.env.cameras[0].name)
    out = Path("results")
    (out / f"{args.tag}_memory.jsonl").unlink(missing_ok=True)
    memory = EpisodicMemory(out / f"{args.tag}_memory.jsonl")
    rng = np.random.default_rng(cfg.seed)

    # Calibrate surprise on clean episodes with their own seeds, never the evaluation seeds.
    raw = []
    for i in range(args.calibration_episodes):
        run_episode(env, wm, cfg, -1, seed=5000 + i, device=device, raw=raw, perceive=perceive)
    calibration = calibrate(np.stack([r[0] for r in raw]), np.stack([r[1] for r in raw]), [r[2] for r in raw])

    def split(v):  # position channels in mm (gripper, object, scene), force in N
        return (1000 * np.r_[v[:3], v[6:]]).round(2).tolist(), v[3:6].round(2).tolist()

    floor_mm, floor_n = split(calibration.floor)
    per_step = {k: dict(zip(("position_mm", "force_N"), split(v)))
                for k, v in calibration.step_floors.items()}
    (out / f"{args.tag}_calibration.json").write_text(json.dumps(
        {"position_floor_mm": floor_mm, "force_floor_N": floor_n, "threshold": round(calibration.threshold, 2),
         "per_step_floors": per_step, "per_step_thresholds": {k: round(v, 1) for k, v in calibration.step_thresholds.items()},
         "steps": len(raw)}, indent=1))
    print(f"calibration: floor {floor_mm} mm and {floor_n} N, threshold {calibration.threshold:.1f} "
          f"({len(raw)} clean steps); per step "
          + ", ".join(f"{k} {v:.0f}" for k, v in calibration.step_thresholds.items()), flush=True)

    rows, ep = [], 0
    ev_z, ev_hold, ev_phase, ev_ep, labels, teacher, onsets = [], [], [], [], [], [], []
    for cond in args.conditions:
        for i in range(int(cfg.slice.episodes)):
            frames = [] if (i == 0 and not args.no_gif) else None
            evidence = []
            rec = run_episode(env, wm, cfg, ep, seed=1000 + i, disturbances=disturbance_for(cond, rng),
                              device=device, frames=frames, calibration=calibration, perceive=perceive,
                              adapt=not args.no_adapt, evidence=evidence)
            if evidence:
                ev_z.append(np.stack([e[0] for e in evidence]).astype(np.float32))
                ev_hold.append(np.array([e[1] for e in evidence], bool))
                ev_phase.append(np.array([e[2] for e in evidence], np.int64))
                ev_ep.append(np.full(len(evidence), ep, np.int64))
            labels.append(cond)
            post = rec.cause_posterior or {"none": 1.0}
            teacher.append([post.get(c, 0.0) for c in CAUSES])
            onsets.append(rec.true_disturbances[0]["t"] // int(cfg.slice.action_repeat) if rec.true_disturbances else -1)
            memory.add(rec)
            if frames:
                import imageio
                imageio.mimsave(out / f"{args.tag}_{cond}.gif", frames, duration=0.08, loop=0)
            rows.append({"condition": cond, "success": rec.success, "inferred": rec.inferred_cause,
                         "fallbacks": rec.fallbacks, "adaptations": rec.adaptations,
                         "estimate": truth_estimate_error(cond, rec),
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
    masses = [r["estimate"] for r in rows if "mass_true_kg" in r["estimate"]]
    if masses:
        summary["mass_abs_error_kg"] = float(np.mean([abs(m["mass_est_kg"] - m["mass_true_kg"]) for m in masses]))
    cams = [r["estimate"]["offset_error_mm"] for r in rows if "offset_error_mm" in r["estimate"]]
    if cams:
        summary["camera_offset_error_mm"] = float(np.mean(cams))
    (out / f"{args.tag}_eval.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=1))
    if ev_z:
        np.savez_compressed(out / f"{args.tag}_evidence.npz", z=np.concatenate(ev_z), holding=np.concatenate(ev_hold),
                            phase=np.concatenate(ev_phase), episode=np.concatenate(ev_ep), labels=np.array(labels),
                            teacher=np.array(teacher, np.float32), causes=np.array(CAUSES),
                            channels=np.array(CHANNELS), onset=np.array(onsets, np.int64))

    lines = ["# Vertical slice: on(red, plate)", "",
             f"{len(rows)} episodes, {'objects perceived from the camera (DINOv2 + slots)' if pixels else 'privileged object state'}, "
             "learned entity world model, MPPI planning"
             f"{', explanations do not change behaviour (ablation)' if args.no_adapt else ''}.", "",
             "| condition | task success | cause accuracy | inferred causes |", "|---|---|---|---|"]
    for cond in args.conditions:
        s = summary[cond]
        lines.append(f"| {cond} | {s['success']:.0%} | {s['cause_accuracy']:.0%} | {s['inferred']} |")
    lines += ["", "Confusion (rows: true cause, columns: inferred):", "",
              "| true \\ inferred | " + " | ".join(CAUSES) + " |", "|---" * (len(CAUSES) + 1) + "|"]
    for cond in args.conditions:
        c = Counter(r["inferred"] for r in rows if r["condition"] == cond)
        lines.append(f"| {cond} | " + " | ".join(str(c.get(k, 0)) for k in CAUSES) + " |")
    if "mass_abs_error_kg" in summary or "camera_offset_error_mm" in summary:
        lines += ["", "Estimates when the cause was named correctly:", ""]
        if "mass_abs_error_kg" in summary:
            lines.append(f"- extra mass: mean absolute error {1000 * summary['mass_abs_error_kg']:.0f} g")
        if "camera_offset_error_mm" in summary:
            lines.append(f"- camera offset: mean error {summary['camera_offset_error_mm']:.1f} mm")
    (out / f"{args.tag}_eval.md").write_text("\n".join(lines) + "\n")
    dt = cfg.env.control_dt * cfg.slice.action_repeat
    reports = "\n\n".join(episode_report(r, dt) for r in memory.records)
    (out / f"{args.tag}_reports.md").write_text("# Episode reports (the agent's own account vs ground truth)\n\n```\n"
                                          + reports + "\n```\n")
    print("\n".join(lines))
