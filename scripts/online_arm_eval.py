"""Compare the arm's Bayesian cause controller with its live HRM cause hook."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.agents.slice_agent import run_episode  # noqa: E402
from pai.causes.inference import Calibration  # noqa: E402
from pai.cognition.thinker import OnlineArmCauseThinker  # noqa: E402
from pai.config import load_config  # noqa: E402
from pai.envs import TabletopEnv  # noqa: E402
from pai.world.model import load_world_model  # noqa: E402


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    def floor(row):
        positions = np.asarray(row["position_mm"]) / 1000
        return np.r_[positions[:3], np.asarray(row["force_N"]), positions[3:]]
    positions = np.asarray(data["position_floor_mm"]) / 1000
    return Calibration(floor=np.r_[positions[:3], np.asarray(data["force_floor_N"]), positions[3:]],
                       threshold=data["threshold"],
                       step_floors={k: floor(v) for k, v in data["per_step_floors"].items()},
                       step_thresholds=data["per_step_thresholds"])


def evaluate(seeds, checkpoint, calibration_path):
    cfg = load_config(str(ROOT / "configs/tabletop.yaml"), [])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_world_model(cfg.slice.world_model, device)
    calibration = load_calibration(calibration_path)
    env_cfg = type(cfg.env)({**cfg.env, "render_images": False})
    rows = []
    for seed in seeds:
        disturbances = [{"type": "camera_shift", "start": 150, "duration": -1,
                         "offset": [0.04, 0.0, 0.0]}]
        for condition in ("bayesian", "online_thinker"):
            env = TabletopEnv(env_cfg, disturbances=[])
            thinker = OnlineArmCauseThinker(checkpoint, device="cpu") if condition == "online_thinker" else None
            # Constructing HRM consumes Torch RNG. Restore the same planner
            # stream in both paired conditions before the first MPPI action.
            torch.manual_seed(seed)
            record = run_episode(env, model, cfg, 0, seed, disturbances=disturbances,
                                 device=device, calibration=calibration, online_thinker=thinker)
            rows.append({"seed": seed, "condition": condition, "success": record.success,
                         "steps": record.steps, "inferred_cause": record.inferred_cause,
                         "adaptations": record.adaptations,
                         "thinker_reports": thinker.reports if thinker is not None else []})
    return {"config": {"seeds": list(seeds), "checkpoint": str(checkpoint)},
            "summary": {condition: {"n": sum(r["condition"] == condition for r in rows),
                                    "success_rate": sum(r["success"] for r in rows
                                                        if r["condition"] == condition)
                                    / len(seeds)}
                        for condition in ("bayesian", "online_thinker")}, "episodes": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=[123])
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "runs/cross_domain_thinker/arm_pretrained.pt")
    ap.add_argument("--calibration", type=Path,
                    default=ROOT / "results/slice_v7/slice_calibration.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/online_arm_eval.json")
    args = ap.parse_args()
    result = evaluate(args.seeds, args.checkpoint, args.calibration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
