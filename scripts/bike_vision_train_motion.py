"""Fit an RGB orientation candidate on disjoint whole motion episodes."""
import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from pai.bike.vision import ImageOrientation
from pai.bike.vision_teacher import collect_motion


def evaluate(model, datasets):
    episodes = []
    for images, labels, summary in datasets:
        predicted = np.asarray([model.predict(image) for image in images])
        error = (predicted - labels + np.pi) % (2 * np.pi) - np.pi
        episodes.append({**summary, "mae_radians": np.abs(error).mean(axis=0).tolist(),
                         "zero_predictor_mae_radians": np.abs(labels).mean(axis=0).tolist()})
    return {"episodes": episodes, "episode_mean_mae_radians": np.mean([e["mae_radians"] for e in episodes], axis=0).tolist(),
            "episode_mean_zero_mae_radians": np.mean([e["zero_predictor_mae_radians"] for e in episodes], axis=0).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seeds", type=int, nargs="+", default=list(range(830, 840)))
    p.add_argument("--validation-seeds", type=int, nargs="+", default=list(range(840, 844)))
    p.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(850, 855)))
    p.add_argument("--policy-checkpoint", type=Path)
    p.add_argument("--grid-size", type=int, choices=[4, 8, 16, 32], default=8)
    p.add_argument("--artifacts", type=Path, default=ROOT / "runs/bike/vision_motion")
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_vision_motion_training.json")
    args = p.parse_args()
    all_seeds = args.train_seeds + args.validation_seeds + args.eval_seeds
    if len(set(all_seeds)) != len(all_seeds):
        p.error("whole episode seeds must be distinct across all splits")
    datasets = {}
    controller = None
    if args.policy_checkpoint:
        from pai.bike.observation_learning import ObservedRidingPolicy
        controller = ObservedRidingPolicy.load(args.policy_checkpoint)
    for name, seeds in (("training", args.train_seeds), ("validation", args.validation_seeds), ("evaluation", args.eval_seeds)):
        datasets[name] = [collect_motion(seed, controller) for seed in seeds]
        print(f"Collected {name}: {len(seeds)} rides", flush=True)
    images = np.concatenate([row[0] for row in datasets["training"]])
    labels = np.concatenate([row[1] for row in datasets["training"]])
    choices = []
    for ridge in (.0001, .001, .01, .1):
        model = ImageOrientation.fit(images, labels, ridge, args.grid_size)
        validation = evaluate(model, datasets["validation"])
        choices.append((np.mean(validation["episode_mean_mae_radians"]), model, validation))
    _, model, validation = min(choices, key=lambda choice: choice[0])
    model.summary.update(train_episode_seeds=args.train_seeds, validation_episode_seeds=args.validation_seeds,
                         teacher_policy_checkpoint=str(args.policy_checkpoint) if args.policy_checkpoint else None,
                         teacher_policy_sha256=hashlib.sha256(args.policy_checkpoint.read_bytes()).hexdigest() if args.policy_checkpoint else None,
                         angle_convention="frame ZYX Euler roll, pitch, yaw",
                         camera={"distance": 3, "azimuth": 120, "elevation": -15, "tracking": "supplied root position"})
    model.save(args.artifacts / "orientation.json")
    evaluation = evaluate(model, datasets["evaluation"])
    measured, zero = evaluation["episode_mean_mae_radians"], evaluation["episode_mean_zero_mae_radians"]
    report = {"scope": "RGB orientation regression trained on moving rides, whole-episode split",
              "training": model.summary, "validation": validation, "evaluation": evaluation,
              "improves_zero_baseline_all_channels": all(a < b for a, b in zip(measured, zero)),
              "limitations": ["supervised simulator labels; not observation-only learning from arbitrary videos",
                              "fixed calibrated camera with root tracking supplied by simulator",
                              "expert controls the ride; no image-controlled bicycle or full joint dynamics extraction",
                              "same model, lighting, geometry and start distribution across episode splits"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation": evaluation, "improves_zero_baseline_all_channels": report["improves_zero_baseline_all_channels"]}, indent=2))


if __name__ == "__main__":
    main()
