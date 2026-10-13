"""Teach bicycle orientation from rendered RGB and evaluate separate pose seeds."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from pai.bike.env import BicycleEnv
from pai.bike.vision import ImageOrientation


def poses(env, renderer, seed, count):
    rng = np.random.default_rng(seed)
    images, labels = [], []
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, .69]
    camera.distance, camera.azimuth, camera.elevation = 3, 120, -15
    for _ in range(count):
        roll, pitch, yaw = rng.uniform([-.35, -.12, -.4], [.35, .12, .4])
        # Compose labelled rotations in the simulator-side teacher only.
        quat = np.empty(4)
        mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), "xyz")
        env.reset(speed=0, roll=0)
        env.data.qpos[3:7] = quat
        for joint, bound in (("steering", .3), ("torso_lean", .2)):
            env.data.qpos[env.joints[joint].qposadr[0]] = rng.uniform(-bound, bound)
        mujoco.mj_forward(env.model, env.data)
        renderer.update_scene(env.data, camera=camera)
        images.append(renderer.render().copy())
        # Teacher labels use the same conventional frame roll/yaw as sensation.
        w, x, y, z = quat
        labels.append([math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                       math.asin(np.clip(2 * (w * y - z * x), -1, 1)),
                       math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])
    return np.asarray(images), np.asarray(labels)


def metrics(model, images, labels):
    predictions = np.asarray([model.predict(image) for image in images])
    error = predictions - labels
    return {"frames": len(images), "mae_radians": np.abs(error).mean(axis=0).tolist(),
            "rmse_radians": np.sqrt((error ** 2).mean(axis=0)).tolist(),
            "zero_predictor_mae_radians": np.abs(labels).mean(axis=0).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-seed", type=int, default=810)
    p.add_argument("--validation-seed", type=int, default=811)
    p.add_argument("--eval-seed", type=int, default=812)
    p.add_argument("--train-frames", type=int, default=600)
    p.add_argument("--eval-frames", type=int, default=200)
    p.add_argument("--artifacts", type=Path, default=ROOT / "runs/bike/vision")
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_vision.json")
    args = p.parse_args()
    if len({args.train_seed, args.validation_seed, args.eval_seed}) != 3 or min(args.train_frames, args.eval_frames) < 1:
        p.error("distinct pose seeds and positive frame counts required")
    env = BicycleEnv()
    with mujoco.Renderer(env.model, height=96, width=96) as renderer:
        training = poses(env, renderer, args.train_seed, args.train_frames)
        validation = poses(env, renderer, args.validation_seed, 160)
        evaluation = poses(env, renderer, args.eval_seed, args.eval_frames)
    candidates = []
    for ridge in (.001, .01, .1, 1.0):
        candidate = ImageOrientation.fit(*training, ridge=ridge)
        candidates.append((metrics(candidate, *validation), candidate))
    validation_metric, model = min(candidates, key=lambda row: np.mean(row[0]["mae_radians"]))
    model.summary.update(train_seed=args.train_seed, validation_seed=args.validation_seed,
                         camera={"distance": 3, "azimuth": 120, "elevation": -15, "lookat": [0, 0, .69]},
                         angle_convention="frame ZYX Euler roll, pitch, yaw")
    model.save(args.artifacts / "orientation.json")
    from PIL import Image
    Image.fromarray(evaluation[0][0]).save(args.artifacts / "example.png")
    measured = metrics(model, *evaluation)
    report = {"scope": "single calibrated view RGB orientation regression, simulator-labelled static poses",
              "model": model.summary, "validation": validation_metric, "evaluation": measured,
              "improves_zero_baseline_all_channels": all(a < b for a, b in zip(measured["mae_radians"], measured["zero_predictor_mae_radians"])),
              "limitations": ["fixed camera, geometry, lighting and colours", "synthetic static poses, not independent video episodes",
                              "no translation, wheel phase, joint pose or velocity extraction",
                              "not yet connected to the inverse-dynamics student or closed-loop riding"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
