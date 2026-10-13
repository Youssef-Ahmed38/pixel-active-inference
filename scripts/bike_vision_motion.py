"""Evaluate a frozen image orientation model on actual moving bicycle episodes."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from pai.bike.env import BicycleEnv, BalanceController
from pai.bike.vision import ImageOrientation
from pai.cognition.contracts import SkillCommand


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/bike/vision/orientation.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[820, 821, 822, 823, 824])
    p.add_argument("--video", type=Path, default=ROOT / "docs/media/bike/vision_motion.mp4")
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_vision_motion.json")
    args = p.parse_args()
    model = ImageOrientation.load(args.checkpoint)
    if len(args.seeds) != len(set(args.seeds)):
        p.error("motion episode seeds must be distinct")
    excluded = {model.summary.get("train_seed"), model.summary.get("validation_seed")}
    excluded.update(model.summary.get("train_episode_seeds", ()))
    excluded.update(model.summary.get("validation_episode_seeds", ()))
    if set(args.seeds) & excluded:
        p.error("motion episode seeds must be separate from teaching")
    episodes = []
    writer = None
    try:
        for index, seed in enumerate(args.seeds):
            rng = np.random.default_rng(seed)
            env = BicycleEnv()
            env.reset(seed, speed=rng.uniform(2.5, 4.5), roll=rng.uniform(-.1, .1))
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 3, 120, -15
            expert = BalanceController()
            predictions, labels = [], []
            with mujoco.Renderer(env.model, height=96, width=96) as renderer:
                if index == 0:
                    import imageio.v2 as imageio
                    args.video.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(str(args.video), format="FFMPEG", fps=25, macro_block_size=1)
                while not env.done:
                    if env.steps % 2 == 0:
                        # External tracking is supplied by this benchmark. It is
                        # not an image-estimated camera target or world position.
                        camera.lookat[:] = env.data.qpos[:3]
                        renderer.update_scene(env.data, camera=camera)
                        rgb = renderer.render().copy()
                        predictions.append(model.predict(rgb).tolist())
                        w, x, y, z = env.data.sensor("orientation").data
                        labels.append([math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                                       math.asin(np.clip(2 * (w * y - z * x), -1, 1)),
                                       math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])
                        if writer is not None:
                            writer.append_data(rgb)
                    action = expert.action(env.sensation())
                    env.execute(SkillCommand("ride", parameters=tuple(map(float, action))))
            if writer is not None:
                writer.close()
                writer = None
            predictions, labels = np.asarray(predictions), np.asarray(labels)
            # Wrap angular residuals even though these upright trials are small.
            error = (predictions - labels + np.pi) % (2 * np.pi) - np.pi
            episodes.append({"seed": seed, "frames": len(labels), "success": not env.fallen and env.travelled >= env.distance,
                             "mae_radians": np.abs(error).mean(axis=0).tolist(),
                             "zero_predictor_mae_radians": np.abs(labels).mean(axis=0).tolist(),
                             "predicted_angles": predictions.tolist(), "teacher_angles": labels.tolist()})
    finally:
        if writer is not None:
            writer.close()
    # Episodes, rather than temporally correlated frames, have equal weight.
    mean_error = np.mean([episode["mae_radians"] for episode in episodes], axis=0)
    zero_error = np.mean([episode["zero_predictor_mae_radians"] for episode in episodes], axis=0)
    report = {"scope": "frozen RGB orientation model on moving expert rides with supplied camera tracking",
              "checkpoint": str(args.checkpoint), "episodes": episodes,
              "episode_mean_mae_radians": mean_error.tolist(), "episode_mean_zero_mae_radians": zero_error.tolist(),
              "improves_zero_baseline_all_channels": bool(np.all(mean_error < zero_error)),
              "limitations": ["expert still controls using privileged sensations; model predictions do not control the bicycle",
                              "camera lookat tracks simulator root; no image-based tracking or translation estimate",
                              "fixed scene and camera calibration; frames within a ride are correlated",
                              "no full motion extraction, inverse dynamics from video or Stage 2 transfer"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
