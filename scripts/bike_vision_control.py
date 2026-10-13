"""Compare hybrid RGB-attitude balance against the sensor-based baseline."""
import argparse
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from pai.bike.env import BicycleEnv, BalanceController
from pai.bike.vision import ImageOrientation
from pai.bike.vision_control import ImageBalanceController
from pai.cognition.contracts import SkillCommand


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/bike/vision_motion/orientation.json")
    p.add_argument("--policy-checkpoint", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(860, 865)))
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_vision_control.json")
    args = p.parse_args()
    model = ImageOrientation.load(args.checkpoint)
    teaching = set(model.summary.get("train_episode_seeds", ())) | set(model.summary.get("validation_episode_seeds", ()))
    if len(set(args.seeds)) != len(args.seeds) or set(args.seeds) & teaching:
        p.error("distinct evaluation seeds outside teaching required")
    episodes = []
    for seed in args.seeds:
        for condition in ("sensor_baseline", "hybrid_rgb"):
            rng = np.random.default_rng(seed)
            env = BicycleEnv()
            env.reset(seed, speed=rng.uniform(2.5, 4.5), roll=rng.uniform(-.1, .1))
            if args.policy_checkpoint:
                from pai.bike.observation_learning import ObservedRidingPolicy
                baseline = ObservedRidingPolicy.load(args.policy_checkpoint)
            else:
                baseline = BalanceController()
            visual = ImageBalanceController(model, baseline)
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 3, 120, -15
            with mujoco.Renderer(env.model, height=96, width=96) as renderer:
                while not env.done:
                    if condition == "sensor_baseline":
                        action = baseline.action(env.sensation())
                    else:
                        camera.lookat[:] = env.data.qpos[:3]
                        renderer.update_scene(env.data, camera=camera)
                        # External speed and joint encoders are declared hybrid
                        # inputs. Frame attitude/rate come only from images.
                        action = visual.action(renderer.render(), env.data.time,
                                               float(env.data.sensor("velocity").data[0]),
                                               float(env.data.qpos[env.joints["steering"].qposadr[0]]),
                                               float(env.data.qpos[env.joints["torso_lean"].qposadr[0]]))
                    env.execute(SkillCommand("ride", parameters=tuple(map(float, action))))
            episodes.append({"seed": seed, "condition": condition, "success": not env.fallen and env.travelled >= env.distance,
                             "distance": env.travelled, "fallen": env.fallen, "seconds": env.data.time})
    summary = {condition: {"episodes": len(args.seeds),
                          "successes": sum(e["success"] for e in episodes if e["condition"] == condition)}
               for condition in ("sensor_baseline", "hybrid_rgb")}
    report = {"scope": "hybrid RGB orientation plus external speed/joint encoders, paired fresh starts", "summary": summary,
              "episodes": episodes, "checkpoint": str(args.checkpoint),
              "policy_checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else None,
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "policy_sha256": hashlib.sha256(args.policy_checkpoint.read_bytes()).hexdigest() if args.policy_checkpoint else None,
              "limitations": ["camera tracks simulator root; this is not image-based tracking",
                              "speed and steering/lean encoders are supplied sensor channels; not fully pixel-controlled riding",
                              "fixed image and control models; no visual policy training on these evaluation episodes",
                              "small paired sample; no Stage 2 transfer comparison"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
