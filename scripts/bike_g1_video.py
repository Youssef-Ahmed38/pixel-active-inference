"""Video: the Unitree G1 riding the realistic bicycle down the road, with its practised balance gains.

    python scripts/bike_g1_video.py --gains-from results/bike_g1_practice.json --seed 3300

Chase camera behind and to the side, overlay with speed, distance and lean. Writes
docs/media/bike/g1_riding.mp4 / .gif / .png (< 3 MB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.bike.g1_rider import G1BikeEnv
from scripts.bike_g1_practice import policy, start


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gains-from", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3300)
    ap.add_argument("--distance", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="docs/media/bike/g1_riding")
    args = ap.parse_args()
    gains = np.array(json.loads(args.gains_from.read_text())["practised_gains"])
    env = G1BikeEnv(distance=args.distance, max_seconds=30.0)
    roll, speed = start(args.seed)
    s = env.reset(seed=args.seed, speed=speed, roll=roll)
    m, d = env.model, env.data
    r = mujoco.Renderer(m, 540, 960)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    every = max(1, int(round(1 / (args.fps * env.control_dt))))
    frames, k = [], 0
    while not env.done:
        s = env.step(policy(gains, s, speed))
        if k % every == 0:
            x, y = d.xpos[m.body("frame").id][:2]
            t = d.time
            cam.lookat[:] = [x + 0.4, y, 0.75]
            cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 140 + 25 * np.sin(0.25 * t)
            r.update_scene(d, cam)
            img = r.render().copy()
            txt = f"G1 on a 24-inch bicycle   speed {s[2]:4.1f} m/s   distance {env.travelled:5.1f} m   lean {np.degrees(s[0]):+5.1f} deg"
            cv2.putText(img, txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            frames.append(img)
        k += 1
    r.close()
    status = "fell" if env.fallen else f"rode {env.travelled:.1f} m"
    last = frames[-1].copy()
    cv2.putText(last, status, (14, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 230, 120) if not env.fallen else (255, 80, 80),
                2, cv2.LINE_AA)
    frames += [last] * args.fps
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out.with_suffix(".mp4"), frames, fps=args.fps, quality=6, macro_block_size=8)
    cv2.imwrite(str(out.with_suffix(".png")), cv2.cvtColor(frames[len(frames) // 3], cv2.COLOR_RGB2BGR))
    print(status, len(frames), "frames")


if __name__ == "__main__":
    main()
