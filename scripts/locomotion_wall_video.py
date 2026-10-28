"""Video: v3 and v4 walk into the same solid wall, keep pushing, then back off (side by side).

    python scripts/locomotion_wall_video.py --seed 57100

Same body dynamics, same wall 1.0 m ahead, same commands (0.6 m/s for 6 s, then -0.3 m/s for 3 s).
Writes docs/media/locomotion/g1_wall_contact.mp4 / .gif / .png (< 3 MB).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.locomotion.body import STATE_SPEC
from pai.locomotion.env import EnvConfig, LocomotionEnv
from pai.locomotion.policy import LocomotionController


def episode(policy: Path, seed: int, gap: float, fps: int):
    cfg = EnvConfig(randomize=False, nthread=1, episode_s=1e6, walls=1)
    env = LocomotionEnv(1, cfg, seed=seed, stage=len(cfg.stages) - 1)
    env.pushes_enabled = False
    m = env.models[0]
    yaw = float(env.yaw()[0])
    xy = env.qpos()[0, :2]
    g = m.geom("train_wall_0").id
    m.geom_pos[g, :2] = xy + (gap + 0.5) * np.array([np.cos(yaw), np.sin(yaw)])
    m.geom_quat[g] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    ctrl = LocomotionController(policy)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 320, 400)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.elevation, cam.azimuth = 3.0, -12, np.degrees(yaw) + 90
    obs = env.observe()
    frames, fell = [], False
    every = max(1, int(round(1 / (fps * cfg.ctrl_dt))))
    k = 0
    for label, vx, secs in (("walk into the wall (0.6 m/s)", 0.6, 6.0), ("back off (-0.3 m/s)", -0.3, 3.0)):
        env.set_command([0], vx, 0.0, 0.0, heading=yaw)
        for _ in range(int(round(secs / cfg.ctrl_dt))):
            obs, _, _, info = env.step(ctrl.act(obs))
            fell = fell or bool(info["fell"][0])
            if k % every == 0:
                mujoco.mj_setState(m, d, env.state[0], STATE_SPEC)
                mujoco.mj_forward(m, d)
                r.update_scene(d, cam)
                frames.append((r.render().copy(), label, fell))
            k += 1
            if fell:
                break
        if fell:
            break
    r.close()
    return frames, fell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=57100)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default="docs/media/locomotion/g1_wall_contact")
    args = ap.parse_args()
    runs = {name: episode(ROOT / f"runs/locomotion/policy_{name}/policy.pt", args.seed, args.gap, args.fps)
            for name in ("v3", "v4")}
    n = max(len(f) for f, _ in runs.values())
    frames = []
    for i in range(n + args.fps):
        tiles = []
        for name, (fr, fell) in runs.items():
            img, label, f = fr[min(i, len(fr) - 1)]
            img = img.copy()
            cv2.putText(img, f"{name}: {label}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            status = "FELL" if f else ("done, upright" if i >= len(fr) - 1 else "")
            cv2.putText(img, status, (8, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 80, 80) if f else (120, 230, 120), 2, cv2.LINE_AA)
            tiles.append(img)
        frames.append(np.hstack(tiles))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out.with_suffix(".mp4"), frames, fps=args.fps, quality=6, macro_block_size=8)
    imageio.mimsave(out.with_suffix(".gif"), [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)) for f in frames[::2]],
                    duration=2 / args.fps, loop=0)
    cv2.imwrite(str(out.with_suffix(".png")), cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_RGB2BGR))
    print({k: ("fell" if v[1] else "upright") for k, v in runs.items()})
    for p in (out.with_suffix(".mp4"), out.with_suffix(".gif")):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
