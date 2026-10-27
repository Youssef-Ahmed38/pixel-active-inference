"""Record the learned G1 controller: stand, walk, turn to a heading, run, take a push.

    python scripts/locomotion_video.py --policy runs/locomotion/policy/policy.pt --seed 50100

Writes docs/media/locomotion/g1_locomotion.mp4 and a smaller .gif (both kept under 3 MB).
The body has randomised dynamics drawn from ``--seed`` (a fresh seed), the goal changes on a
fixed script shown in the overlay.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pai.locomotion.body import STATE_SPEC  # noqa: E402
from pai.locomotion.env import EnvConfig, LocomotionEnv, wrap  # noqa: E402
from pai.locomotion.policy import LocomotionController  # noqa: E402

SCRIPT = [  # (seconds, label, vx, heading offset in deg or None, push dv)
    (2.5, "stand (goal: v = 0)", 0.0, 0, 0.0),
    (4.0, "walk 0.8 m/s", 0.8, 0, 0.0),
    (3.5, "turn to heading +90 deg", 0.6, 90, 0.0),
    (4.0, "run 2.0 m/s", 2.0, 90, 0.0),
    (2.0, "slow to 0.5 m/s", 0.5, 90, 0.0),
    (3.0, "push 1.0 m/s from the side", 0.5, 90, 1.0),
    (2.0, "stop", 0.0, 90, 0.0),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="runs/locomotion/policy/policy.pt")
    ap.add_argument("--seed", type=int, default=50100)
    ap.add_argument("--out", default="docs/media/locomotion/g1_locomotion")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    ctrl = LocomotionController(Path(args.policy))
    env = LocomotionEnv(1, EnvConfig(nthread=1, n_variants=1, episode_s=1e5), seed=args.seed, stage=4)
    env.pushes_enabled = False
    env.set_command([0], 0.0)
    env.reset([0])
    yaw0 = float(env.yaw()[0])
    m = env.models[0]
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.elevation = 3.2, -15
    frames = []
    obs = env.observe()
    every = max(1, int(round(1 / (args.fps * env.cfg.ctrl_dt))))
    k = 0
    falls = 0
    for secs, label, vx, head, push in SCRIPT:
        env.set_command([0], vx, 0.0, 0.0, heading=wrap(yaw0 + np.deg2rad(head)))
        if push:
            f = env.push_force_for_dv(push)
            side = yaw0 + np.deg2rad(head) + np.pi / 2
            env.apply_push([0], [f * np.cos(side), f * np.sin(side), 0.0])
        for _ in range(int(round(secs / env.cfg.ctrl_dt))):
            obs, _, _, info = env.step(ctrl.act(obs))
            falls += int(info["fell"][0])
            if k % every == 0:
                mujoco.mj_setState(m, d, env.state[0], STATE_SPEC)
                mujoco.mj_forward(m, d)
                cam.azimuth = np.rad2deg(yaw0) + 120
                r.update_scene(d, cam)
                img = r.render().copy()
                v = env.kinematics()["v_head"][0, 0]
                txt = [label, f"forward speed {v:4.2f} m/s   falls {falls}"]
                if env.push_left[0] > 0:
                    txt.append("PUSH")
                for j, t in enumerate(txt):
                    cv2.putText(img, t, (10, 24 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                frames.append(img)
            k += 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out.with_suffix(".mp4"), frames, fps=args.fps, quality=6, macro_block_size=8)
    small = [cv2.resize(f, (240, 180), interpolation=cv2.INTER_AREA) for f in frames[::3]]
    imageio.mimsave(out.with_suffix(".gif"), small, duration=3 / args.fps, loop=0)
    for p in (out.with_suffix(".mp4"), out.with_suffix(".gif")):
        print(p, f"{p.stat().st_size / 1e6:.2f} MB")
    print("falls during the clip:", falls)


if __name__ == "__main__":
    main()
