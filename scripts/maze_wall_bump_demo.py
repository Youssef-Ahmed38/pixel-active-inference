"""Video: the same walk into a wall with visual walls (left) and solid walls (right).

    python scripts/maze_wall_bump_demo.py --seed 0

The G1 is told to walk from the start cell into a neighbouring wall cell, then into an open cell.
With visual walls the body walks into the wall until its pelvis is inside the wall cell; with solid
walls its legs and arms hit the wall, it feels the stall from its own odometry, backs off and the
agent records a bump.
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
from pai.maze.env import MazeWorld
from pai.maze.walker import WalkerMaze

DIRECTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]


def run(seed: int, solid: bool, every: int) -> tuple[list, WalkerMaze]:
    world = MazeWorld(seed)
    walker = WalkerMaze(world, seed=seed, solid=solid)
    r0, c0 = world.start
    wall = next(d for d in DIRECTIONS if world.grid[r0 + d[0], c0 + d[1]])
    free = next(d for d in DIRECTIONS if not world.grid[r0 + d[0], c0 + d[1]])
    m = walker.env.models[0]
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # look from the side at the face of the wall the body walks into (0.5 m from the start centre)
    toward = np.array([wall[1], -wall[0]], float)
    cam.lookat[:] = [0.35 * toward[0], 0.35 * toward[1], 0.45]
    yaw = np.degrees(np.arctan2(toward[1], toward[0]))
    cam.distance, cam.elevation, cam.azimuth = 2.4, -38, yaw + 90
    frames, k = [], [0]
    label = "solid walls" if solid else "visual walls (first version)"

    def on_frame(env):
        k[0] += 1
        if k[0] % every:
            return
        mujoco.mj_setState(m, d, env.state[0], STATE_SPEC)
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, cam)
        img = renderer.render().copy()
        for j, t in enumerate((label, f"t {walker.t:4.1f} s   bumps {walker.bumps}   falls {int(walker.fell)}")):
            cv2.putText(img, t, (10, 22 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(img)

    walker.on_frame = on_frame
    walker.step(wall)
    walker.step(free)
    return frames, walker


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--out", default="docs/media/maze/walker_solid_walls")
    args = ap.parse_args()
    left, wl = run(args.seed, False, args.every)
    right, wr = run(args.seed, True, args.every)
    n = max(len(left), len(right))
    left += [left[-1]] * (n - len(left))
    right += [right[-1]] * (n - len(right))
    frames = [np.hstack([a, b]) for a, b in zip(left, right)] + [np.hstack([left[-1], right[-1]])] * 25
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out.with_suffix(".mp4"), frames, fps=25, quality=6, macro_block_size=8)
    small = [cv2.resize(f, (f.shape[1] // 3, f.shape[0] // 3), interpolation=cv2.INTER_AREA) for f in frames[::3]]
    imageio.mimsave(out.with_suffix(".gif"), small, duration=3 / 25, loop=0)
    cv2.imwrite(str(out.with_suffix(".png")), cv2.cvtColor(frames[len(frames) // 3], cv2.COLOR_RGB2BGR))
    print(f"visual: bumps {wl.bumps} fell {wl.fell} t {wl.t:.1f}   solid: bumps {wr.bumps} fell {wr.fell} t {wr.t:.1f}")
    for p in (out.with_suffix(".mp4"), out.with_suffix(".gif"), out.with_suffix(".png")):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
