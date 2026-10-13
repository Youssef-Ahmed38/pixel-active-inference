"""Render ground truth, the agent's place map and its local camera to a GIF.

    python scripts/maze_demo.py --seed 0 --out docs/media/maze/exploration.gif
The sidecar JSON records the actual disturbances, reports and episode metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.maze.advanced import run_advanced_episode  # noqa: E402
from pai.maze.env import MazeWorld, VISUAL_COLORS, VisualObservation  # noqa: E402


def _panel(world, agent, camera):
    size = world.size
    truth = np.empty((size, size, 3), np.uint8)
    memory = np.full_like(truth, VISUAL_COLORS["unknown"])
    truth[:] = VISUAL_COLORS["floor"]
    truth[world.grid.astype(bool)] = VISUAL_COLORS["wall"]
    for p in world.visited:
        truth[p] = (155, 215, 170)
    truth[world.goal] = VISUAL_COLORS["goal"]
    truth[world.position] = (245, 40, 215)
    for p, kind in agent.localizer.map_labels.items():
        if 0 <= p[0] < size and 0 <= p[1] < size:
            memory[p] = VISUAL_COLORS.get(kind, VISUAL_COLORS["unknown"])
    for p in agent.memory.visits:
        if 0 <= p[0] < size and 0 <= p[1] < size:
            memory[p] = (155, 215, 170)
    p = agent.localizer.position
    if 0 <= p[0] < size and 0 <= p[1] < size:
        memory[p] = (245, 40, 215)
    scale = 8
    left = np.repeat(np.repeat(truth, scale, 0), scale, 1)
    mid = np.repeat(np.repeat(memory, scale, 0), scale, 1)
    local = camera.image if isinstance(camera, VisualObservation) else np.zeros_like(left)
    # nearest-neighbour resize local camera to the same panel size
    indices = np.linspace(0, local.shape[0] - 1, left.shape[0]).astype(int)
    right = local[indices][:, indices]
    gap = np.full((left.shape[0], 5, 3), 15, np.uint8)
    return np.concatenate([left, gap, mid, gap, right], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=21)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "media" / "maze" / "exploration.gif")
    args = ap.parse_args()
    world = MazeWorld(args.seed, size=args.size)
    frames = []

    def capture(world, agent, camera, causes):
        if world.steps % args.every == 0:
            frames.append(_panel(world, agent, camera))

    report = run_advanced_episode(world, visual=True, mobile=True, budget=args.budget,
                                  block_at=30, shift_at=50, odom_noise=0.35,
                                  seed=args.seed, frame_hook=capture)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=0.12, loop=0)
    imageio.imwrite(args.out.with_name(args.out.stem + "_first.png"), frames[0])
    imageio.imwrite(args.out.with_name(args.out.stem + "_last.png"), frames[-1])
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps({"panels": ["truth for viewer", "agent memory", "local camera"],
                                   "report": report, "events": world.events}, indent=2) + "\n")
    print(f"wrote {args.out} ({len(frames)} frames) and {sidecar}")


if __name__ == "__main__":
    main()
