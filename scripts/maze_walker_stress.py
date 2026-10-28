"""Falls of the maze walker per 1000 moves: a stress test with enough falls to compare fixes.

    python scripts/maze_walker_stress.py --start 21000 --num 40 --moves 150 --smooth --output results/maze_walker_stress_dev_smooth.json

Full exploration episodes fall too rarely (2-8 falls in 60 episodes) to tell fixes apart, and a
replay of the same maze does not reproduce its fall. Here the G1 walker (``pai/maze/walker.py``, solid
walls, policy v4) makes ``--moves`` moves per maze of the kinds the explorers make: turn on the spot
and walk to a neighbouring open cell (70%), or walk into a wall to touch it and back off (30%, as the
explorer's touches of doubtful walls). The moves are random (seeded per maze); the measure is falls,
bumps and walking time, plus the body's state in the 2 s before each fall.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def run(job):
    seed, moves, policy, smooth, steady = job
    import torch
    torch.set_num_threads(1)
    from pai.maze.walker import WalkerMaze
    from pai.maze.env import MazeWorld
    world = MazeWorld(seed)
    kw = {"steady": steady} if steady is not None else {}
    w = WalkerMaze(world, seed=seed, solid=True, stall_s=1.0, policy=Path(policy), smooth=smooth, **kw)
    rng = np.random.default_rng(seed + 5150)
    buf = collections.deque(maxlen=100)
    tick = w._tick

    def logged_tick():
        cmd = w.env.command[0].copy()
        f = tick()
        g = w.env.kinematics()["grav"][0]
        buf.append([round(w.t, 2), round(float(cmd[0]), 2), round(math.degrees(math.atan2(g[1], -g[2])), 1),
                    round(math.degrees(math.atan2(-g[0], -g[2])), 1)])
        return f

    w._tick = logged_tick
    made = 0
    for _ in range(moves):
        p = world.position
        open_dirs = [d for d in DIRS if not w._is_wall((p[0] + d[0], p[1] + d[1]))]
        wall_dirs = [d for d in DIRS if d not in open_dirs]
        if wall_dirs and (rng.random() < 0.3 or not open_dirs):
            d = wall_dirs[int(rng.integers(len(wall_dirs)))]
        else:
            d = open_dirs[int(rng.integers(len(open_dirs)))]
        w.step(d)
        made += 1
        if w.fell:
            break
    return {"seed": seed, "moves": made, "fell": w.fell, "bumps": w.bumps, "walk_seconds": round(w.t, 1),
            "steadies": getattr(w, "steadies", 0), "before_fall": list(buf)[::5] if w.fell else []}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--moves", type=int, default=150)
    ap.add_argument("--policy", default="runs/locomotion/policy_v4/policy.pt")
    ap.add_argument("--smooth", action="store_true")
    ap.add_argument("--steady", type=float, default=None, help="tilt (deg) that makes the walker stop and steady itself")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    jobs = [(s, args.moves, args.policy, args.smooth, args.steady) for s in range(args.start, args.start + args.num)]
    with Pool(args.workers) as pool:
        eps = pool.map(run, jobs, chunksize=1)
    moves = sum(e["moves"] for e in eps)
    falls = sum(e["fell"] for e in eps)
    summary = {"mazes": len(eps), "moves": moves, "falls": falls, "falls_per_1000_moves": round(1000 * falls / moves, 2),
               "bumps": sum(e["bumps"] for e in eps), "walk_hours": round(sum(e["walk_seconds"] for e in eps) / 3600, 2),
               "steadies": sum(e["steadies"] for e in eps)}
    out = {"seeds": [args.start, args.start + args.num - 1], "moves_per_maze": args.moves, "policy": args.policy,
           "smooth": args.smooth, "steady": args.steady, "summary": summary, "episodes": eps}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
