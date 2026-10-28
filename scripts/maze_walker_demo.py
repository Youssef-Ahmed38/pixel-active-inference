"""Video: the maze agent explores while the learned G1 walker carries out every move.

    python scripts/maze_walker_demo.py --seed 74000            # third person + belief map
    python scripts/maze_walker_demo.py --seed 74000 --pov      # + what the agent sees and decides

Third person: the G1 in the maze (tracking camera). Belief map: the agent's own map (dark =
unknown, grey = believed wall, light = believed open, blue = visited, red = the agent, green = its
current target, gold = the goal once seen). With --pov: the robot's eye view (head-height camera
along its heading), the raw local sensor patch the agent receives this step, and its decision: the
chosen target and the top alternatives with their expected-free-energy terms.
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
from pai.maze.efe_agent import RouteEFEExplorer
from pai.maze.env import MazeWorld
from pai.maze.walker import WalkerMaze


def belief_panel(agent, world, height: int) -> np.ndarray:
    n = world.size
    px = height // n
    img = np.zeros((n * px, n * px, 3), np.uint8)
    b = agent.map.beliefs
    for r in range(n):
        for c in range(n):
            if (r, c) not in agent.map.observed_cells:
                col = (25, 28, 35)
            elif b[r, c, 2] > 0.5:
                col = (245, 190, 35)
            elif b[r, c, 0] > 0.5:
                col = (95, 100, 112)
            else:
                col = (215, 215, 205)
            if (r, c) in agent.map.visits and b[r, c, 0] <= 0.5:
                col = (120, 160, 230)
            img[r * px:(r + 1) * px, c * px:(c + 1) * px] = col
    if agent.target is not None:
        tr, tc = agent.target
        cv2.rectangle(img, (tc * px + 2, tr * px + 2), ((tc + 1) * px - 3, (tr + 1) * px - 3), (60, 200, 90), 2)
    pr, pc = world.position
    cv2.circle(img, (pc * px + px // 2, pr * px + px // 2), px // 3, (220, 50, 50), -1)
    out = np.zeros((height, n * px, 3), np.uint8)
    out[:n * px] = img
    return out


DIRS = {(-1, 0): "north", (1, 0): "south", (0, 1): "east", (0, -1): "west"}
KIND_RGB = {"wall": (95, 100, 112), "floor": (215, 215, 205), "goal": (245, 190, 35)}


def eye_view(renderer, d, cam, pelvis_id, yaw) -> np.ndarray:
    """Head-height camera looking along the body's heading."""
    pos = d.xpos[pelvis_id]
    fwd = np.array([np.cos(yaw), np.sin(yaw)])
    cam.lookat[:] = [pos[0] + 1.5 * fwd[0], pos[1] + 1.5 * fwd[1], 0.9]
    cam.distance, cam.azimuth, cam.elevation = 1.4, np.degrees(yaw), -12
    renderer.update_scene(d, cam)
    img = renderer.render().copy()
    cv2.putText(img, "robot's eye view", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def sensor_patch(obs, size: int, radius: int = 3) -> np.ndarray:
    """The cells the agent receives this step, centred on itself (black = not in view)."""
    n = 2 * radius + 1
    px = size // n
    img = np.zeros((size, size, 3), np.uint8)
    r0, c0 = obs.position
    for (r, c), kind in obs.visible.items():
        i, j = r - r0 + radius, c - c0 + radius
        img[i * px:(i + 1) * px - 1, j * px:(j + 1) * px - 1] = KIND_RGB[kind]
    cv2.circle(img, (radius * px + px // 2, radius * px + px // 2), px // 3, (220, 50, 50), -1)
    cv2.putText(img, "what it senses now", (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def decision_panel(agent, move, step, width: int, height: int) -> np.ndarray:
    img = np.full((height, width, 3), 18, np.uint8)
    lines = [(f"decision at grid step {step}", (255, 255, 255)),
             (f"move: {DIRS.get(move, 'none')}", (255, 255, 255))]
    cands = getattr(agent, "last_candidates", None) or []
    for k, c in enumerate(cands):
        tag = "CHOSEN" if c["cell"] == agent.target else f"alt {k}"
        col = (90, 210, 110) if tag == "CHOSEN" else (170, 170, 170)
        lines.append((f"{tag}: target {c['cell']}  EFE {c['efe']:+.1f}", col))
        lines.append((f"   info {c['gain']:.1f} bits  goal/risk {c['pragmatic']:+.2f}  walk {c['effort']}", col))
    if cands and agent.target == cands[0]["cell"]:
        c = cands[0]
        if len(cands) > 1:
            # name the EFE term that gives the chosen target its lead over the runner-up
            a = cands[1]
            ew, fw = getattr(agent, "epistemic_weight", 1.0), getattr(agent, "effort_weight", 2.0)
            lead = {"goal belief pulls it there": c["pragmatic"] - a["pragmatic"],
                    "more new information there": -ew * (c["gain"] - a["gain"]),
                    "shorter walk": fw * (c["effort"] - a["effort"])}
            why = min(lead, key=lead.get)
        else:
            why = "the only candidate"
        lines.append((f"why: {why}", (240, 200, 90)))
    elif cands:
        lines.append(("why: keeps its current target", (240, 200, 90)))
    lines.append(("EFE = goal/risk - info gain + 2 x walk", (130, 130, 130)))
    for i, (t, col) in enumerate(lines):
        cv2.putText(img, t, (8, 22 + 21 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=74000)
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--every", type=int, default=6, help="render every k control steps (20 ms each)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--branchy", action="store_true", help="Prim maze with more loops and open rooms")
    ap.add_argument("--pov", action="store_true", help="add the eye view, sensor patch and decision")
    ap.add_argument("--sensing-from", default=None,
                    help="a results/maze_walker_<regime>.json file: use its regime and the sensor model the agent learned")
    args = ap.parse_args()

    world = MazeWorld(args.seed, **(dict(generator="prim", loop_probability=0.25, rooms=3) if args.branchy else {}))
    walker = WalkerMaze(world, seed=args.seed, wall_height=1.4 if args.pov else 0.5)
    env, obs_model, repeat_weight = walker, None, 1.0
    if args.sensing_from:
        import json
        from pai.maze.map_model import ObservationModel
        from pai.maze.noisy import NoisySensorMaze
        rep = json.loads(Path(args.sensing_from).read_text())
        obs_model = ObservationModel(matrix=np.array(rep["sensor"]["matrix"]))
        repeat_weight = rep["sensor"]["repeat_weight"]
        env = NoisySensorMaze(walker, seed=args.seed, persistent=rep["regime"] == "persistent")
    agent = RouteEFEExplorer(size=world.size, sensor_range=world.sensor_range, seed=args.seed,
                             effort_weight=2.0, switch_margin=0.0, obs_model=obs_model)
    agent.map.repeat_weight = repeat_weight
    m = walker.env.models[0]
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 360, 480)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.elevation, cam.azimuth = (7.5, -80, 90) if args.pov else (6.5, -58, 90)
    frames = []
    state = {"k": 0, "step": 0, "obs": None, "move": None}
    eye = mujoco.MjvCamera()
    eye.type = mujoco.mjtCamera.mjCAMERA_FREE
    pelvis = m.body("pelvis").id

    def on_frame(env):
        state["k"] += 1
        if state["k"] % args.every:
            return
        mujoco.mj_setState(m, d, env.state[0], STATE_SPEC)
        mujoco.mj_forward(m, d)
        r.update_scene(d, cam)
        img = r.render().copy()
        txt = [f"grid step {state['step']}   walked {walker.t:5.1f} s   bumps {walker.bumps}   falls {int(walker.fell)}",
               f"cells visited {len(world.visited)}   goal {'FOUND' if world.position == world.goal or world.goal in world.visited else 'not yet'}"]
        for j, t in enumerate(txt):
            cv2.putText(img, t, (10, 22 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if not args.pov:
            frames.append(np.hstack([img, belief_panel(agent, world, 360)]))
            return
        top = np.hstack([img, eye_view(r, d, eye, pelvis, float(env.yaw()[0]))])
        bmap = cv2.resize(belief_panel(agent, world, 300), (300, 300), interpolation=cv2.INTER_NEAREST)
        bottom = np.hstack([bmap, sensor_patch(state["obs"], 300),
                            decision_panel(agent, state["move"], state["step"], top.shape[1] - 600, 300)])
        frames.append(np.vstack([top, bottom]))

    walker.on_frame = on_frame
    obs = env.observe()
    for step in range(1, args.budget + 1):
        state["step"] = step
        move = agent.act(obs)
        if move is None:
            break
        state["obs"], state["move"] = obs, move
        obs = env.step(move)
        if walker.fell or obs.reached_goal:
            break
    state["k"] = args.every - 1               # render the final state, then hold it for a second
    on_frame(walker.env)
    for _ in range(args.fps):
        frames.append(frames[-1])
    out = ROOT / (args.out or ("docs/media/maze/walker_maze" + ("_branchy" if args.branchy else "")
                               + ("_pov" if args.pov else "")))
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out.with_suffix(".mp4"), frames, fps=args.fps, quality=6, macro_block_size=8)
    small = [cv2.resize(f, (f.shape[1] * 2 // 5, f.shape[0] * 2 // 5)) for f in frames[::2]]
    imageio.mimsave(out.with_suffix(".gif"), small, duration=2 / args.fps, loop=0)
    cv2.imwrite(str(out.with_suffix(".png")), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
    print(f"steps {state['step']}  goal {obs.reached_goal}  fell {walker.fell}  bumps {walker.bumps}  "
          f"walked {walker.t:.1f} s  visited {len(world.visited)}  frames {len(frames)}")
    for p in (out.with_suffix(".mp4"), out.with_suffix(".gif")):
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
