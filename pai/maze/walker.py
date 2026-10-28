"""The learned G1 walker inside the maze (Stage 2 under Stage 3).

The maze becomes a MuJoCo scene: every wall cell of a ``MazeWorld`` grid is a box around the G1, one
cell per ``cell`` metres, with the start cell at the robot's origin. ``WalkerMaze`` offers the grid
maze's interface (``observe``, ``step(direction)``), so every maze agent runs unchanged, but a step is
now carried out by the body: the learned locomotion policy (released v3) walks to the next cell
centre, steered only through its goal (speed and heading), turning on the spot first at corners.

Walls are solid by default: every collision capsule of the G1 (pelvis, legs, feet, torso, head,
arms, hands) collides with every wall box through one collision bit that nothing else uses, so
contacts with the floor and between body parts are exactly as before. A blocked walk is felt from
the body's own odometry (commanded forward but no progress for a second); the body then backs off to
its cell centre and the agent observes a bump. ``solid=False`` gives the first version's visual walls
(entering a wall cell was detected from the pelvis position).

Stated simplifications:
- The agent's local view of cells still comes from the grid sensor (``MazeWorld.observe`` at the
  cell the body is in), and its cell position comes from the simulator pose (odometry).
- A fall ends the episode (counted as a failure; the body is not re-placed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from pai.locomotion.body import G1Body, build_spec
from pai.locomotion.env import EnvConfig, LocomotionEnv, wrap
from pai.locomotion.policy import LocomotionController
from pai.maze.env import Cell, MazeWorld, Observation

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "runs" / "locomotion" / "policy_v3" / "policy.pt"
WALL_RGBA = (0.42, 0.45, 0.52, 1.0)
GOAL_RGBA = (0.96, 0.75, 0.14, 1.0)


def cell_xy(cell: Cell, start: Cell, size_m: float) -> np.ndarray:
    """World (x, y) of a cell centre; columns go +x, rows go -y, the start cell is the origin."""
    return np.array([(cell[1] - start[1]) * size_m, -(cell[0] - start[0]) * size_m])


WALL_BIT = 2   # collision bit shared only by the walls and the body's collision capsules


def build_maze_body(world: MazeWorld, cell: float = 1.0, wall_height: float = 0.5,
                    show_goal: bool = True, solid: bool = True) -> G1Body:
    """G1 scene with the maze's walls and, for videos, a goal marker."""
    spec = build_spec()
    half = cell / 2
    for r in range(world.size):
        for c in range(world.size):
            if world.grid[r, c]:
                x, y = cell_xy((r, c), world.start, cell)
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, wall_height / 2],
                                        pos=[x, y, wall_height / 2], contype=0,
                                        conaffinity=WALL_BIT if solid else 0,
                                        rgba=list(WALL_RGBA), name=f"wall_{r}_{c}")
    if show_goal:
        x, y = cell_xy(world.goal, world.start, cell)
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.3 * cell, 0.01, 0],
                                pos=[x, y, 0.01], contype=0, conaffinity=0, rgba=list(GOAL_RGBA),
                                name="goal_marker")
    if solid:
        # the capsules the G1 already uses against the floor (its explicit "*_floor" pairs); set on
        # the spec so the compiler also updates the per-body masks its broad phase uses
        for pair in spec.pairs:
            names = {pair.geomname1, pair.geomname2}
            if "floor" in names:
                (name,) = names - {"floor"}
                spec.geom(name).contype = WALL_BIT
    return G1Body(spec.compile())


@dataclass
class StepReport:
    seconds: float
    fell: bool
    bumped: bool
    reached: bool


class WalkerMaze:
    """Grid-maze interface whose moves are walked by the learned G1 controller."""

    def __init__(self, world: MazeWorld, policy: Path = POLICY, cell: float = 1.0, seed: int = 0,
                 speed: float = 0.6, step_timeout_s: float = 8.0, device: str = "cpu",
                 wall_height: float = 0.5, solid: bool = True, stall_s: float = 1.0):
        self.world = world
        self.cell = cell
        self.speed = speed
        self.timeout = step_timeout_s
        self.solid = solid
        self.stall_s = stall_s
        self.body = build_maze_body(world, cell, wall_height=wall_height, solid=solid)
        cfg = EnvConfig(randomize=False, nthread=1, episode_s=1e6)
        self.env = LocomotionEnv(1, cfg, seed=seed, stage=len(cfg.stages) - 1, body=self.body)
        self.env.pushes_enabled = False
        self.env.set_command([0], 0.0, 0.0, 0.0)
        self.ctrl = LocomotionController(policy, device=device)
        self.obs = self.env.observe()
        self.t = 0.0
        self.fell = False
        self.walk_seconds = 0.0
        self.bumps = 0
        self.frames: list | None = None
        self.on_frame = None          # optional callback(env) after every control step (video)

    # ------------------------------------------------------------------ body
    def xy(self) -> np.ndarray:
        return self.env.qpos()[0, 0:2].copy()

    def _tick(self) -> bool:
        a = self.ctrl.act(self.obs)
        self.obs, _, _, info = self.env.step(a)
        self.t += self.env.cfg.ctrl_dt
        if self.on_frame is not None:
            self.on_frame(self.env)
        if bool(info["fell"][0]):
            self.fell = True
        return self.fell

    def _cell_of(self, xy: np.ndarray) -> Cell:
        s = self.world.start
        return (int(round(s[0] - xy[1] / self.cell)), int(round(s[1] + xy[0] / self.cell)))

    def _walk_to(self, target_xy: np.ndarray, stop_if_wall: bool = True) -> StepReport:
        t0 = self.t
        dt = self.env.cfg.ctrl_dt
        trail = []                         # (time, xy) while commanded forward, for stall detection
        while self.t - t0 < self.timeout:
            d = target_xy - self.xy()
            dist = float(np.linalg.norm(d))
            if dist < 0.15 * self.cell:
                self.env.set_command([0], 0.0, 0.0, 0.0, heading=float(self.env.yaw()[0]))
                return StepReport(self.t - t0, False, False, True)
            heading = float(np.arctan2(d[1], d[0]))
            err = abs(float(wrap(heading - self.env.yaw()[0])))
            # turn on the spot first, then walk while steering toward the target point
            v = 0.0 if err > np.radians(25) else min(self.speed, 1.5 * dist)
            self.env.set_command([0], v, 0.0, 0.0, heading=heading)
            if self._tick():
                return StepReport(self.t - t0, True, False, False)
            here = self._cell_of(self.xy())
            if stop_if_wall and self._is_wall(here):
                return StepReport(self.t - t0, False, True, False)
            if v > 0.2:
                trail.append((self.t, self.xy()))
                while trail and self.t - trail[0][0] > self.stall_s:
                    trail.pop(0)
                if (stop_if_wall and self.t - t0 > self.stall_s and len(trail) * dt >= 0.9 * self.stall_s
                        and np.linalg.norm(trail[-1][1] - trail[0][1]) < 0.05 * self.cell):
                    return StepReport(self.t - t0, False, True, False)   # pressing against something
            else:
                trail.clear()
        return StepReport(self.t - t0, False, False, False)

    def _is_wall(self, c: Cell) -> bool:
        n = self.world.size
        return not (0 <= c[0] < n and 0 <= c[1] < n) or bool(self.world.grid[c])

    # ------------------------------------------------------------------ maze interface
    def observe(self, collided: bool = False) -> Observation:
        return self.world.observe(collided)

    def step(self, direction: Cell) -> Observation:
        if self.fell:
            raise RuntimeError("the walker fell; the episode is over")
        p = self.world.position
        target = (p[0] + direction[0], p[1] + direction[1])
        rep = self._walk_to(cell_xy(target, self.world.start, self.cell))
        self.walk_seconds += rep.seconds
        if rep.bumped:
            self.bumps += 1
            back = self._walk_to(cell_xy(p, self.world.start, self.cell), stop_if_wall=False)
            self.walk_seconds += back.seconds
        # the grid world is told what the body did; a wall target or a failed walk is a bump
        moved = rep.reached and not self._is_wall(target)
        obs = self.world.step(direction) if moved or self._is_wall(target) else self._stay(direction)
        return obs

    def _stay(self, direction: Cell) -> Observation:
        """The body did not arrive (timeout or fall): the grid position stays, as after a bump."""
        self.world.steps += 1
        self.world.events.append({"step": self.world.steps, "type": "walk_failed",
                                  "position": self.world.position})
        return self.world.observe(True)
