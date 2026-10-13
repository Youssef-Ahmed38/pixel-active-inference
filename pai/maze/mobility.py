"""Kinematic continuous mobile skill behind the maze planner's movement action.

This models a moving base with a velocity limit and sampled continuous path.
It is not a learned gait or a MuJoCo humanoid; the L1 planner only commands
adjacent movement and is unchanged when this L0 adapter is selected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pai.maze.env import Cell, DIRECTIONS, MazeWorld, Observation


@dataclass(frozen=True)
class MoveResult:
    observation: Observation
    executed: bool
    stalled: bool
    duration: float
    path: tuple[tuple[float, float], ...]


class ContinuousMobile:
    def __init__(self, world: MazeWorld, speed: float = 1.0, dt: float = 0.1):
        if speed <= 0 or dt <= 0:
            raise ValueError("speed and dt must be positive")
        self.world, self.speed, self.dt = world, speed, dt
        self.xy = np.asarray(world.position, dtype=float)
        self.time = 0.0

    def move(self, direction: Cell) -> MoveResult:
        if direction not in DIRECTIONS:
            raise ValueError("expected an adjacent move")
        start = self.xy.copy()
        target = start + np.asarray(direction, float)
        n = max(1, int(np.ceil(1 / (self.speed * self.dt))))
        next_cell = tuple(map(int, np.rint(target)))
        blocked = not (0 <= next_cell[0] < self.world.size and 0 <= next_cell[1] < self.world.size) \
            or bool(self.world.grid[next_cell])
        path = [tuple(start)]
        if not blocked:
            for k in range(1, n + 1):
                path.append(tuple(start + (target - start) * k / n))
            self.xy = target
        self.time += n * self.dt
        obs = self.world.step(direction)
        return MoveResult(obs, not blocked, blocked, n * self.dt, tuple(path))
