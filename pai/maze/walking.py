"""A planar stepping body behind the maze's adjacent-move skill.

The body alternates support feet, samples its centre-of-mass path, and can fall
and recover under an injected perturbation. It is a small gait simulator, not a
MuJoCo humanoid or a learned walking controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pai.maze.env import Cell, DIRECTIONS, MazeWorld, Observation


@dataclass(frozen=True)
class WalkResult:
    observation: Observation
    executed: bool
    stalled: bool
    duration: float
    path: tuple[tuple[float, float], ...]
    fell: bool = False


class WalkingMobile:
    def __init__(self, world: MazeWorld, *, cadence: float = 2.0,
                 step_width: float = 0.18, fall_at: int | None = None):
        if cadence <= 0 or step_width <= 0:
            raise ValueError("cadence and step width must be positive")
        self.world = world
        self.cadence = cadence
        self.step_width = step_width
        self.fall_at = fall_at
        self.falls = 0
        self.recoveries = 0
        self.support = 0
        self.xy = np.asarray(world.position, dtype=float)
        self.feet = np.stack((self.xy + [0, -step_width / 2],
                              self.xy + [0, step_width / 2]))
        self.time = 0.0

    def move(self, direction: Cell) -> WalkResult:
        if direction not in DIRECTIONS:
            raise ValueError("expected an adjacent move")
        if self.fall_at is not None and not self.falls and self.world.steps >= self.fall_at:
            self.falls += 1
            self.world.steps += 1
            self.world.events.append({"step": self.world.steps, "type": "body_fell"})
            self.time += 1.0
            self.recoveries += 1
            self.world.events.append({"step": self.world.steps, "type": "body_recovered"})
            return WalkResult(self.world.observe(collided=True), False, True, 1.0,
                              (tuple(self.xy),), fell=True)
        start = self.xy.copy()
        target = start + np.asarray(direction, dtype=float)
        next_cell = tuple(map(int, target))
        blocked = not (0 <= next_cell[0] < self.world.size and
                       0 <= next_cell[1] < self.world.size) or bool(self.world.grid[next_cell])
        duration = 2.0 / self.cadence
        self.time += duration
        if blocked:
            obs = self.world.step(direction)
            return WalkResult(obs, False, True, duration, (tuple(start),))
        # Each skill consists of a left and a right support transfer. The swing
        # foot is placed ahead before the centre of mass crosses it.
        path = [tuple(start)]
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        for phase in (0.25, 0.5, 0.75, 1.0):
            if phase in (0.5, 1.0):
                swing = 1 - self.support
                self.feet[swing] = start + phase * (target - start) + (
                    (-1 if swing == 0 else 1) * self.step_width / 2) * lateral
                self.support = swing
            path.append(tuple(start + phase * (target - start)))
        self.xy = target
        obs = self.world.step(direction)
        return WalkResult(obs, True, False, duration, tuple(path))
