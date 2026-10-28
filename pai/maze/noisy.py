"""Noisy local sensing for the maze: the regime where map uncertainty is real.

With the exact ``MazeWorld.observe`` every seen cell is known after one look, so curiosity has
nothing to resolve and nearest-frontier exploration is close to optimal. Here each visible cell's
label is drawn from a confusion matrix P(seen | true), independently at every step (or, with
``persistent=True``, once per cell and viewpoint, so a repeat look from the same spot repeats any
misreading), and a bump
reports the blocked cell as a wall (touch is reliable). Agents never see the matrix; they can only
fit an ``ObservationModel`` from training mazes.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from pai.maze.env import Cell, MazeWorld, Observation

KINDS = ("wall", "floor", "goal")

# columns: true wall, floor, goal; rows: seen wall, floor, goal. The goal is hard to recognise
# (40% of looks report plain floor) and 2% of floor looks are false goal sightings.
DEFAULT_CONFUSION = np.array([
    [0.90, 0.08, 0.02],
    [0.09, 0.90, 0.38],
    [0.01, 0.02, 0.60],
])


@dataclass(frozen=True)
class TouchObservation(Observation):
    touched: Cell | None = None  # the cell bumped into this step (touch is reliable, not a look)


class NoisySensorMaze:
    """MazeWorld wrapper whose ``observe``/``step`` return confused labels."""

    def __init__(self, world: MazeWorld, confusion: np.ndarray = DEFAULT_CONFUSION, seed: int = 0,
                 persistent: bool = False):
        self.world = world
        self.confusion = np.asarray(confusion, dtype=float)
        self.rng = np.random.default_rng(seed + 700000)
        # persistent: a cell seen from the same spot is always read the same way (a reflection or a
        # shadow), so only a new viewpoint gives an independent look
        self.persistent = persistent
        self._readings: dict[tuple[Cell, Cell], str] = {}

    def __getattr__(self, name):
        return getattr(self.world, name)

    def _corrupt(self, obs: Observation, bumped: Cell | None) -> Observation:
        visible = {}
        for cell, kind in obs.visible.items():
            key = (cell, obs.position)
            if self.persistent and key in self._readings:
                visible[cell] = self._readings[key]
                continue
            column = self.confusion[:, KINDS.index(kind)]
            visible[cell] = KINDS[int(self.rng.choice(3, p=column))]
            self._readings[key] = visible[cell]
        if bumped is not None and 0 <= bumped[0] < self.world.size and 0 <= bumped[1] < self.world.size:
            visible[bumped] = "wall"
        else:
            bumped = None
        return TouchObservation(obs.position, visible, obs.collided, obs.reached_goal, bumped)

    def observe(self, collided: bool = False) -> Observation:
        return self._corrupt(self.world.observe(collided), None)

    def step(self, direction: Cell) -> Observation:
        start = self.world.position
        obs = self.world.step(direction)
        bumped = (start[0] + direction[0], start[1] + direction[1]) if obs.collided else None
        return self._corrupt(obs, bumped)
