"""Place memory and information-seeking exploration over partial observations.

L2 requests coverage or finding a visible goal. L1 chooses a route to the most
valuable frontier using expected unseen cells per travel step. L0 executes one
adjacent mobile move. The policy never accesses MazeWorld.grid or truth().
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pai.maze.env import DIRECTIONS, Cell, MazeWorld, Observation, neighbors
from pai.planning.objective import expected_free_energy


@dataclass
class PlaceMemory:
    cells: dict[Cell, str] = field(default_factory=dict)
    visits: dict[Cell, int] = field(default_factory=dict)
    edges: dict[Cell, set[Cell]] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def update(self, observation: Observation) -> None:
        p = observation.position
        self.visits[p] = self.visits.get(p, 0) + 1
        self.edges.setdefault(p, set())
        for cell, kind in observation.visible.items():
            if cell not in self.cells:
                self.events.append({"type": "observed", "cell": cell, "kind": kind})
            self.cells[cell] = kind
            if kind == "wall" and cell in self.edges:
                for q in self.edges[cell]:
                    self.edges[q].discard(cell)
                del self.edges[cell]
        for cell, kind in observation.visible.items():
            if kind not in ("floor", "goal"):
                continue
            self.edges.setdefault(cell, set())
            for q in neighbors(cell):
                if self.cells.get(q) in ("floor", "goal"):
                    self.edges[cell].add(q)
                    self.edges.setdefault(q, set()).add(cell)

    def route(self, start: Cell, destination: Cell) -> list[Cell]:
        if destination == start:
            return []
        previous: dict[Cell, Cell | None] = {start: None}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for q in sorted(self.edges.get(p, ())):
                if q in previous:
                    continue
                previous[q] = p
                if q == destination:
                    path = [q]
                    while previous[path[-1]] != start:
                        path.append(previous[path[-1]])
                    return list(reversed(path))
                queue.append(q)
        return []


class Explorer:
    def __init__(self, sensor_range: int = 3, size: int = 21, seed: int = 0,
                 gain_weight: float = 0.03):
        self.sensor_range = sensor_range
        self.size = size  # known arena extent, never its walls or goal position
        self.rng = np.random.default_rng(seed)
        self.memory = PlaceMemory()
        self.target: Cell | None = None
        self.gain_weight = gain_weight

    def _gain(self, p: Cell) -> float:
        # Optimistic visible area. Actual walls may hide some of these cells;
        # observations correct that belief after each move.
        r0, c0 = p
        unseen = sum((r, c) not in self.memory.cells
                     for r in range(r0 - self.sensor_range, r0 + self.sensor_range + 1)
                     for c in range(c0 - self.sensor_range, c0 + self.sensor_range + 1)
                     if 0 <= r < self.size and 0 <= c < self.size
                     and abs(r - r0) + abs(c - c0) <= self.sensor_range)
        unvisited = p not in self.memory.visits
        return unseen + (1.0 if unvisited else 0.0)

    def act(self, obs: Observation) -> Cell | None:
        self.memory.update(obs)
        p = obs.position
        goals = [q for q, kind in self.memory.cells.items() if kind == "goal" and q not in self.memory.visits]
        if goals:
            route = self.memory.route(p, goals[0])
            if route:
                self.target = goals[0]
                return route[0][0] - p[0], route[0][1] - p[1]
        best: tuple[float, int, Cell, list[Cell]] | None = None
        for q, kind in self.memory.cells.items():
            if kind not in ("floor", "goal") or q == p or q in self.memory.visits:
                continue
            route = self.memory.route(p, q)
            if not route:
                continue
            # A frontier is a known passable place not yet visited. Its immediate
            # visit is guaranteed new information; unseen cells beyond walls are
            # only an optimistic tie-breaker and cannot trap the policy.
            information_rate = (1.0 + self.gain_weight * self._gain(q)) / len(route)
            score = -expected_free_energy(0.0, information_rate)
            candidate = (score, -len(route), q, route)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            return None
        self.target = best[2]
        next_cell = best[3][0]
        return next_cell[0] - p[0], next_cell[1] - p[1]


class RandomWalker:
    """Same observation and movement contract; no place memory or route planning."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Observation) -> Cell:
        choices = [d for d in DIRECTIONS
                   if obs.visible.get((obs.position[0] + d[0], obs.position[1] + d[1])) in ("floor", "goal")]
        return choices[int(self.rng.integers(len(choices)))] if choices else DIRECTIONS[int(self.rng.integers(4))]


def run_episode(world: MazeWorld, agent: Explorer | RandomWalker, budget: int = 160,
                stop_at_goal: bool = False) -> dict:
    obs = world.observe()
    first_goal_step = 0 if obs.reached_goal else None
    for _ in range(budget):
        move = agent.act(obs)
        if move is None:
            break
        obs = world.step(move)
        if obs.reached_goal and first_goal_step is None:
            first_goal_step = world.steps
        if obs.reached_goal and stop_at_goal:
            break
    return {"steps": world.steps, "coverage": world.truth()["coverage"],
            "visited": len(world.visited), "reachable": world.truth()["reachable"],
            "found_goal": first_goal_step is not None, "goal_step": first_goal_step,
            "collisions": sum(e["type"] == "collision" for e in world.events)}
