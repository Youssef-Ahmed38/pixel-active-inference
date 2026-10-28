"""Procedural corridor maze and a simple mobile body.

Only ``observe`` and ``step`` are available to agents. ``grid`` and ``truth`` are
evaluation-only state; the explorer does not read them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

Cell = tuple[int, int]
DIRECTIONS: tuple[Cell, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
VISUAL_COLORS = {
    "unknown": (20, 24, 30), "wall": (70, 75, 85), "floor": (205, 205, 195),
    "goal": (245, 190, 35), "landmark_red": (205, 55, 55),
    "landmark_blue": (55, 105, 220), "landmark_green": (45, 170, 100),
}


def neighbors(p: Cell):
    for dr, dc in DIRECTIONS:
        yield p[0] + dr, p[1] + dc


@dataclass(frozen=True)
class Observation:
    position: Cell             # odometry in the map frame, supplied by L0
    visible: dict[Cell, str]  # wall, floor, or goal; no cells behind a wall
    collided: bool
    reached_goal: bool


@dataclass(frozen=True)
class VisualObservation:
    image: np.ndarray       # local camera image, no absolute coordinates
    odom_delta: tuple[float, float]  # noisy body displacement since the previous observation
    collided: bool
    reached_goal: bool
    body_fell: bool = False


class MazeWorld:
    """Odd-sized DFS maze with loops, landmarks and a hidden goal cell."""

    def __init__(self, seed: int = 0, size: int = 21, sensor_range: int = 3,
                 loop_probability: float = 0.08, appearance_jitter: float = 0.0,
                 appearance_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 generator: str = "dfs", rooms: int = 0):
        """``generator="dfs"`` (default): long winding corridors with few junctions.
        ``generator="prim"``: randomized Prim, many short branches and junctions. ``rooms`` opens
        that many 3x3 open areas. Defaults reproduce every maze used before 2026-10-28."""
        if size < 7 or size % 2 != 1:
            raise ValueError("size must be odd and at least 7")
        if generator not in ("dfs", "prim"):
            raise ValueError("generator must be 'dfs' or 'prim'")
        self.rng = np.random.default_rng(seed)
        self.size, self.sensor_range = size, sensor_range
        self.grid = np.ones((size, size), dtype=np.uint8)
        self.start: Cell = (1, 1)
        stack = [self.start]
        seen = {self.start}
        self.grid[self.start] = 0
        if generator == "prim":
            # grow from a random frontier cell each time (not the newest): many short branches
            frontier = [(self.start, d) for d in DIRECTIONS]
            while frontier:
                (r, c), (dr, dc) = frontier.pop(int(self.rng.integers(len(frontier))))
                nr, nc = r + 2 * dr, c + 2 * dc
                if not (1 <= nr < size - 1 and 1 <= nc < size - 1) or (nr, nc) in seen:
                    continue
                self.grid[r + dr, c + dc] = self.grid[nr, nc] = 0
                seen.add((nr, nc))
                frontier.extend(((nr, nc), d) for d in DIRECTIONS)
            stack = []
        while stack:
            r, c = stack[-1]
            choices = [(r + 2 * dr, c + 2 * dc, dr, dc) for dr, dc in DIRECTIONS
                       if 1 <= r + 2 * dr < size - 1 and 1 <= c + 2 * dc < size - 1
                       and (r + 2 * dr, c + 2 * dc) not in seen]
            if not choices:
                stack.pop()
                continue
            nr, nc, dr, dc = choices[int(self.rng.integers(len(choices)))]
            self.grid[r + dr, c + dc] = self.grid[nr, nc] = 0
            seen.add((nr, nc))
            stack.append((nr, nc))
        # Open some cross-links: the policy must choose among routes, rather than
        # merely follow one long corridor.
        for r in range(2, size - 2):
            for c in range(2, size - 2):
                if self.grid[r, c] and self.grid[r - 1, c] == self.grid[r + 1, c] == 0 \
                        and self.rng.random() < loop_probability:
                    self.grid[r, c] = 0
                elif self.grid[r, c] and self.grid[r, c - 1] == self.grid[r, c + 1] == 0 \
                        and self.rng.random() < loop_probability:
                    self.grid[r, c] = 0
        for _ in range(rooms):
            # a 3x3 open area centred on a random cell-centre (odd coordinates)
            rr = 2 + 2 * int(self.rng.integers(max(1, (size - 5) // 2)))
            cc = 2 + 2 * int(self.rng.integers(max(1, (size - 5) // 2)))
            self.grid[rr - 1:rr + 2, cc - 1:cc + 2] = 0
        floors = [tuple(map(int, p)) for p in np.argwhere(self.grid == 0) if tuple(p) != self.start]
        self.goal: Cell = floors[int(self.rng.integers(len(floors)))]
        landmark_cells = [p for p in floors if p != self.goal]
        picks = self.rng.choice(len(landmark_cells), size=min(9, len(landmark_cells)), replace=False)
        colors = ("landmark_red", "landmark_blue", "landmark_green")
        self.landmarks = {landmark_cells[int(i)]: colors[j % 3] for j, i in enumerate(picks)}
        self.position = self.start
        self.visited = {self.start}
        self.steps = 0
        self.events: list[dict] = []
        self.camera_offset: Cell = (0, 0)
        self._visual_previous = self.start
        self._visual_rng = np.random.default_rng(seed + 900000)
        tint_rng = np.random.default_rng(seed + 910000)
        self.visual_colors = {kind: tuple(np.clip(np.asarray(rgb, float)
                                                 + tint_rng.normal(0, appearance_jitter, 3)
                                                 + np.asarray(appearance_bias, float),
                                                 0, 255).astype(np.uint8))
                              for kind, rgb in VISUAL_COLORS.items()}

    def _visible(self, target: Cell) -> bool:
        """Ray cast through cell centres; a wall is visible, cells past it are not."""
        r0, c0 = self.position
        r1, c1 = target
        distance = max(abs(r1 - r0), abs(c1 - c0))
        if distance == 0:
            return True
        for k in range(1, distance):
            r = int(round(r0 + (r1 - r0) * k / distance))
            c = int(round(c0 + (c1 - c0) * k / distance))
            if self.grid[r, c]:
                return False
        return True

    def observe(self, collided: bool = False) -> Observation:
        r0, c0 = self.position
        visible: dict[Cell, str] = {}
        for r in range(max(0, r0 - self.sensor_range), min(self.size, r0 + self.sensor_range + 1)):
            for c in range(max(0, c0 - self.sensor_range), min(self.size, c0 + self.sensor_range + 1)):
                if abs(r - r0) + abs(c - c0) > self.sensor_range or not self._visible((r, c)):
                    continue
                visible[(r, c)] = "wall" if self.grid[r, c] else "goal" if (r, c) == self.goal else "floor"
        return Observation(self.position, visible, collided, self.position == self.goal)

    def step(self, direction: Cell) -> Observation:
        if direction not in DIRECTIONS:
            raise ValueError("mobile skill requires one cardinal direction")
        target = self.position[0] + direction[0], self.position[1] + direction[1]
        collision = not (0 <= target[0] < self.size and 0 <= target[1] < self.size) or bool(self.grid[target])
        if not collision:
            self.position = target
            if target not in self.visited:
                self.events.append({"step": self.steps + 1, "type": "place_discovered", "position": target})
            self.visited.add(target)
        else:
            self.events.append({"step": self.steps + 1, "type": "collision", "position": target})
        self.steps += 1
        if self.position == self.goal:
            self.events.append({"step": self.steps, "type": "goal_reached", "position": self.goal})
        return self.observe(collision)

    def visual_observe(self, collided: bool = False, odom_noise: float = 0.18,
                       camera_offset: Cell | None = None,
                       body_fell: bool = False) -> VisualObservation:
        """Render a local RGB camera grid; landmarks have colors but no identity labels."""
        if camera_offset is not None:
            self.camera_offset = camera_offset
        radius, tile = self.sensor_range, 6
        n = 2 * radius + 1
        pixels = np.empty((n, n, 3), dtype=np.uint8)
        visible = self.observe(collided).visible
        for rr in range(n):
            for cc in range(n):
                cell = (self.position[0] + rr - radius + self.camera_offset[0],
                        self.position[1] + cc - radius + self.camera_offset[1])
                kind = visible.get(cell, "unknown")
                if kind == "floor":
                    kind = self.landmarks.get(cell, kind)
                pixels[rr, cc] = self.visual_colors[kind]
        delta = np.asarray(self.position, float) - self._visual_previous
        if self.steps:
            delta += self._visual_rng.normal(0.0, odom_noise, 2)
        self._visual_previous = self.position
        image = np.repeat(np.repeat(pixels, tile, axis=0), tile, axis=1)
        return VisualObservation(image, (float(delta[0]), float(delta[1])), collided,
                                 self.position == self.goal, body_fell)

    def block(self, cell: Cell) -> bool:
        """Introduce a new wall if the remaining floor stays connected. Agent sees only outcome."""
        if cell in (self.position, self.start, self.goal) or self.grid[cell] or cell in self.visited:
            return False
        self.grid[cell] = 1
        if len(self.distances_from(self.start)) != int((self.grid == 0).sum()):
            self.grid[cell] = 0
            return False
        self.events.append({"step": self.steps, "type": "passage_blocked", "position": cell})
        return True

    def shift_camera(self, offset: Cell) -> None:
        self.camera_offset = offset
        self.events.append({"step": self.steps, "type": "camera_shift", "offset": offset})

    def truth(self) -> dict:
        """Evaluation only. Never pass this to a policy."""
        return {"reachable": int((self.grid == 0).sum()), "goal": self.goal,
                "visited": len(self.visited), "coverage": len(self.visited) / int((self.grid == 0).sum())}

    def distances_from(self, start: Cell) -> dict[Cell, int]:
        """Evaluation helper for characterising generated mazes."""
        distances = {start: 0}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            for q in neighbors(p):
                if 0 <= q[0] < self.size and 0 <= q[1] < self.size and not self.grid[q] and q not in distances:
                    distances[q] = distances[p] + 1
                    queue.append(q)
        return distances
