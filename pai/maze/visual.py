"""Camera-only local map with noisy odometry and remembered-view loop closure.

The localizer receives RGB patches and odometry, never world.position or grid.
Color classification is deliberately simple simulator supervision; place matching
is a stored-view nearest-neighbor model, not a learned neural visual encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np

from pai.cognition.experiments import BinaryExperiment, CausalExperimentPlanner
from pai.maze.agent import Explorer
from pai.maze.env import DIRECTIONS, Cell, Observation, VISUAL_COLORS, VisualObservation

LABELS = tuple(VISUAL_COLORS)
PALETTE = np.asarray([VISUAL_COLORS[k] for k in LABELS], dtype=np.float32)
FREE = {"floor", "goal", "landmark_red", "landmark_blue", "landmark_green"}


def decode(image: np.ndarray, classifier=None) -> np.ndarray:
    """Classify each rendered tile by nearest RGB prototype."""
    if image.ndim != 3 or image.shape[0] != image.shape[1] or image.shape[0] % 6:
        raise ValueError("expected square RGB image with 6-pixel tiles")
    n = image.shape[0] // 6
    rgb = image.reshape(n, 6, n, 6, 3).mean((1, 3))
    labels = (classifier.predict_rgb(rgb) if classifier is not None else
              ((rgb[:, :, None, :] - PALETTE) ** 2).sum(-1).argmin(-1))
    return labels.astype(np.uint8)


def _kind(i: int) -> str:
    return LABELS[int(i)]


@dataclass
class VisualLocalizer:
    start: Cell = (1, 1)
    classifier: object | None = None
    position: Cell = field(init=False)
    camera_offset: Cell = (0, 0)
    map_labels: dict[Cell, str] = field(default_factory=dict)
    views: dict[Cell, np.ndarray] = field(default_factory=dict)
    corrections: int = 0
    loop_closures: int = 0
    camera_corrections: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.position = self.start

    def _agreement(self, labels: np.ndarray, center: Cell, offset: Cell) -> tuple[int, int]:
        radius = labels.shape[0] // 2
        matches = comparisons = 0
        for rr, cc in np.ndindex(labels.shape):
            kind = _kind(labels[rr, cc])
            if kind == "unknown":
                continue
            p = (center[0] + rr - radius + offset[0], center[1] + cc - radius + offset[1])
            old = self.map_labels.get(p)
            if old is not None:
                comparisons += 1
                matches += old == kind
        return matches, comparisons

    def update(self, visual: VisualObservation, expected_move: Cell | None = None) -> Observation:
        labels = decode(visual.image, self.classifier)
        delta = np.asarray(visual.odom_delta)
        predicted = np.asarray(self.position, float) + delta
        expected = None if expected_move is None else (
            self.position if visual.collided else
            (self.position[0] + expected_move[0], self.position[1] + expected_move[1]))
        # A cardinal body skill reports success or collision exactly. Keep that
        # displacement and use image agreement to estimate only the camera offset.
        moves = ((0, 0),) if expected is not None else ((0, 0), *DIRECTIONS)
        candidates = []
        for dr, dc in moves:
            center = expected if expected is not None else (self.position[0] + dr, self.position[1] + dc)
            for offset in ((0, 0), *DIRECTIONS):
                matches, comparisons = self._agreement(labels, center, offset)
                disagreement = (comparisons - matches) / max(comparisons, 1)
                motion_error = float(np.linalg.norm(np.asarray(center) - predicted) ** 2)
                skill_error = 0.0 if expected is None or center == expected else 3.0
                offset_change = abs(offset[0] - self.camera_offset[0]) + abs(offset[1] - self.camera_offset[1])
                score = 4.0 * disagreement + 0.8 * motion_error + 0.35 * offset_change + skill_error
                candidates.append((score, -comparisons, center, offset))
        _, _, chosen, offset = min(candidates)
        if chosen != (int(round(predicted[0])), int(round(predicted[1]))):
            self.corrections += 1
        if chosen in self.views and self.views[chosen].shape == labels.shape:
            old = self.views[chosen]
            shared = (old != LABELS.index("unknown")) & (labels != LABELS.index("unknown"))
            if shared.sum() >= 5 and float((old[shared] == labels[shared]).mean()) >= 0.8:
                self.loop_closures += 1
                self.events.append({"type": "loop_closure", "position": chosen})
        if offset != self.camera_offset:
            self.camera_corrections += 1
            self.events.append({"type": "camera_recalibrated", "offset": offset})
        self.position, self.camera_offset = chosen, offset
        self.views[chosen] = labels.copy()
        radius = labels.shape[0] // 2
        visible = {}
        for rr, cc in np.ndindex(labels.shape):
            kind = _kind(labels[rr, cc])
            if kind == "unknown":
                continue
            cell = (chosen[0] + rr - radius + offset[0], chosen[1] + cc - radius + offset[1])
            self.map_labels[cell] = kind
            visible[cell] = "floor" if kind in FREE and kind != "goal" else kind
        return Observation(chosen, visible, visual.collided, visual.reached_goal)


class VisualExplorer:
    """The existing L1 frontier policy fed by a camera-based place estimate."""

    def __init__(self, size: int = 21, sensor_range: int = 3, seed: int = 0,
                 gain_weight: float = 0.03, classifier=None,
                 enable_body_recovery: bool = True):
        self.localizer = VisualLocalizer(classifier=classifier)
        self.planner = Explorer(sensor_range, size, seed, gain_weight)
        self.last_move: Cell | None = None
        self.blocked_cells: set[Cell] = set()
        # Discrete body outcomes provide an independent topological trace. It is
        # useful when a shifted camera makes every remaining frontier look closed.
        self.body_position: Cell = (1, 1)
        self.body_visited: set[Cell] = {(1, 1)}
        self.body_edges: dict[Cell, set[Cell]] = {(1, 1): set()}
        self.body_attempted: set[tuple[Cell, Cell]] = set()
        self.body_recovery = False
        self.enable_body_recovery = enable_body_recovery
        self.last_probe: BinaryExperiment | None = None
        self.probe_reports: list[dict] = []

    @staticmethod
    def _passage_probe(label: str | None, cell: Cell, move: Cell) -> BinaryExperiment:
        if label == "wall":
            likelihood = {"passage_blocked": 0.05, "camera_shift": 0.8}
        elif label in FREE:
            likelihood = {"passage_blocked": 0.65, "camera_shift": 0.9}
        else:
            likelihood = {"passage_blocked": 0.5, "camera_shift": 0.5}
        return BinaryExperiment(f"move:{cell}:{move}", likelihood)

    @property
    def memory(self):
        return self.planner.memory

    def _body_frontier_move(self) -> Cell | None:
        # Visit confirmed free, unseen neighbors first. When the camera is
        # ambiguous, still probe every remaining physical frontier eventually.
        queue = deque([self.body_position])
        parent: dict[Cell, Cell | None] = {self.body_position: None}
        distance = {self.body_position: 0}
        best: tuple[tuple[int, int, Cell, Cell], Cell, Cell] | None = None
        experiment_planner = CausalExperimentPlanner({"passage_blocked": 0.5,
                                                       "camera_shift": 0.5})
        while queue:
            cell = queue.popleft()
            for move in DIRECTIONS:
                neighbor = (cell[0] + move[0], cell[1] + move[1])
                if not (0 <= neighbor[0] < self.planner.size and
                        0 <= neighbor[1] < self.planner.size) or (cell, move) in self.body_attempted:
                    continue
                label = self.localizer.map_labels.get(neighbor)
                rank = (0 if neighbor not in self.body_visited and label in FREE else
                        1 if neighbor not in self.body_visited and label != "wall" else
                        2 if neighbor not in self.body_visited else 3)
                probe = self._passage_probe(label, cell, move)
                candidate = (distance[cell] + (0, 4, 12, 20)[rank]
                             - 0.25 * experiment_planner.information_gain(probe),
                             distance[cell], cell, move)
                if best is None or candidate < best[0]:
                    best = candidate, cell, move
            for neighbor in self.body_edges.get(cell, ()):
                if neighbor not in parent:
                    parent[neighbor] = cell
                    distance[neighbor] = distance[cell] + 1
                    queue.append(neighbor)
        if best is None:
            return None
        _, cell, move = best
        if cell == self.body_position:
            neighbor = (cell[0] + move[0], cell[1] + move[1])
            self.last_probe = self._passage_probe(self.localizer.map_labels.get(neighbor), cell, move)
            return move
        self.last_probe = None
        while parent[cell] != self.body_position:
            cell = parent[cell]
        return cell[0] - self.body_position[0], cell[1] - self.body_position[1]

    def act(self, visual: VisualObservation) -> Cell | None:
        if self.last_probe is not None and not visual.body_fell:
            planner = CausalExperimentPlanner({"passage_blocked": 0.5,
                                               "camera_shift": 0.5})
            self.probe_reports.append({"probe": self.last_probe.name,
                                       "succeeded": not visual.collided,
                                       "posterior": planner.posterior(self.last_probe,
                                                                      not visual.collided)})
            self.last_probe = None
        if self.last_move is not None and not visual.body_fell:
            previous = self.body_position
            self.body_attempted.add((previous, self.last_move))
            if not visual.collided:
                self.body_position = (previous[0] + self.last_move[0],
                                      previous[1] + self.last_move[1])
                reverse = (-self.last_move[0], -self.last_move[1])
                self.body_attempted.add((self.body_position, reverse))
                self.body_visited.add(self.body_position)
                self.body_edges.setdefault(previous, set()).add(self.body_position)
                self.body_edges.setdefault(self.body_position, set()).add(previous)
        obs = self.localizer.update(visual, self.last_move)
        if visual.collided and not visual.body_fell and self.last_move is not None:
            # The body skill's contact outcome overrides an old image that still
            # paints the attempted passage as free after a new blockage.
            blocked = (obs.position[0] + self.last_move[0],
                       obs.position[1] + self.last_move[1])
            self.blocked_cells.add(blocked)
        if self.blocked_cells:
            visible = dict(obs.visible)
            for blocked in self.blocked_cells:
                visible[blocked] = "wall"
                self.localizer.map_labels[blocked] = "wall"
            obs = Observation(obs.position, visible, visual.collided, obs.reached_goal)
        planned = self.planner.act(obs)
        if planned is None and self.enable_body_recovery:
            self.body_recovery = True
        if self.body_recovery:
            self.last_move = self._body_frontier_move()
            return self.last_move
        self.last_move = planned
        return self.last_move
