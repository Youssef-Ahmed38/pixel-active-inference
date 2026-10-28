"""Probabilistic map and learned observation model for Active Inference in mazes.

Maintains Bayesian categorical beliefs over cell occupancy/types (wall, floor, goal),
computes exact Shannon entropy and expected information gain under a learned observation
model, and tracks topological connectivity for route planning.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from pai.maze.env import Cell, DIRECTIONS, Observation, neighbors

KINDS: tuple[str, ...] = ("wall", "floor", "goal")
KIND_TO_IDX: dict[str, int] = {k: i for i, k in enumerate(KINDS)}
IDX_TO_KIND: dict[int, str] = {i: k for i, k in enumerate(KINDS)}


@dataclass
class ObservationModel:
    """Categorical observation model P(Obs = o | True = m).

    matrix[o_idx, m_idx] = P(Obs = o | True = m)
    Columns sum to 1.0 (valid probability distribution over observations for each true state).
    """

    matrix: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.96, 0.02, 0.02],  # P(Obs=wall | m=wall, floor, goal)
            [0.03, 0.96, 0.02],  # P(Obs=floor | m=wall, floor, goal)
            [0.01, 0.02, 0.96],  # P(Obs=goal | m=wall, floor, goal)
        ], dtype=float)
    )

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        # Normalize columns so P(o|m) sums to 1
        col_sums = self.matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        self.matrix /= col_sums

    @classmethod
    def fit(cls, observations: Sequence[str], actuals: Sequence[str], alpha: float = 1.0) -> ObservationModel:
        """Fit P(Obs | True) from empirical counts with Laplace smoothing."""
        counts = np.full((3, 3), alpha, dtype=float)
        for obs, act in zip(observations, actuals):
            if obs in KIND_TO_IDX and act in KIND_TO_IDX:
                counts[KIND_TO_IDX[obs], KIND_TO_IDX[act]] += 1.0
        return cls(matrix=counts)

    def obs_probabilities(self, cell_belief: np.ndarray) -> np.ndarray:
        """P(O = o) = sum_m P(O = o | M = m) P(M = m). Vector of length 3."""
        return self.matrix @ cell_belief

    def posterior(self, prior_belief: np.ndarray, obs_kind: str) -> np.ndarray:
        """Bayes update: P(M = m | O = o) proportional to P(O = o | M = m) * P(M = m)."""
        o_idx = KIND_TO_IDX.get(obs_kind, 1)
        likelihood = self.matrix[o_idx, :]
        unnorm = likelihood * prior_belief
        total = unnorm.sum()
        if total <= 1e-12:
            return prior_belief.copy()
        return unnorm / total

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"matrix": self.matrix.tolist(), "kinds": list(KINDS)}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> ObservationModel:
        data = json.loads(Path(path).read_text())
        return cls(matrix=np.array(data["matrix"], dtype=float))


@dataclass
class SpatialPrior:
    """Prior belief over cell types across the maze."""

    size: int = 21
    # Prior shape: (size, size, 3)
    priors: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.priors is None:
            # Default uninformative prior: borders are walls with high probability.
            # Interior cells have prior: ~55% wall, ~44% floor, ~1% goal.
            p = np.zeros((self.size, self.size, 3), dtype=float)
            p[:, :] = [0.55, 0.44, 0.01]
            p[0, :, :] = [0.999, 0.0009, 0.0001]
            p[self.size - 1, :, :] = [0.999, 0.0009, 0.0001]
            p[:, 0, :] = [0.999, 0.0009, 0.0001]
            p[:, self.size - 1, :] = [0.999, 0.0009, 0.0001]
            self.priors = p
        else:
            self.priors = np.asarray(self.priors, dtype=float)

    @classmethod
    def fit_from_mazes(cls, maze_grids: Sequence[np.ndarray], goal_positions: Sequence[Cell],
                       size: int = 21, alpha: float = 1.0) -> SpatialPrior:
        """Learn empirical spatial prior P0(M_rc) from training maze grids."""
        counts = np.full((size, size, 3), alpha, dtype=float)
        for grid, goal in zip(maze_grids, goal_positions):
            for r in range(size):
                for c in range(size):
                    if (r, c) == goal:
                        counts[r, c, KIND_TO_IDX["goal"]] += 1.0
                    elif grid[r, c] == 1:
                        counts[r, c, KIND_TO_IDX["wall"]] += 1.0
                    else:
                        counts[r, c, KIND_TO_IDX["floor"]] += 1.0
        sums = counts.sum(axis=-1, keepdims=True)
        sums[sums == 0] = 1.0
        return cls(size=size, priors=counts / sums)

    @classmethod
    def fit_from_beliefs(cls, maps: Sequence[np.ndarray], observed: Sequence[set], size: int = 21,
                         alpha: float = 2.0, base: "SpatialPrior | None" = None) -> SpatialPrior:
        """Learn P0(M_rc) from the agent's own end-of-practice maps, without true layouts.

        Each practice map contributes its posterior belief at the cells the agent observed (soft
        counts); unobserved cells contribute nothing. ``alpha`` pseudo-counts of the generic prior
        keep cells the agent rarely saw close to it.
        """
        base = base or cls(size=size)
        counts = alpha * base.priors.copy()
        for beliefs, seen in zip(maps, observed):
            for (r, c) in seen:
                counts[r, c] += beliefs[r, c]
        return cls(size=size, priors=counts / counts.sum(-1, keepdims=True))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"size": self.size, "priors": self.priors.tolist()}
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path | str) -> SpatialPrior:
        data = json.loads(Path(path).read_text())
        return cls(size=data["size"], priors=np.array(data["priors"], dtype=float))


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in bits for a 1D probability vector."""
    p_nz = p[p > 1e-12]
    return float(-np.sum(p_nz * np.log2(p_nz)))


class ProbabilisticMap:
    """Bayesian map storing cell belief distributions and topological routes."""

    def __init__(self, size: int = 21, sensor_range: int = 3,
                 prior: SpatialPrior | None = None,
                 obs_model: ObservationModel | None = None):
        self.size = size
        self.sensor_range = sensor_range
        self.obs_model = obs_model or ObservationModel()
        prior_obj = prior or SpatialPrior(size=size)
        self.beliefs = prior_obj.priors.copy()  # shape (size, size, 3)
        self.observed_cells: set[Cell] = set()
        self.visits: dict[Cell, int] = field(default_factory=dict)
        self.visits = {}
        self.edges: dict[Cell, set[Cell]] = {}
        self.events: list[dict] = []
        # Evidence weight of a repeat look at a cell from a viewpoint already used (1.0: looks are
        # independent; 0.0: a repeat carries no new information). Learned from practice, see
        # pai/maze/self_calibration.py:repeat_look_weight.
        self.repeat_weight = 1.0
        self.looked: set[tuple[Cell, Cell]] = set()
        self.touched: set[Cell] = set()   # cells bumped into: walls known by touch

    @property
    def cells(self) -> dict[Cell, str]:
        """Backward-compatible view returning the most likely label for observed cells."""
        res = {}
        for c in self.observed_cells:
            best_idx = int(np.argmax(self.beliefs[c[0], c[1]]))
            res[c] = IDX_TO_KIND[best_idx]
        return res

    def get_belief(self, cell: Cell) -> np.ndarray:
        return self.beliefs[cell[0], cell[1]]

    def get_kind(self, cell: Cell) -> str:
        return IDX_TO_KIND[int(np.argmax(self.beliefs[cell[0], cell[1]]))]

    def is_passable(self, cell: Cell, threshold: float = 0.5) -> bool:
        """Cell is believed passable if P(floor) + P(goal) >= threshold."""
        b = self.beliefs[cell[0], cell[1]]
        return float(b[1] + b[2]) >= threshold

    def cell_entropy(self, cell: Cell) -> float:
        return shannon_entropy(self.beliefs[cell[0], cell[1]])

    def total_entropy(self) -> float:
        """Total map uncertainty in bits across all cells."""
        ent = 0.0
        for r in range(self.size):
            for c in range(self.size):
                ent += self.cell_entropy((r, c))
        return ent

    def update(self, obs: Observation) -> None:
        """Bayesian update given local observation."""
        p = obs.position
        self.visits[p] = self.visits.get(p, 0) + 1
        self.edges.setdefault(p, set())

        # Update beliefs for all visible cells
        for cell, obs_kind in obs.visible.items():
            r, c = cell
            if not (0 <= r < self.size and 0 <= c < self.size) or cell == p:
                continue
            if cell not in self.observed_cells:
                self.observed_cells.add(cell)
                self.events.append({"type": "observed", "cell": cell, "kind": obs_kind})
            old_b = self.beliefs[r, c]
            if (cell, p) in self.looked and self.repeat_weight != 1.0 and cell != getattr(obs, "touched", None):
                likelihood = self.obs_model.matrix[KIND_TO_IDX.get(obs_kind, 1), :] ** self.repeat_weight
                new_b = likelihood * old_b / max(float(likelihood @ old_b), 1e-12)
            else:
                new_b = self.obs_model.posterior(old_b, obs_kind)
            self.looked.add((cell, p))
            self.beliefs[r, c] = new_b

        if getattr(obs, "touched", None) is not None:
            self.touched.add(obs.touched)
        # Body evidence: the cell the agent stands on is open (the goal if it reports reaching it),
        # whatever the camera says about it.
        r0, c0 = p
        self.beliefs[r0, c0] = (0.0, 0.0, 1.0) if obs.reached_goal else (1e-6, 1.0 - 2e-6, 1e-6)
        self.observed_cells.add(p)

        # If a collision occurred on an attempted move, update that cell toward wall
        if obs.collided and hasattr(obs, "attempted_target"):
            tgt = getattr(obs, "attempted_target")
            if tgt and 0 <= tgt[0] < self.size and 0 <= tgt[1] < self.size:
                # Strong likelihood of wall upon collision
                self.beliefs[tgt[0], tgt[1]] = self.obs_model.posterior(
                    self.beliefs[tgt[0], tgt[1]], "wall"
                )

        # Update graph edges for passable cells
        for cell in obs.visible:
            if not self.is_passable(cell):
                if cell in self.edges:
                    for q in list(self.edges[cell]):
                        self.edges[q].discard(cell)
                    del self.edges[cell]
                continue
            self.edges.setdefault(cell, set())
            for q in neighbors(cell):
                if 0 <= q[0] < self.size and 0 <= q[1] < self.size and self.is_passable(q):
                    self.edges[cell].add(q)
                    self.edges.setdefault(q, set()).add(cell)

    def expected_cell_information_gain(self, cell: Cell) -> float:
        """Mutual information I(M_c; O_c) = H(M_c) - E_{O_c}[H(M_c | O_c)]."""
        if cell not in self.observed_cells:
            # An unobserved cell still has its prior belief, which can differ from cell to cell (a
            # learned spatial prior). Cached per cell; the first version cached one value for all
            # unobserved cells, so a spatial prior could not change the information values.
            cache = self.__dict__.setdefault("_unobserved_ig", {})
            if cell not in cache:
                prior_b = self.beliefs[cell[0], cell[1]]
                h_prior = shannon_entropy(prior_b)
                p_obs = self.obs_model.obs_probabilities(prior_b)
                expected_post = 0.0
                for o_idx in range(3):
                    if p_obs[o_idx] > 1e-12:
                        post_b = self.obs_model.posterior(prior_b, IDX_TO_KIND[o_idx])
                        expected_post += float(p_obs[o_idx] * shannon_entropy(post_b))
                cache[cell] = max(0.0, h_prior - expected_post)
            return cache[cell]

        prior_b = self.beliefs[cell[0], cell[1]]
        h_prior = shannon_entropy(prior_b)
        if h_prior < 1e-4:
            return 0.0

        p_obs = self.obs_model.obs_probabilities(prior_b)
        expected_post_entropy = 0.0
        for o_idx in range(3):
            if p_obs[o_idx] > 1e-12:
                post_b = self.obs_model.posterior(prior_b, IDX_TO_KIND[o_idx])
                expected_post_entropy += float(p_obs[o_idx] * shannon_entropy(post_b))

        return float(max(0.0, h_prior - expected_post_entropy))

    def visible_cells_from(self, vantage: Cell) -> list[Cell]:
        """Estimated line-of-sight cells from a vantage point under current map beliefs."""
        r0, c0 = vantage
        visible = []
        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                if abs(dr) + abs(dc) > self.sensor_range:
                    continue
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < self.size and 0 <= c < self.size):
                    continue
                # Line of sight check: if line intersects a cell believed to be wall, cut off
                dist = max(abs(dr), abs(dc))
                blocked = False
                for k in range(1, dist):
                    kr = int(round(r0 + dr * k / dist))
                    kc = int(round(c0 + dc * k / dist))
                    if 0 <= kr < self.size and 0 <= kc < self.size:
                        if self.beliefs[kr, kc, 0] > 0.8:
                            blocked = True
                            break
                if not blocked:
                    visible.append((r, c))
        return visible

    def expected_information_gain(self, vantage: Cell) -> float:
        """Total expected information gain across visible cells from vantage point."""
        vis = self.visible_cells_from(vantage)
        return float(sum(self.expected_cell_information_gain(c) for c in vis))

    def route(self, start: Cell, destination: Cell) -> list[Cell]:
        """Shortest path through believed passable cells via BFS."""
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

    def frontiers(self, current: Cell) -> list[Cell]:
        """Frontiers: believed passable cells that have not been visited or border unobserved cells."""
        unvisited = [cell for cell in self.edges if cell != current and cell not in self.visits]
        if unvisited:
            return unvisited

        frontiers = []
        for cell in self.edges:
            if cell == current:
                continue
            for dr, dc in DIRECTIONS:
                nr, nc = cell[0] + dr, cell[1] + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.observed_cells:
                    frontiers.append(cell)
                    break
        return frontiers
