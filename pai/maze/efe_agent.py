"""Active Inference and baseline explorers for procedural mazes.

Implements:
1. ActiveInferenceExplorer: Action selection by minimizing Expected Free Energy (EFE)
   combining expected epistemic information gain, pragmatic goal attraction, traversal effort,
   and collision risk via pai.planning.objective.expected_free_energy.
2. NearestFrontierExplorer: Classical Yamauchi frontier exploration minimizing route length.
3. InfoGainFrontierExplorer: Classical greedy entropy/unseen area ratio heuristic.
4. RandomWalker: Unbiased local walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from pai.maze.env import Cell, DIRECTIONS, Observation, neighbors
from pai.maze.map_model import ObservationModel, ProbabilisticMap, SpatialPrior
from pai.planning.objective import expected_free_energy


class ActiveInferenceExplorer:
    """Active inference explorer using a probabilistic generative map and EFE planning."""

    def __init__(self, sensor_range: int = 3, size: int = 21, seed: int = 0,
                 prior: SpatialPrior | None = None,
                 obs_model: ObservationModel | None = None,
                 epistemic_weight: float = 1.0,
                 effort_weight: float = 0.5,
                 risk_weight: float = 1.0,
                 goal_weight: float = 5.0):
        self.sensor_range = sensor_range
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.obs_model = obs_model or ObservationModel()
        self.map = ProbabilisticMap(size=size, sensor_range=sensor_range,
                                    prior=prior, obs_model=self.obs_model)
        self.epistemic_weight = epistemic_weight
        self.effort_weight = effort_weight
        self.risk_weight = risk_weight
        self.goal_weight = goal_weight
        self.target: Cell | None = None
        self.last_efe_score: float | None = None
        self.last_epistemic: float | None = None
        self.last_pragmatic: float | None = None
        self.last_effort: float | None = None

    @property
    def memory(self) -> ProbabilisticMap:
        """Backward-compatible access to place memory / map."""
        return self.map

    def _locate_goal_candidate(self) -> Cell | None:
        """Find the cell with highest goal belief if above threshold."""
        best_cell = None
        best_prob = 0.5
        for r in range(self.size):
            for c in range(self.size):
                p = self.map.beliefs[r, c, 2]  # P(goal)
                if p > best_prob and (r, c) not in self.map.visits:
                    best_prob = p
                    best_cell = (r, c)
        return best_cell

    def _evaluate_candidate_efe(self, current: Cell, target: Cell,
                                route: list[Cell], goal_candidate: Cell | None) -> tuple[float, float, float, float]:
        """Compute Expected Free Energy for a candidate destination.

        EFE = pragmatic - epistemic_weight * epistemic + effort_weight * effort
        Returns (efe, epistemic, pragmatic, effort).
        """
        route_len = len(route)
        effort = float(route_len)

        # Epistemic value: expected information gain (entropy reduction) at candidate vantage point
        epistemic = float(self.map.expected_information_gain(target))

        # Risk cost: sum of wall probabilities along path
        risk = float(sum(self.map.beliefs[s[0], s[1], 0] for s in route))

        # Pragmatic cost:
        if goal_candidate is not None:
            # Goal is localized: pragmatic cost is distance from target to goal
            dist_to_goal = abs(target[0] - goal_candidate[0]) + abs(target[1] - goal_candidate[1])
            goal_cost = float(dist_to_goal * self.goal_weight)
            if target == goal_candidate:
                goal_cost -= 10.0  # High preference for arriving at the goal
        else:
            # Goal not yet localized: preference for reducing goal uncertainty
            # Expected distance to unexplored cells + risk
            goal_cost = 1.0  # Baseline positive pragmatic term

        pragmatic = float(goal_cost + self.risk_weight * risk)

        # Calculate EFE via shared objective
        efe = float(expected_free_energy(
            pragmatic=pragmatic,
            epistemic=epistemic,
            effort=effort,
            smoothness=0.0,
            epistemic_weight=self.epistemic_weight,
            effort_weight=self.effort_weight,
        ))
        return efe, epistemic, pragmatic, effort

    def act(self, obs: Observation) -> Cell | None:
        self.map.update(obs)
        p = obs.position

        # Check for confirmed or believed visible goal
        goal_cell = self._locate_goal_candidate()
        if goal_cell is not None:
            route = self.map.route(p, goal_cell)
            if route:
                self.target = goal_cell
                next_c = route[0]
                return next_c[0] - p[0], next_c[1] - p[1]

        # Identify candidate destinations (frontiers and unvisited passable cells)
        candidates = self.map.frontiers(p)
        if not candidates:
            # Fallback to any passable cell with remaining uncertainty
            candidates = [c for c in self.map.edges if c != p and (c not in self.map.visits or self.map.cell_entropy(c) > 0.05)]

        if not candidates:
            return None

        best_choice: tuple[float, int, Cell, list[Cell], float, float, float] | None = None
        for q in candidates:
            route = self.map.route(p, q)
            if not route:
                continue
            efe, epi, prag, eff = self._evaluate_candidate_efe(p, q, route, goal_cell)
            # We want to MINIMIZE EFE (lower free energy is better)
            # Tie breaker: shorter route, then cell coords
            item = (efe, len(route), q, route, epi, prag, eff)
            if best_choice is None or item[:2] < best_choice[:2]:
                best_choice = item

        if best_choice is None:
            return None

        self.last_efe_score, _, self.target, best_route, self.last_epistemic, self.last_pragmatic, self.last_effort = best_choice
        next_step = best_route[0]
        return next_step[0] - p[0], next_step[1] - p[1]


class NearestFrontierExplorer:
    """Classical Yamauchi frontier exploration baseline.

    Selects the reachable frontier that minimizes topological route distance.
    No epistemic gain calculation, no active inference.
    """

    def __init__(self, sensor_range: int = 3, size: int = 21, seed: int = 0):
        self.sensor_range = sensor_range
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.map = ProbabilisticMap(size=size, sensor_range=sensor_range)
        self.target: Cell | None = None

    @property
    def memory(self) -> ProbabilisticMap:
        return self.map

    def act(self, obs: Observation) -> Cell | None:
        self.map.update(obs)
        p = obs.position

        # If goal is visible and unvisited, go directly
        for c, kind in self.map.cells.items():
            if kind == "goal" and c not in self.map.visits:
                route = self.map.route(p, c)
                if route:
                    self.target = c
                    return route[0][0] - p[0], route[0][1] - p[1]

        candidates = self.map.frontiers(p)
        if not candidates:
            candidates = [c for c in self.map.edges if c != p and c not in self.map.visits]
        if not candidates:
            return None

        best: tuple[int, Cell, list[Cell]] | None = None
        for q in candidates:
            route = self.map.route(p, q)
            if not route:
                continue
            # Pick shortest route length
            choice = (len(route), q, route)
            if best is None or choice[:2] < best[:2]:
                best = choice

        if best is None:
            return None

        self.target = best[1]
        next_c = best[2][0]
        return next_c[0] - p[0], next_c[1] - p[1]


class InfoGainFrontierExplorer:
    """Information-gain frontier baseline (greedy unseen area / entropy ratio).

    Ranks frontiers by information gain / route length without pragmatic value.
    """

    def __init__(self, sensor_range: int = 3, size: int = 21, seed: int = 0,
                 gain_weight: float = 0.03):
        self.sensor_range = sensor_range
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.map = ProbabilisticMap(size=size, sensor_range=sensor_range)
        self.gain_weight = gain_weight
        self.target: Cell | None = None

    @property
    def memory(self) -> ProbabilisticMap:
        return self.map

    def act(self, obs: Observation) -> Cell | None:
        self.map.update(obs)
        p = obs.position

        for c, kind in self.map.cells.items():
            if kind == "goal" and c not in self.map.visits:
                route = self.map.route(p, c)
                if route:
                    self.target = c
                    return route[0][0] - p[0], route[0][1] - p[1]

        candidates = self.map.frontiers(p)
        if not candidates:
            candidates = [c for c in self.map.edges if c != p and c not in self.map.visits]
        if not candidates:
            return None

        best: tuple[float, int, Cell, list[Cell]] | None = None
        for q in candidates:
            route = self.map.route(p, q)
            if not route:
                continue
            # Information rate = (1.0 + weight * expected_info_gain) / route_len
            gain = self.map.expected_information_gain(q)
            rate = (1.0 + self.gain_weight * gain) / len(route)
            # Maximize rate
            choice = (-rate, len(route), q, route)
            if best is None or choice[:2] < best[:2]:
                best = (rate, len(route), q, route)

        if best is None:
            return None

        self.target = best[2]
        next_c = best[3][0]
        return next_c[0] - p[0], next_c[1] - p[1]
