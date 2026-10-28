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


def relook_move(belief_map: ProbabilisticMap, p: Cell, floor: float = 0.05) -> Cell | None:
    """No route to any candidate: step toward the neighbour most likely to be open.

    Under noisy sensing a single misread wall can cut every route; moving (or bumping) gives a new
    look. Returns None only when every neighbour is almost surely a wall.
    """
    best, best_p = None, floor
    for d in DIRECTIONS:
        q = p[0] + d[0], p[1] + d[1]
        if 0 <= q[0] < belief_map.size and 0 <= q[1] < belief_map.size:
            b = belief_map.beliefs[q[0], q[1]]
            if float(b[1] + b[2]) > best_p:
                best, best_p = d, float(b[1] + b[2])
    if best is None and belief_map.repeat_weight < 1.0:
        # Looking again from here cannot help (repeat looks carry little information), so test the
        # untouched neighbour most likely to be open by trying to step into it: touch is reliable.
        best_p = -1.0
        for d in DIRECTIONS:
            q = p[0] + d[0], p[1] + d[1]
            if 0 <= q[0] < belief_map.size and 0 <= q[1] < belief_map.size and q not in belief_map.touched:
                b = belief_map.beliefs[q[0], q[1]]
                if float(b[1] + b[2]) > best_p:
                    best, best_p = d, float(b[1] + b[2])
    return best


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
            return relook_move(self.map, p)

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
            return relook_move(self.map, p)

        self.last_efe_score, _, self.target, best_route, self.last_epistemic, self.last_pragmatic, self.last_effort = best_choice
        next_step = best_route[0]
        return next_step[0] - p[0], next_step[1] - p[1]


class RouteEFEExplorer(ActiveInferenceExplorer):
    """EFE scored over whole routes instead of at the destination only.

    One breadth-first tree over believed-passable cells gives the route to every candidate (the
    same routes as ``ProbabilisticMap.route``). Along a route:

    - epistemic: expected information gain of the cells that come newly into view at each route
      cell (visible from it but not from its predecessor), summed; cells already counted further
      back on the route can be counted again, so this slightly favours winding routes;
    - pragmatic: ``-goal_weight * P(goal lies on the route)`` plus collision risk (sum of P(wall));
    - effort: route length.

    The previous target is kept unless another candidate is better by ``switch_margin``, so the
    agent does not dither between frontiers whose scores change a little at every step.
    """

    def __init__(self, *args, switch_margin: float = 1.0, min_goal_belief: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch_margin = switch_margin
        # None: every unvisited cell is worth a visit (all grid-sensor results). With a long-range
        # sensor (the head camera) many cells are seen but not stood on; a visit is then worth it only
        # if the agent's own belief that the goal is there exceeds this, or the route brings information.
        self.min_goal_belief = min_goal_belief

    def _tree(self, start: Cell) -> tuple[dict[Cell, Cell], list[Cell]]:
        parent: dict[Cell, Cell] = {start: start}
        order: list[Cell] = []
        queue = [start]
        for p in queue:
            for q in sorted(self.map.edges.get(p, ())):
                if q not in parent:
                    parent[q] = p
                    order.append(q)
                    queue.append(q)
        return parent, order

    def _touch_move(self, p: Cell, parent: dict[Cell, Cell], order: list[Cell],
                    depth: dict[Cell, int]) -> Cell | None:
        """Touch as an epistemic action, used when looking can no longer change the map.

        Stepping into a cell settles for certain whether it is open, so its expected information
        gain is the entropy of P(open). Score each untouched cell next to the reachable region by
        EFE = effort_weight * (walk + 1) - epistemic_weight * H(open), and walk to the best one.
        """
        kept = getattr(self, "_touch_plan", None)
        if kept is not None:
            v, d = kept
            q = v[0] + d[0], v[1] + d[1]
            if (v == p or v in parent) and q not in self.map.touched and q not in parent:
                return self._walk_to_touch(p, parent, v, d)   # finish the chosen test first
        best = None
        for v in [p] + order:
            for d in DIRECTIONS:
                q = v[0] + d[0], v[1] + d[1]
                if q in parent or q in self.map.touched or not (0 <= q[0] < self.size and 0 <= q[1] < self.size):
                    continue
                b = self.map.beliefs[q[0], q[1]]
                p_open = float(b[1] + b[2])
                if not 1e-3 < p_open < 1.0 - 1e-3:
                    continue
                h = -(p_open * np.log2(p_open) + (1 - p_open) * np.log2(1 - p_open))
                efe = self.effort_weight * (depth[v] + 1) - self.epistemic_weight * h
                if best is None or (efe, depth[v]) < best[:2]:
                    best = (efe, depth[v], v, d)
        if best is None:
            self._touch_plan = None
            return None
        _, _, v, d = best
        self._touch_plan = (v, d)
        return self._walk_to_touch(p, parent, v, d)

    def _walk_to_touch(self, p: Cell, parent: dict[Cell, Cell], v: Cell, d: Cell) -> Cell:
        self.target = (v[0] + d[0], v[1] + d[1])
        if v == p:
            return d
        step = v
        while parent[step] != p:
            step = parent[step]
        return step[0] - p[0], step[1] - p[1]

    def act(self, obs: Observation) -> Cell | None:
        self.map.update(obs)
        p = obs.position

        goal_cell = self._locate_goal_candidate()
        if goal_cell is not None:
            route = self.map.route(p, goal_cell)
            if route:
                self.target = goal_cell
                return route[0][0] - p[0], route[0][1] - p[1]

        parent, order = self._tree(p)
        beliefs = self.map.beliefs
        views: dict[Cell, frozenset[Cell]] = {}
        cell_gain: dict[Cell, float] = {}

        def view(v: Cell) -> frozenset[Cell]:
            if v not in views:
                views[v] = frozenset(self.map.visible_cells_from(v))
            return views[v]

        def gain_of(c: Cell) -> float:
            if c not in cell_gain:
                cell_gain[c] = self.map.expected_cell_information_gain(c)
            return cell_gain[c]

        gain, goal_mass, risk, depth = {p: 0.0}, {p: 0.0}, {p: 0.0}, {p: 0}
        for v in order:
            u = parent[v]
            # a cell already seen from v gives only repeat_weight of a fresh look's information
            w = self.map.repeat_weight
            gain[v] = gain[u] + sum(gain_of(c) * (w if (c, v) in self.map.looked else 1.0)
                                    for c in view(v) - view(u))
            goal_mass[v] = goal_mass[u] + float(beliefs[v[0], v[1], 2])
            risk[v] = risk[u] + float(beliefs[v[0], v[1], 0])
            depth[v] = depth[u] + 1

        # A visited cell is a target only if it still has something to show: with persistent
        # misreads a cell already seen from every reachable spot has nothing left, and going there
        # and back is a dead loop. An unvisited cell is always worth a visit (it may be the goal).
        def worth(q: Cell) -> bool:
            if q in self.map.visits:
                return gain[q] > 0.01
            if self.min_goal_belief is None:
                return True
            return gain[q] > 0.01 or float(beliefs[q[0], q[1], 2]) > self.min_goal_belief

        candidates = [q for q in self.map.frontiers(p) if q in parent and worth(q)]
        if not candidates:
            candidates = [q for q in order if (q not in self.map.visits or self.map.cell_entropy(q) > 0.05)
                          and worth(q)]
        if not candidates and self.map.repeat_weight < 1.0:
            touch = self._touch_move(p, parent, order, depth)
            if touch is not None:
                return touch
        if not candidates:
            return relook_move(self.map, p)

        def score(q: Cell) -> tuple[float, float, float]:
            pragmatic = -self.goal_weight * min(1.0, goal_mass[q]) + self.risk_weight * risk[q]
            efe = float(expected_free_energy(
                pragmatic=pragmatic, epistemic=gain[q], effort=float(depth[q]), smoothness=0.0,
                epistemic_weight=self.epistemic_weight, effort_weight=self.effort_weight,
            ))
            return efe, gain[q], pragmatic

        ranked = sorted(candidates, key=lambda q: (score(q)[0], depth[q], q))
        # the top alternatives with their EFE terms, for explanations and videos
        self.last_candidates = [dict(cell=q, efe=score(q)[0], gain=score(q)[1], pragmatic=score(q)[2],
                                     effort=depth[q]) for q in ranked[:3]]
        best = ranked[0]
        if self.target in candidates and self.target != best:
            if score(self.target)[0] <= score(best)[0] + self.switch_margin:
                best = self.target

        self.target = best
        self.last_efe_score, self.last_epistemic, self.last_pragmatic = score(best)
        self.last_effort = float(depth[best])
        step = best
        while parent[step] != p:
            step = parent[step]
        return step[0] - p[0], step[1] - p[1]


class NearestFrontierExplorer:
    """Classical Yamauchi frontier exploration baseline.

    Selects the reachable frontier that minimizes topological route distance.
    No epistemic gain calculation, no active inference.
    """

    def __init__(self, sensor_range: int = 3, size: int = 21, seed: int = 0,
                 obs_model: ObservationModel | None = None, prior: SpatialPrior | None = None):
        self.sensor_range = sensor_range
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.map = ProbabilisticMap(size=size, sensor_range=sensor_range, obs_model=obs_model, prior=prior)
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
            return relook_move(self.map, p)

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
            return relook_move(self.map, p)

        self.target = best[1]
        next_c = best[2][0]
        return next_c[0] - p[0], next_c[1] - p[1]


class InfoGainFrontierExplorer:
    """Information-gain frontier baseline (greedy unseen area / entropy ratio).

    Ranks frontiers by information gain / route length without pragmatic value.
    """

    def __init__(self, sensor_range: int = 3, size: int = 21, seed: int = 0,
                 gain_weight: float = 0.03, obs_model: ObservationModel | None = None,
                 prior: SpatialPrior | None = None):
        self.sensor_range = sensor_range
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.map = ProbabilisticMap(size=size, sensor_range=sensor_range, obs_model=obs_model, prior=prior)
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
            return relook_move(self.map, p)

        best: tuple[float, int, Cell, list[Cell]] | None = None
        for q in candidates:
            route = self.map.route(p, q)
            if not route:
                continue
            # Information rate = (1.0 + weight * expected_info_gain) / route_len
            gain = self.map.expected_information_gain(q)
            rate = (1.0 + self.gain_weight * gain) / len(route)
            # Maximize rate (store the negated rate so that smaller is better throughout; the first
            # version stored +rate but compared -rate against it, which picked frontiers almost
            # arbitrarily and made this baseline weaker than nearest frontier)
            choice = (-rate, len(route), q, route)
            if best is None or choice[:2] < best[:2]:
                best = choice

        if best is None:
            return relook_move(self.map, p)

        self.target = best[2]
        next_c = best[3][0]
        return next_c[0] - p[0], next_c[1] - p[1]
