"""Skill-level maze body for the learned chain, with sensor-coupled passage.

Navigation to a selected entity is a GIVEN L0 reach skill. The policy decides
which entity/action comes next; this body never chooses a drawer, key or fixture
role. It uses one existing Explorer map across every reach. This is a bridge of
two mocks, not a trained mobile manipulator or a physical carrying simulation.
"""
from __future__ import annotations

import numpy as np

from pai.discovery.belief import entity_of
from pai.discovery.interface import Outcome
from pai.discovery.key_chain_recipe import EvidenceWorld
from pai.maze.agent import Explorer
from pai.maze.env import Observation, neighbors


class SituatedChainBody:
    """Route generic reaches by observed position; the fixture owns the door.

    Scene construction supplies two fixture-workspace anchors. A metric radius
    .8m maps any observed part/object to its nearest workspace; this is a given
    spatial discretisation, not a membership label or a policy prior.
    """
    def __init__(self, maze, fixture, seed=0, move_budget=10000):
        self.maze = maze
        self.body = EvidenceWorld(fixture)
        self.move_budget = move_budget
        self.navigator = Explorer(maze.sensor_range, maze.size, seed)
        self.navigation = []
        # Ground-truth geometry is used ONLY here for scene construction.
        distances = maze.distances_from(maze.start)
        near = [p for p in distances if sum(abs(a-b) for a, b in zip(p, maze.goal)) == 1]
        self.approach = min(near, key=lambda p: distances[p])
        # A locked exit must not be used as a path to place reachable furniture.
        accessible = {maze.start: 0}
        queue = [maze.start]
        for p in queue:
            for q in neighbors(p):
                if q != maze.goal and q in distances and q not in accessible:
                    accessible[q] = accessible[p] + 1
                    queue.append(q)
        far = [p for p in accessible if p not in (maze.goal, maze.start, self.approach)
               and sum(abs(a-b) for a, b in zip(p, maze.goal)) >= 4]
        if not far:
            raise ValueError('layout has no reachable remote workspace before the locked exit')
        self.remote = max(far, key=lambda p: (accessible[p], p))
        self.entered_exit = False

    def entities(self):
        return self.body.entities()

    def features(self):
        return self.body.features()

    def actions(self):
        return self.body.actions()

    def goal(self):
        return self.body.goal()

    def goal_reached(self):
        return self.body.goal_reached()

    def _step(self, delta):
        target = tuple(a+b for a, b in zip(self.maze.position, delta))
        if target == self.maze.goal and not self.body.goal_reached():
            self.maze.steps += 1
            self.navigation.append({'type': 'closed_passage_contact', 'position': list(target)})
            return False
        self.maze.step(delta)
        return True

    def walk_to(self, target):
        # Remove stale goals when the same map is reused for a new reach.
        for cell, kind in list(self.navigator.memory.cells.items()):
            if kind == 'goal':
                self.navigator.memory.cells[cell] = 'floor'
        while self.maze.position != target and self.maze.steps < self.move_budget:
            base = self.maze.observe()
            visible = {p: ('goal' if p == target and kind != 'wall' else
                           'floor' if kind == 'goal' else kind) for p, kind in base.visible.items()}
            if not self.body.goal_reached() and self.maze.goal in visible:
                visible[self.maze.goal] = 'wall'
            obs = Observation(base.position, visible, False, False)
            move = self.navigator.act(obs)
            route = self.navigator.memory.route(base.position, target)
            if route:
                move = tuple(a-b for a, b in zip(route[0], base.position))
            if move is None or not self._step(move):
                return False
        return self.maze.position == target

    def execute(self, action):
        ents = {e.id: e for e in self.body.entities()}
        eid = action[2] if action[0] == 'insert' else action[1]
        goal_id = entity_of(next(iter(self.body.goal())))
        distance = np.linalg.norm(np.subtract(ents[eid].pos, ents[goal_id].pos))
        destination = self.approach if distance <= .8 else self.remote
        before = self.maze.steps
        reached = self.walk_to(destination)
        self.navigation.append({'type': 'reach', 'entity': eid, 'destination': list(destination),
                                'moves': self.maze.steps - before, 'reached': reached})
        if not reached:
            return Outcome(tuple(action), False, False, {}, [], float(self.maze.steps-before), '')
        out = self.body.execute(action)
        out.cost += self.maze.steps - before
        return out

    def pass_goal(self):
        """Execute the requested exit navigation after its observed guard holds.

        Traversal/navigation primitives are given L0; the manipulated dependency
        chain is learned. No writable door-open flag exists on this adapter.
        """
        if self.body.goal_reached():
            self.entered_exit = self.walk_to(self.maze.goal)
        return self.entered_exit
