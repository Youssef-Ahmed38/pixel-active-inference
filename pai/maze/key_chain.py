"""One observation-driven route through a maze, drawer, key, and locked door.

The maze and fixture simulators retain their own body dynamics. This module
composes their public skills and keeps fixture actions local to the visited site.
Simulator positions and fixture names are used only to build and score a scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pai.cognition.discovery import run_discovery_cognitive
from pai.discovery.interface import Outcome
from pai.maze.agent import Explorer
from pai.maze.env import Cell, MazeWorld, Observation, neighbors


def _distance(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class KeyChainMaze:
    """Expose a drawer and a locked passage as observed maze landmarks."""

    def __init__(self, maze: MazeWorld, drawer: Cell):
        if drawer == maze.goal or drawer == maze.start or maze.grid[drawer]:
            raise ValueError("drawer must occupy a separate traversable cell")
        self.maze = maze
        self.drawer = drawer
        self.door = maze.goal
        self.door_open = False

    def observe(self, collided: bool = False) -> Observation:
        base = self.maze.observe(collided)
        visible = dict(base.visible)
        if self.drawer in visible:
            visible[self.drawer] = "drawer"
        if self.door in visible:
            visible[self.door] = "exit" if self.door_open else "door"
        return Observation(base.position, visible, collided,
                           self.door_open and base.position == self.door)

    def step(self, move: Cell) -> Observation:
        target = self.maze.position[0] + move[0], self.maze.position[1] + move[1]
        if target == self.door and not self.door_open:
            self.maze.steps += 1
            self.maze.events.append({"step": self.maze.steps, "type": "locked_passage_contact",
                                     "position": target})
            return self.observe(True)
        return self._after(self.maze.step(move))

    def _after(self, base: Observation) -> Observation:
        return self.observe(base.collided)


class LocalFixture:
    """A fixture's public entities and actions, restricted by the body's location."""

    def __init__(self, world, part_ids: set[str], goal_feature: str,
                 object_ids: set[str] | None = None):
        self.world = world
        self.part_ids = set(part_ids)
        self.object_ids = object_ids
        self.goal_feature = goal_feature

    def _ids(self) -> set[str]:
        if self.object_ids is not None:
            return self.part_ids | self.object_ids
        return self.part_ids | {e.id for e in self.world.entities() if e.kind == "object"}

    def entities(self):
        ids = self._ids()
        return [e for e in self.world.entities() if e.id in ids]

    def features(self):
        ids = self._ids()
        return {key: value for key, value in self.world.features().items()
                if key == "body:hand" or (key.startswith(("part:", "obj:"))
                                          and key.split(":", 1)[1].split(".", 1)[0] in ids)}

    def actions(self):
        ids = self._ids()
        return [action for action in self.world.actions()
                if action[1] in ids and (action[0] != "insert" or action[2] in self.part_ids)]

    def execute(self, action):
        if action not in self.actions():
            raise ValueError(f"fixture action out of reach: {action}")
        result = self.world.execute(action)
        ids = self._ids()
        changed = {key: value for key, value in result.changed.items()
                   if key == "body:hand" or (key.startswith(("part:", "obj:"))
                                             and key.split(":", 1)[1].split(".", 1)[0] in ids)}
        return Outcome(action, result.executed, result.stalled, changed,
                       [e for e in result.revealed if e.id in ids], result.cost, result.notes)

    def goal(self):
        return {self.goal_feature: "open"}

    def goal_reached(self):
        return self.features().get(self.goal_feature) == "open"


@dataclass
class KeyChainResult:
    success: bool
    stage: str
    moves: int
    fixture_actions: int
    drawer_open: bool
    door_open: bool
    entered_exit: bool
    events: list[dict] = field(default_factory=list)


class KeyChainAgent:
    """Maintain one place map while switching between spatial and fixture goals."""

    def __init__(self, size: int, sensor_range: int, seed: int = 0):
        self.explorer = Explorer(sensor_range, size, seed)
        self.sightings: dict[str, Cell] = {}

    def move(self, obs: Observation, target: str) -> Cell | None:
        for cell, kind in obs.visible.items():
            if kind in ("drawer", "door", "exit"):
                self.sightings["door" if kind == "exit" else kind] = cell
        known = self.sightings.get(target)
        visible = {cell: ("goal" if kind == target else
                          "wall" if kind == "door" and target == "drawer" else
                          "floor" if kind in ("drawer", "door", "exit") else kind)
                   for cell, kind in obs.visible.items()}
        # The same map survives the switch from drawer to door. Re-label older
        # sightings so a previously visited destination remains routable.
        for name, cell in self.sightings.items():
            if cell in self.explorer.memory.cells:
                self.explorer.memory.cells[cell] = (
                    "goal" if name == target else
                    "wall" if name == "door" and target == "drawer" else "floor")
        if target == "door" and known is not None:
            self.explorer.memory.cells[known] = "goal"
            self.explorer.memory.edges.setdefault(known, set())
            for near in neighbors(known):
                if self.explorer.memory.cells.get(near) in ("floor", "goal"):
                    self.explorer.memory.edges[known].add(near)
                    self.explorer.memory.edges.setdefault(near, set()).add(known)
        planned = self.explorer.act(Observation(obs.position, visible, obs.collided, False))
        if known is not None and known != obs.position:
            route = self.explorer.memory.route(obs.position, known)
            if route:
                return route[0][0] - obs.position[0], route[0][1] - obs.position[1]
        return planned


def _drawer_cell(maze: MazeWorld) -> Cell:
    """Scene generation only: place the drawer away from the start and door."""
    distances = {maze.start: 0}
    queue = [maze.start]
    for cell in queue:
        for near in neighbors(cell):
            if (near != maze.goal and 0 <= near[0] < maze.size and
                    0 <= near[1] < maze.size and not maze.grid[near] and
                    near not in distances):
                distances[near] = distances[cell] + 1
                queue.append(near)
    candidates = [cell for cell in distances if cell != maze.goal and
                  _distance(cell, maze.goal) >= 4]
    return max(candidates, key=lambda cell: (distances[cell], cell))


def _repair_opening(door: LocalFixture, object_id: str, limit: int = 24) -> tuple[int, list[dict]]:
    """Retry a carried object's effect and nearby fixture probes from observations.

    This is a bounded fallback after the discovery policy has spent its budget.
    It uses appearance, angle and action outcomes, never fixture role names.
    """
    count, events = 0, []

    def act(action):
        nonlocal count
        if count >= limit or action not in door.actions():
            return None
        outcome = door.execute(action)
        count += 1
        events.append({"stage": "repair", "action": list(action),
                       "stalled": outcome.stalled, "changed": bool(outcome.changed),
                       "notes": outcome.notes})
        return outcome

    where = door.features().get(f"obj:{object_id}.where", "")
    if where == "free" and door.features().get("body:hand") == "empty":
        act(("pick", object_id))
    if door.features().get("body:hand") == object_id:
        where = door.features().get(f"obj:{object_id}.where", "")
        if where == "held":
            entities = {e.id: e for e in door.entities()}
            inserts = [a for a in door.actions() if a[0] == "insert" and a[1] == object_id]
            inserts.sort(key=lambda a: (float(entities[a[2]].attrs.get("size", 1.0)), a[2]))
            for action in inserts:
                outcome = act(action)
                if outcome is not None and not outcome.stalled and door.features().get(
                        f"obj:{object_id}.where", "").startswith("in:"):
                    break
    if door.features().get(f"obj:{object_id}.where", "").startswith("in:"):
        for sign in (1, -1):
            outcome = act(("turn_held", object_id, sign))
            if outcome is not None and not outcome.stalled:
                break
    if door.features().get("body:hand") == object_id:
        act(("release", object_id))

    entities = {e.id: e for e in door.entities()}
    for entity in entities.values():
        if entity.kind != "part" or entity.attrs.get("shape") != "wing":
            continue
        angle = door.features().get(f"part:{entity.id}.angle")
        if angle in ("pos", "neg"):
            act(("turn", entity.id, -1 if angle == "pos" else 1))
    goal_id = door.goal_feature.split(":", 1)[1].split(".", 1)[0]
    goal = entities.get(goal_id)
    probes = [a for a in door.actions() if a[0] in ("turn_pull", "turn_push", "press_push", "pull", "push")
              and a[1] in entities and entities[a[1]].graspable]
    probes.sort(key=lambda a: (
        0 if entities[a[1]].attrs.get("shape") in ("bar", "round", "ball") else 1,
        sum((x-y)**2 for x, y in zip(entities[a[1]].pos, goal.pos)) if goal else 0,
        {"turn_pull": 0, "turn_push": 1, "press_push": 2, "pull": 3, "push": 4}[a[0]], a))
    for action in probes:
        if door.goal_reached() or count >= limit:
            break
        act(action)
    return count, events


def run_key_chain(maze: MazeWorld, fixture, *, seed: int = 0,
                  move_budget: int = 700, fixture_budget: int = 120,
                  recipes=None, learn_recipe: bool = False) -> KeyChainResult:
    """Find the drawer, carry an observed object, unlock the door, enter the exit.

    Fixture identity is used by this environment adapter to scope reachability.
    The navigation and discovery policies see only observations/actions/outcomes.
    """
    scene = KeyChainMaze(maze, _drawer_cell(maze))
    agent = KeyChainAgent(maze.size, maze.sensor_range, seed)
    ids = fixture.pid
    drawer_id, door_id = ids["drawer"], ids["door"]
    drawer = LocalFixture(fixture, {drawer_id}, f"part:{drawer_id}.open", set())
    door_parts = {ids[name] for name in ("door", "handle", "thumb", "cylinder") if name in ids}
    door = LocalFixture(fixture, door_parts, f"part:{door_id}.open", set())
    events: list[dict] = []
    fixture_actions = 0
    pending_keys: list[str] = []
    stage = "find_drawer"
    obs = scene.observe()
    while maze.steps < move_budget:
        if stage in ("find_drawer", "return_drawer") and obs.visible.get(obs.position) == "drawer":
            if stage == "find_drawer":
                before_objects = {e.id for e in fixture.entities() if e.kind == "object"}
                record, _ = run_discovery_cognitive(drawer, seed=seed, budget_actions=fixture_budget)
                fixture_actions += record.steps
                events.append({"stage": "open_drawer", "success": record.success,
                               "actions": record.steps})
                if not record.success:
                    stage = "open_drawer_failed"
                    break
                # The pool contains only newly revealed elongated objects.
                # Their simulator match labels are never passed to this policy.
                pending_keys = sorted(e.id for e in fixture.entities() if e.kind == "object"
                                      and e.id not in before_objects and
                                      e.attrs.get("shape") == "elongated")
            if not pending_keys:
                stage = "keys_exhausted"
                break
            key_id = pending_keys.pop(0)
            drawer.object_ids = {key_id}
            outcome = drawer.execute(("pick", key_id))
            fixture_actions += 1
            events.append({"stage": "carry_object", "object": key_id,
                           "executed": outcome.executed, "stalled": outcome.stalled})
            if fixture.features().get("body:hand") != key_id:
                stage = "pick_failed"
                break
            stage = "find_door"
            door.object_ids = {key_id}
        if stage == "find_door" and "door" in agent.sightings:
            door_cell = agent.sightings["door"]
            if _distance(obs.position, door_cell) == 1:
                stage = "open_door"
                held = fixture.features().get("body:hand", "empty")
                if held != "empty":
                    inserts = [action for action in door.actions()
                               if action[0] == "insert" and action[1] == held]
                    appearances = {entity.id: entity for entity in door.entities()}
                    inserts.sort(key=lambda action: (
                        float(appearances[action[2]].attrs.get("size", 1.0)), action[2]))
                    inserted = False
                    for action in inserts:
                        outcome = door.execute(action)
                        fixture_actions += 1
                        events.append({"stage": "try_fixture", "action": list(action),
                                       "stalled": outcome.stalled})
                        if not outcome.stalled and fixture.features().get(
                                f"obj:{held}.where", "").startswith("in:"):
                            inserted = True
                            break
                    if not inserted:
                        events.append({"stage": "diagnose", "cause": "incompatible_object_or_alignment",
                                       "object": held, "evidence": "all local insertions stalled"})
                        door.execute(("release", held))
                        fixture_actions += 1
                        stage = "return_drawer"
                        continue
                    rotated = False
                    for sign in (1, -1):
                        action = ("turn_held", held, sign)
                        outcome = door.execute(action)
                        fixture_actions += 1
                        events.append({"stage": "try_turn", "action": list(action),
                                       "stalled": outcome.stalled})
                        if not outcome.stalled:
                            rotated = True
                            break
                    door.execute(("release", held))
                    fixture_actions += 1
                    if not rotated:
                        # A seated object that cannot rotate in either direction
                        # is disconfirmed by contact, independent of key identity.
                        events.append({"stage": "diagnose", "cause": "incompatible_key",
                                       "object": held, "evidence": "both inserted turns stalled"})
                        withdraw = door.execute(("pick", held))
                        fixture_actions += 1
                        if withdraw.stalled or fixture.features().get("body:hand") != held:
                            stage = "withdraw_failed"
                            break
                        door.execute(("release", held))
                        fixture_actions += 1
                        stage = "return_drawer"
                        continue
                record, log = run_discovery_cognitive(door, seed=seed,
                                                      budget_actions=fixture_budget,
                                                      object_trials=True, recipes=recipes)
                fixture_actions += record.steps
                events.append({"stage": stage, "success": record.success,
                               "actions": record.steps})
                if not record.success:
                    repair_count, repair_events = _repair_opening(door, held)
                    fixture_actions += repair_count
                    events.extend(repair_events)
                    if not door.goal_reached():
                        events.append({"stage": "diagnose", "cause": "opening_still_blocked",
                                       "evidence": "key and local release probes exhausted"})
                        break
                if learn_recipe and recipes is not None:
                    recipes.learn(log, source=f"key_chain_seed_{seed}")
                scene.door_open = True
                stage = "pass_door"
        if stage == "pass_door":
            door_cell = agent.sightings["door"]
            move = (door_cell[0] - obs.position[0], door_cell[1] - obs.position[1])
            obs = scene.step(move)
            stage = "complete" if obs.reached_goal else "pass_failed"
            break
        target = "drawer" if stage in ("find_drawer", "return_drawer") else "door"
        move = agent.move(obs, target)
        if move is None:
            stage = "route_exhausted"
            break
        obs = scene.step(move)
    if maze.steps >= move_budget and stage not in ("complete", "pass_failed"):
        stage = "move_budget"
    return KeyChainResult(stage == "complete", stage, maze.steps, fixture_actions,
                          drawer.goal_reached(), door.goal_reached(), obs.reached_goal, events)
