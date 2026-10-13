"""Translate arm and maze observations/episodes into the shared cognitive contract."""

from __future__ import annotations

from pai.cognition.contracts import (CognitiveEpisode, EntityState, GoalRelation, SkillCommand,
                                     SkillOutcome, StateBelief, SurpriseEvidence)

import numpy as np
from math import sqrt


def maze_state(position, visible: dict, step: int, observed_count: int = 0,
               reached_goal: bool = False) -> StateBelief:
    """Only the agent's observed cells are encoded; no full maze truth is read."""
    entities = [EntityState("body", "mobile_body", {"row": position[0], "col": position[1]})]
    relations = {("at", "body", f"place:{position[0]},{position[1]}")}
    if reached_goal:
        relations.add(("goal_reached", "body", None))
    for (r, c), kind in visible.items():
        place = f"place:{r},{c}"
        entities.append(EntityState(place, "place", {"appearance": kind}))
        relations.add(("observed", "body", place))
        if kind == "goal":
            relations.add(("goal_visible", "body", place))
    return StateBelief("maze", step, tuple(entities), frozenset(relations),
                       uncertainty=1.0 / (1.0 + observed_count))


def arm_state(tokens, names: list[str], step: int, relations=()) -> StateBelief:
    """Privileged-state arm adapter used until Job A's pixel observations are ready."""
    entities = []
    for i, name in enumerate(names):
        pos = tokens[i, :3]
        entities.append(EntityState(name, "body" if i == 0 else "object",
                                    {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}))
    normalized = set()
    for relation in relations:
        if len(relation) == 2:
            predicate, subject = relation
            target = None
        elif len(relation) == 3:
            predicate, subject, target = relation
        else:
            continue
        normalized.add((str(predicate), str(subject), str(target) if target is not None else None))
    return StateBelief("arm", step, tuple(entities), frozenset(normalized))


def from_arm_record(record, names: list[str]) -> CognitiveEpisode:
    relation, args = record.goal.split("(", 1)
    subject, target = args.rstrip(")").split(", ")
    goal = GoalRelation(relation, subject, target)
    beliefs = []
    if record.tokens is not None:
        beliefs = [arm_state(tokens, names, i) for i, tokens in enumerate(record.tokens)]
        if record.success and beliefs:
            last = beliefs[-1]
            beliefs[-1] = StateBelief(last.domain, last.step, last.entities,
                                      last.relations | frozenset({(relation, subject, target)}), last.uncertainty)
    actions = [SkillCommand("move_end_effector", parameters=(float(a[0]), float(a[1]),
                                                             float(a[2]), 0.0, float(a[3])))
               for a in record.actions] if record.actions is not None else []
    evidence = [SurpriseEvidence(i, ("arm_surprise",), (sqrt(2 * max(float(s), 0.0)),), (1.0,))
                for i, s in enumerate(record.surprise)]
    return CognitiveEpisode("arm", record.episode, goal, record.success, record.steps,
                            beliefs=beliefs, commands=actions, evidence=evidence,
                            cause_posterior=record.cause_posterior or {},
                            adaptations=record.adaptations, events=record.events)


class MazeDomainAdapter:
    """L0 skill and observation adapter; planning never sees MazeWorld.grid."""

    domain = "maze"

    def __init__(self, world, mobile: bool = True):
        from pai.maze.mobility import ContinuousMobile
        self.world = world
        self.body = ContinuousMobile(world) if mobile else None
        self.last_observation = world.observe()

    def observe(self) -> StateBelief:
        return maze_state(self.last_observation.position, self.last_observation.visible,
                          self.world.steps, len(self.last_observation.visible),
                          self.last_observation.reached_goal)

    def available_skills(self) -> tuple[SkillCommand, ...]:
        from pai.maze.env import DIRECTIONS
        return tuple(SkillCommand("move", parameters=tuple(map(float, d))) for d in DIRECTIONS)

    def execute(self, command: SkillCommand) -> SkillOutcome:
        if command not in self.available_skills():
            raise ValueError("unsupported maze skill")
        direction = tuple(map(int, command.parameters))
        if self.body:
            result = self.body.move(direction)
            self.last_observation = result.observation
            return SkillOutcome(command, result.executed, result.stalled, result.duration)
        self.last_observation = self.world.step(direction)
        return SkillOutcome(command, not self.last_observation.collided,
                            self.last_observation.collided, 1.0)


class ArmDomainAdapter:
    """Arm adapter for the common loop: perceived entity tokens and relations derived from them
    (objects seen through the calibrated camera pose), end-effector skill. Simulator predicates are
    used only in `last_diagnostic`, as ground truth for scoring."""

    domain = "arm"

    def __init__(self, env, seed: int = 0, action_repeat: int = 1,
                 settle_steps: int = 0):
        from pai.world.entities import entity_names
        from pai.perception.camera_frame import CameraFramePerception
        self.env = env
        self.names = entity_names(env)
        self.last_observation = env.reset(seed=seed)
        self.action_repeat = action_repeat
        self.settle_steps = settle_steps
        self.prev_ee = self.last_observation["ee_pos"].copy()
        self.last_tokens = None
        self.sense = CameraFramePerception(env)
        self.sense.reset()
        self.last_diagnostic = None

    def observe(self) -> StateBelief:
        self.last_tokens = self.sense(self.last_observation, self.prev_ee)
        # relations from the agent's own perception, never the simulator's predicates (those stay
        # in the diagnostics below, as ground truth for scoring)
        from pai.envs.predicates import perceived_predicates
        rel = perceived_predicates(self.env.objects, self.last_tokens, self.names)
        return arm_state(self.last_tokens, self.names, self.env.t, rel)

    def available_skills(self) -> tuple[SkillCommand, ...]:
        return (SkillCommand("move_end_effector"),)

    def execute(self, command: SkillCommand) -> SkillOutcome:
        if command.name != "move_end_effector" or len(command.parameters) != 5:
            raise ValueError("unsupported arm skill")
        before = self.last_observation
        event_start = len(self.env.events.events)
        for _ in range(self.action_repeat):
            self.prev_ee = self.last_observation["ee_pos"].copy()
            self.last_observation = self.env.step(np.asarray(command.parameters, dtype=float))
        after = self.last_observation
        before_rel = set(before["predicates"])
        after_rel = set(after["predicates"])
        def snapshot(obs):
            return {name: {"position": np.asarray(st["pos"]).round(5).tolist(),
                           "speed": round(float(np.linalg.norm(st["vel"])), 5)}
                    for name, st in obs["state"].items()}
        self.last_diagnostic = {
            "step": self.env.t,
            "before": snapshot(before), "after": snapshot(after),
            "relations_gained": sorted(map(list, after_rel - before_rel)),
            "relations_lost": sorted(map(list, before_rel - after_rel)),
            "contact_events": self.env.events.events[event_start:],
            "wrist_force": np.asarray(after["wrist_force"]).round(4).tolist(),
        }
        return SkillOutcome(command, True, False, self.env.control_dt * self.action_repeat)

    def finalize(self) -> None:
        for _ in range(self.settle_steps):
            self.last_observation = self.env.step(np.array([0, 0, 0, 0, 1.0]))
