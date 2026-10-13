"""Expose door and drawer discovery actions through the common cognitive record."""

from __future__ import annotations

from pai.cognition.contracts import (CognitiveEpisode, EntityState, GoalRelation,
                                     SkillCommand, SkillOutcome, StateBelief, SurpriseEvidence)


class FeatureGoalWorld:
    """Use an observed door or drawer feature as the discovery goal."""

    def __init__(self, world, feature: str):
        self.world = world
        self.feature = feature

    def entities(self):
        return self.world.entities()

    def features(self):
        return self.world.features()

    def actions(self):
        return self.world.actions()

    def execute(self, action):
        return self.world.execute(action)

    def goal(self):
        return {self.feature: "open"}

    def goal_reached(self):
        return self.features().get(self.feature) == "open"


def discovery_goal(world) -> GoalRelation:
    (feature, value), = world.goal().items()
    return GoalRelation("feature_equals", feature, value)


def command_for(action: tuple) -> SkillCommand:
    return SkillCommand(action[0], target=action[1],
                        arguments=tuple(str(x) for x in action[2:]))


def action_for(command: SkillCommand) -> tuple:
    return (command.name, command.target, *(
        int(x) if x in ("-1", "1") else x for x in command.arguments))


class DiscoveryDomainAdapter:
    """The discovery agent sees only World.entities/features/actions, never truth()."""

    domain = "discovery"

    def __init__(self, world):
        self.world = world
        self.step = 0
        self.commands: list[SkillCommand] = []
        self.outcomes: list[SkillOutcome] = []
        self.beliefs: list[StateBelief] = []
        self.evidence: list[SurpriseEvidence] = []
        self.action_traces: list[dict] = []
        self.beliefs.append(self.observe())

    def observe(self) -> StateBelief:
        entities = tuple(EntityState(e.id, e.kind,
                                     {"x": float(e.pos[0]), "y": float(e.pos[1]),
                                      "z": float(e.pos[2]), "joint": e.joint})
                         for e in self.world.entities())
        relations = frozenset(("feature_equals", str(k), str(v))
                              for k, v in self.world.features().items())
        return StateBelief(self.domain, self.step, entities, relations)

    def available_skills(self) -> tuple[SkillCommand, ...]:
        return tuple(command_for(a) for a in self.world.actions())

    def execute(self, command: SkillCommand) -> SkillOutcome:
        action = action_for(command)
        if action not in self.world.actions():
            raise ValueError(f"unsupported discovery action {action}")
        result = self.world.execute(action)
        self.world_result = result
        self.step += 1
        from pai.cognition.contact_repair import classify_failure
        trace = {"type": "action_trace", "step": self.step, "action": list(action),
                 "executed": bool(result.executed), "stalled": bool(result.stalled),
                 "notes": result.notes,
                 "changed": {key: list(pair) for key, pair in result.changed.items()},
                 "cause": classify_failure(action[0], stalled=result.stalled,
                                           notes=result.notes, changed=result.changed)}
        diagnostic = getattr(getattr(self.world, "world", self.world), "last_diagnostic", None)
        if diagnostic is not None:
            trace["contact"] = diagnostic
        self.action_traces.append(trace)
        outcome = SkillOutcome(command, result.executed, result.stalled, result.cost,
                               tuple((feature, str(after))
                                     for feature, (_, after) in result.changed.items()))
        self.commands.append(command)
        self.outcomes.append(outcome)
        self.evidence.append(SurpriseEvidence(self.step, ("skill_stall", "state_change"),
                                               (float(result.stalled), float(bool(result.changed))),
                                               (0.1, 1.0)))
        self.beliefs.append(self.observe())
        return outcome


class _DiscoveryWorldProxy:
    """Keep the existing discovery policy intact while recording each skill via the adapter."""

    def __init__(self, adapter: DiscoveryDomainAdapter):
        self.adapter = adapter

    def entities(self):
        return self.adapter.world.entities()

    def features(self):
        return self.adapter.world.features()

    def actions(self):
        return self.adapter.world.actions()

    def goal(self):
        return self.adapter.world.goal()

    def goal_reached(self):
        return self.adapter.world.goal_reached()

    def execute(self, action):
        self.adapter.execute(command_for(action))
        return self.adapter.world_result


def run_discovery_cognitive(world, *, seed: int = 0, budget_actions: int = 300,
                            memory=None, recipes=None, object_trials: bool = False):
    """Run the proven discovery policy and store its individual actions in shared memory."""
    from pai.discovery.agent import DiscoveryAgent
    from pai.discovery.explain import explain

    adapter = DiscoveryDomainAdapter(world)
    proxy = _DiscoveryWorldProxy(adapter)
    log = DiscoveryAgent(seed=seed, object_trials=object_trials).run(
        proxy, budget_actions=budget_actions, recipes=recipes)
    explanation = explain(log)
    record = CognitiveEpisode("discovery", seed, discovery_goal(world), log.reached_goal,
                              len(adapter.commands), beliefs=adapter.beliefs,
                              commands=adapter.commands, outcomes=adapter.outcomes,
                              evidence=adapter.evidence,
                              events=adapter.action_traces + [{"type": "discovery_explanation",
                                       "text": explanation.text}])
    if memory is not None:
        memory.add(record)
    return record, log
