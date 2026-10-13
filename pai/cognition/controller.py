"""Shared goal/act/observe/remember loop with domain-specific skill policies."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque

from pai.cognition.contracts import (CognitiveEpisode, DomainAdapter, GoalRelation,
                                     SkillCommand, StateBelief, SurpriseEvidence)


class CognitiveController:
    def __init__(self, memory=None):
        self.memory = memory

    def run(self, adapter: DomainAdapter, goal: GoalRelation, policy,
            budget: int = 100, episode: int = 0) -> CognitiveEpisode:
        """Policy selects a portable SkillCommand; adapter alone touches body state."""
        record = CognitiveEpisode(adapter.domain, episode, goal, False, 0)
        state = adapter.observe()
        record.beliefs.append(state)
        recent = deque(maxlen=12)
        for _ in range(budget):
            if state.satisfies(goal):
                break
            command: SkillCommand | None = policy(state, adapter)
            if command is None:
                break
            stage = (policy.goal.current.name if hasattr(policy, "goal") and policy.goal.current
                     else command.name)
            outcome = adapter.execute(command)
            state = adapter.observe()
            record.commands.append(command)
            record.outcomes.append(outcome)
            record.beliefs.append(state)
            record.evidence.append(SurpriseEvidence(state.step, ("skill_stall",),
                                                    (float(outcome.stalled),), (0.1,)))
            diagnostic = getattr(adapter, "last_diagnostic", None)
            if diagnostic is not None:
                entry = {"stage": stage, **diagnostic}
                recent.append(entry)
                if (entry.get("contact_events") or entry.get("relations_gained")
                        or entry.get("relations_lost")):
                    record.events.append({"type": "action_transition", **entry})
        if hasattr(adapter, "finalize"):
            adapter.finalize()
            state = adapter.observe()
            record.beliefs.append(state)
        record.success = state.satisfies(goal)
        record.steps = len(record.commands)
        if not record.success and recent:
            from pai.cognition.contact_repair import classify_failure
            lost = [relation for entry in recent for relation in entry.get("relations_lost", ())]
            stage = recent[-1]["stage"]
            record.events.append({"type": "failure_trace", "cause": classify_failure(stage, lost_relations=lost),
                                  "goal": [goal.predicate, goal.subject, goal.object],
                                  "last_actions": list(recent)})
        if self.memory is not None:
            self.memory.add(record)
        return record

    def run_sequence(self, adapter: DomainAdapter, goals: list[GoalRelation], policy_factory,
                     budget_per_goal: int = 150, episode: int = 0,
                     max_passes: int = 2, repair_factory=None):
        """Pursue ordered relations and verify all of them after the last action."""
        records = []
        for _ in range(max_passes):
            for goal in goals:
                state = adapter.observe()
                if state.satisfies(goal):
                    continue
                if repair_factory is not None:
                    for repair in repair_factory(goal, state):
                        if not adapter.observe().satisfies(repair):
                            records.append(self.run(adapter, repair, policy_factory(repair),
                                                    budget_per_goal, episode))
                record = self.run(adapter, goal, policy_factory(goal), budget_per_goal, episode)
                records.append(record)
            final = adapter.observe()
            if all(final.satisfies(goal) for goal in goals):
                break
        return SequenceResult(tuple(goals), records,
                              all(final.satisfies(goal) for goal in goals))


@dataclass
class SequenceResult:
    goals: tuple[GoalRelation, ...]
    episodes: list[CognitiveEpisode]
    success: bool


class MazeFrontierPolicy:
    """Existing maze L1 planner exposed through the shared SkillCommand contract."""

    def __init__(self, size: int, sensor_range: int, seed: int = 0):
        from pai.maze.agent import Explorer
        self.explorer = Explorer(sensor_range, size, seed)

    def __call__(self, state: StateBelief, adapter: DomainAdapter) -> SkillCommand | None:
        direction = self.explorer.act(adapter.last_observation)
        return None if direction is None else SkillCommand("move", parameters=tuple(map(float, direction)))


class ArmRelationalPolicy:
    """Existing learned world model, MPPI and subgoals behind the common loop."""

    def __init__(self, env, world_model, slice_config, relation="on", obj="red",
                 target="plate", device="cpu", block_clearance=0.008,
                 contact_repair=True):
        from pai.goals.relations import RelationalGoal
        from pai.planning.mppi import MPPIPlanner
        from pai.world.entities import entity_names
        self.env = env
        self.device = device
        self.goal = RelationalGoal(relation, obj, target, entity_names(env), env.objects,
                                   block_clearance=block_clearance,
                                   contact_repair=contact_repair)
        self._transit_complete = False
        self.planner = MPPIPlanner(world_model, horizon=slice_config.horizon,
                                   samples=slice_config.samples, iterations=slice_config.iterations,
                                   temperature=slice_config.temperature,
                                   epistemic_weight=slice_config.epistemic_weight, device=device)

    def __call__(self, state: StateBelief, adapter: DomainAdapter) -> SkillCommand | None:
        import torch
        tokens = torch.as_tensor(adapter.last_tokens, device=self.device)
        if not self._transit_complete:
            gripper = tokens[self.goal.i_g, :3]
            obj = tokens[self.goal.i_o, :3]
            if float(gripper[2]) < 0.13 and float(torch.linalg.vector_norm(gripper[:2] - obj[:2])) > 0.08:
                origin_xy = gripper[:2].detach().clone()
                def clear_table(x):
                    ee = x[..., self.goal.i_g, :3]
                    xy = ((ee[..., :2] - origin_xy) ** 2).sum(-1)
                    height = torch.relu(0.15 - ee[..., 2]) ** 2
                    return 100 * xy + 150 * height
                action, _ = self.planner.act(tokens, clear_table, grip=1.0)
                a = action.cpu().numpy()
                return SkillCommand("move_end_effector", parameters=(float(a[0]), float(a[1]),
                                                                  float(a[2]), 0.0, float(a[3])))
            self._transit_complete = True
        if self.goal.update(tokens):
            return None
        action, _ = self.planner.act(tokens, self.goal.current.cost, grip=self.goal.current.grip)
        a = action.cpu().numpy()
        return SkillCommand("move_end_effector", parameters=(float(a[0]), float(a[1]),
                                                              float(a[2]), 0.0, float(a[3])))
