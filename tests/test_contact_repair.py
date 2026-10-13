"""Regressions for elevated support placement and dependency clearing."""

import torch

from pai.cognition.contact_repair import classify_failure, stack_clearance_repairs
from pai.cognition.contracts import GoalRelation, StateBelief
from pai.envs.predicates import ObjectSpec
from pai.goals.relations import RelationalGoal


def test_tower_release_height_follows_support_observation():
    objects = {name: ObjectSpec(name, "block", 0.02, 0.04)
               for name in ("green", "red")}
    goal = RelationalGoal("on", "green", "red", ["gripper", "green", "red"], objects)
    x = torch.zeros(3, 19)
    x[0, 8] = 0.04
    x[2, :3] = torch.tensor([0.5, 0.0, 0.078])
    x[1, :3] = torch.tensor([0.5, 0.0, 0.126])
    assert goal.subgoals[5].done(x)
    x[2, 2] += 0.04
    assert not goal.subgoals[5].done(x)
    x[1, 2] += 0.04
    x[2, 3] = 0.08
    assert not goal.subgoals[5].done(x)  # moving support is unsafe to release onto
    x[2, 3] = 0.0
    assert goal.subgoals[5].done(x)


def test_rebuild_clears_descendants_top_first():
    state = StateBelief("arm", 0, (), frozenset({("on", "red", "blue"),
                                                 ("on", "green", "red")}))
    repairs = stack_clearance_repairs(GoalRelation("on", "blue", "plate"), state)
    assert repairs == [GoalRelation("in", "green", "bowl"),
                       GoalRelation("in", "red", "bowl")]
    assert classify_failure("lowered", lost_relations=[("on", "red", "blue")]) == "support_relation_lost"
