"""End-to-end checks for the next independent body and cognition adapters."""

import numpy as np

from pai.cognition.contracts import GoalRelation, SkillCommand, SkillOutcome, StateBelief
from pai.cognition.controller import CognitiveController
from pai.cognition.experiments import BinaryExperiment, CausalExperimentPlanner
from pai.maze.advanced import run_advanced_episode
from pai.maze.env import MazeWorld
from pai.maze.learned_vision import TileColorModel, simulator_samples


def test_causal_probe_changes_belief_after_its_outcome():
    planner = CausalExperimentPlanner({"passage_blocked": 0.5, "camera_shift": 0.5})
    diagnostic = BinaryExperiment("try passage", {"passage_blocked": 0.05, "camera_shift": 0.8})
    uninformative = BinaryExperiment("wait", {"passage_blocked": 0.5, "camera_shift": 0.5})
    assert planner.choose([uninformative, diagnostic]) == diagnostic
    assert planner.posterior(diagnostic, True)["camera_shift"] > 0.9
    assert planner.posterior(diagnostic, False)["passage_blocked"] > 0.8


def test_learned_vision_uses_unseen_rendered_palettes(tmp_path):
    train_x, train_y = simulator_samples(range(100, 115), appearance_jitter=20,
                                          appearance_bias=(30, -20, 15))
    model = TileColorModel.fit(train_x, train_y)
    path = tmp_path / "tile_model.npz"
    model.save(path)
    test_x, test_y = simulator_samples(range(40000, 40005), appearance_jitter=20,
                                        appearance_bias=(30, -20, 15))
    prediction = TileColorModel.load(path).predict_rgb(test_x)
    assert np.mean(prediction == test_y) > 0.98


def test_walking_body_recovers_without_marking_a_false_wall():
    world = MazeWorld(0, size=11)
    result = run_advanced_episode(world, visual=True, mobile=True, walk=True,
                                  fall_at=20, budget=200, odom_noise=0.35)
    assert result["falls"] == result["recoveries"] == 1
    assert result["visited"] == result["reachable"]
    assert not any(report["cause"] == "passage_blocked" and report["step"] == 21
                   for report in result["cause_reports"])


def test_sequence_can_intervene_to_clear_a_blocked_relation():
    class Adapter:
        domain = "toy"
        def __init__(self):
            self.step = 0
            self.relations = {("on", "top", "base")}
        def observe(self):
            return StateBelief("toy", self.step, (), frozenset(self.relations))
        def execute(self, command):
            self.step += 1
            if command.target == "top":
                self.relations.discard(("on", "top", "base"))
                self.relations.add(("in", "top", "bowl"))
            elif ("on", "top", "base") not in self.relations:
                self.relations.add(("on", "base", "plate"))
            return SkillOutcome(command, True, False, 1.0)
    goals = [GoalRelation("on", "base", "plate")]
    without_repair = CognitiveController().run_sequence(
        Adapter(), goals, lambda goal: lambda state, adapter:
        SkillCommand("move", target=goal.subject), budget_per_goal=2)
    result = CognitiveController().run_sequence(
        Adapter(), goals, lambda goal: lambda state, adapter:
        SkillCommand("move", target=goal.subject), budget_per_goal=2,
        repair_factory=lambda goal, state: [GoalRelation("in", "top", "bowl")]
        if ("on", "top", "base") in state.relations else [])
    assert not without_repair.success
    assert result.success
    assert [r.goal.predicate for r in result.episodes] == ["in", "on"]
