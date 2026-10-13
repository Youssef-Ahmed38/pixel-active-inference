"""New task adapters preserve goals, evidence and action provenance."""

import numpy as np

from pai.cognition.contracts import GoalRelation, SkillCommand, SkillOutcome, StateBelief
from pai.cognition.causes import select_arm_cause
from pai.cognition.controller import CognitiveController
from pai.cognition.discovery import FeatureGoalWorld, run_discovery_cognitive
from pai.cognition.memory import CognitiveMemory
from pai.cognition.thinker import OnlineArmCauseThinker, OnlineMazeCauseThinker
from pai.discovery.mock import MockWorld, sample_scenario
from pai.maze.env import MazeWorld


def test_discovery_adapter_records_individual_door_and_drawer_skills():
    scenario = sample_scenario(np.random.default_rng(0), split="test", lock_state="latch", p_fail=0)
    memory = CognitiveMemory()
    for task in ("door", "drawer"):
        base = MockWorld(scenario)
        world = FeatureGoalWorld(base, f"part:{base.pid[task]}.open")
        record, log = run_discovery_cognitive(world, seed=0, budget_actions=300, memory=memory)
        assert record.success and log.reached_goal
        assert len(record.commands) == len(record.outcomes) == record.steps
        assert all(command.name != "discover" for command in record.commands)
        assert record.beliefs[-1].satisfies(record.goal)
    assert len(memory.episodes) == 2


def test_sequence_checks_all_relations_after_final_skill():
    class Adapter:
        domain = "toy"
        def __init__(self):
            self.step = 0
            self.relations = set()
        def observe(self):
            return StateBelief("toy", self.step, (), frozenset(self.relations))
        def available_skills(self):
            return (SkillCommand("set"),)
        def execute(self, command):
            self.step += 1
            self.relations.add(("done", command.target, None))
            return SkillOutcome(command, True, False, 1.0)
    goals = [GoalRelation("done", "first"), GoalRelation("done", "second")]
    result = CognitiveController().run_sequence(Adapter(), goals,
                                                lambda goal: lambda state, adapter:
                                                SkillCommand("set", target=goal.subject), 2)
    assert result.success and len(result.episodes) == 2


def test_sequence_repairs_an_earlier_relation_disrupted_by_later_action():
    class Adapter:
        domain = "toy"
        def __init__(self):
            self.step = 0
            self.relations = set()
            self.disturbed = False
        def observe(self):
            return StateBelief("toy", self.step, (), frozenset(self.relations))
        def execute(self, command):
            self.step += 1
            self.relations.add(("done", command.target, None))
            if command.target == "second" and not self.disturbed:
                self.relations.discard(("done", "first", None))
                self.disturbed = True
            return SkillOutcome(command, True, False, 1.0)
    goals = [GoalRelation("done", "first"), GoalRelation("done", "second")]
    result = CognitiveController().run_sequence(Adapter(), goals,
                                                lambda goal: lambda state, adapter:
                                                SkillCommand("set", target=goal.subject), 2)
    assert result.success and len(result.episodes) == 3


def test_online_thinker_sees_evidence_during_episode():
    class FakeThinker:
        def infer(self, window):
            assert window.shape[0] == 20
            return {"passage_blocked": 0.95, "none": 0.05}
    online = OnlineMazeCauseThinker.__new__(OnlineMazeCauseThinker)
    online.thinker, online.threshold, online.history, online.reports = FakeThinker(), 0.9, [], []
    assert online.observe(0) is None
    report = online.observe(1, unexpected_collision=True)
    assert report["actionable"] and report["cause"] == "passage_blocked"


def test_online_arm_thinker_waits_for_a_surprise_trigger():
    class FakeThinker:
        def infer(self, window):
            assert window.shape == (20, 9)
            return {"camera_shift": 0.95, "none": 0.05}
    online = OnlineArmCauseThinker.__new__(OnlineArmCauseThinker)
    online.thinker, online.threshold, online.history, online.reports = FakeThinker(), 0.9, [], []
    assert online.observe(0, np.zeros(12), False) is None
    report = online.observe(1, np.r_[np.zeros(9), [3, 0, 0]], False, trigger=True)
    assert report["actionable"] and report["cause"] == "camera_shift"


def test_online_false_negative_does_not_veto_calibrated_arm_cause():
    from types import SimpleNamespace
    report = SimpleNamespace(best="camera_shift", posterior={"camera_shift": 0.99, "none": 0.01})
    assert select_arm_cause(report, {"cause": "none", "actionable": False}) == ("camera_shift", True)
    uncertain = SimpleNamespace(best="camera_shift", posterior={"camera_shift": 0.8, "none": 0.2})
    assert select_arm_cause(uncertain, {"cause": "camera_shift", "actionable": True}) == ("camera_shift", True)


def test_new_maze_appearance_changes_pixels_without_changing_map():
    original = MazeWorld(40005, appearance_jitter=0)
    shifted = MazeWorld(40005, appearance_jitter=12)
    assert np.array_equal(original.grid, shifted.grid)
    assert not np.array_equal(original.visual_observe().image, shifted.visual_observe().image)
