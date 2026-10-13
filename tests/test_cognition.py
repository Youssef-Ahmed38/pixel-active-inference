"""Shared contracts and memory must accept both domains without simulator truth as policy input."""

from types import SimpleNamespace

import numpy as np
import pytest

from pai.cognition.adapters import from_arm_record
from pai.cognition.adapters import MazeDomainAdapter
from pai.cognition.causes import CausePosterior
from pai.cognition.contracts import GoalRelation
from pai.cognition.controller import CognitiveController, MazeFrontierPolicy
from pai.cognition.memory import CognitiveMemory
from pai.maze.advanced import run_advanced_episode
from pai.maze.env import MazeWorld
from pai.planning.objective import expected_free_energy


def test_shared_efe_accounting_matches_both_domains():
    assert expected_free_energy(10, 2, 3, 4, 0.5, 0.1, 0.2) == pytest.approx(10.1)
    assert expected_free_energy(0, 3) < expected_free_energy(0, 2)


def test_cause_posterior_needs_confident_evidence():
    p = CausePosterior.from_log_evidence({"none": -8, "camera_shift": 0, "unknown": -4})
    assert p.best == "camera_shift" and p.should_adapt(0.9)
    weak = CausePosterior.from_log_evidence({"none": 0, "camera_shift": 0})
    assert not weak.should_adapt(0.9)


def test_memory_stores_arm_and_maze_records(tmp_path):
    tokens = np.zeros((2, 3, 19), np.float32)
    record = SimpleNamespace(goal="on(red, plate)", episode=0, success=True, steps=1,
                             tokens=tokens, actions=np.zeros((1, 4), np.float32),
                             surprise=[0.1], cause_posterior={"camera_shift": 1.0},
                             adaptations=[], events=[])
    memory = CognitiveMemory(tmp_path / "episodes.jsonl")
    memory.add(from_arm_record(record, ["gripper", "red", "plate"]))
    assert memory.episodes[0].commands[0].parameters == (0.0, 0.0, 0.0, 0.0, 0.0)
    world = MazeWorld(0, size=11)
    result = run_advanced_episode(world, budget=180, cognitive_memory=memory)
    assert result["goal_found"] and len(memory.episodes) == 2
    assert len(memory.recall(GoalRelation("on", "red", "plate"))) == 1
    assert not memory.recall(GoalRelation("on", "red", "bowl"))
    assert len(memory.recall_cause("camera_shift", domain="arm")) == 1
    assert memory.episodes[1].domain == "maze" and memory.episodes[1].success
    assert memory.path.read_text().count("\n") == 2


def test_maze_thinker_uses_observed_evidence_not_cause_reports(monkeypatch):
    from pai.cognition import thinker_data
    from pai.cognition.contracts import CognitiveEpisode, SkillCommand, SurpriseEvidence

    def fake_episode(world, **kwargs):
        memory = kwargs["cognitive_memory"]
        episode = CognitiveEpisode("maze", kwargs["seed"], GoalRelation("cover", "maze"),
                                   True, 2, evidence=[SurpriseEvidence(1, ("motion_stall",),
                                                                         (0.0,), (0.1,))],
                                   adaptations=[{"step": 1, "cause": "camera_shift"}])
        memory.add(episode)
        return {"truth_causes": ["camera_shift", "passage_blocked"]}

    monkeypatch.setattr(thinker_data, "run_advanced_episode", fake_episode)
    data = thinker_data.maze_evidence(n_per_cause=1)
    assert data.x.shape[0] == 3
    assert not data.x[:, :, :8].any()


def test_shared_controller_reaches_maze_goal_with_skill_adapter():
    world = MazeWorld(7, size=11)
    adapter = MazeDomainAdapter(world)
    policy = MazeFrontierPolicy(world.size, world.sensor_range, seed=7)
    record = CognitiveController().run(adapter, GoalRelation("goal_reached", "body"),
                                       policy, budget=180)
    assert record.success
    assert record.commands and all(c.name == "move" for c in record.commands)
    assert all(b.domain == "maze" for b in record.beliefs)
