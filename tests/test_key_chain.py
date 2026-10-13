"""End-to-end navigation and manipulation through a locked maze passage."""

import pytest

from pai.discovery.mock import MockWorld, Scenario
from pai.maze.env import MazeWorld
from pai.maze.key_chain import KeyChainMaze, LocalFixture, _drawer_cell, run_key_chain
from scripts.key_chain_stage4_eval import evaluate as evaluate_stage4


@pytest.mark.parametrize("seed", range(6))
def test_key_chain_reaches_exit_after_opening_drawer_and_door(seed):
    maze = MazeWorld(seed=seed, size=15)
    fixture = MockWorld(Scenario("hd_lever_pull_right", "both", "drawer",
                                 n_keys=1, seed=seed, p_fail=0.0, pen=False))
    result = run_key_chain(maze, fixture, seed=seed, move_budget=500,
                           fixture_budget=120)
    assert result.success, result
    assert result.drawer_open and result.door_open and result.entered_exit
    assert [event["stage"] for event in result.events].index("carry_object") < [
        event["stage"] for event in result.events].index("open_door")
    assert not [event for event in maze.events if event["type"] == "collision"]


def test_locked_passage_blocks_entry_until_fixture_opens():
    maze = MazeWorld(seed=0, size=15)
    scene = KeyChainMaze(maze, _drawer_cell(maze))
    adjacent = next(cell for cell in maze.distances_from(maze.start)
                    if abs(cell[0] - scene.door[0]) + abs(cell[1] - scene.door[1]) == 1)
    maze.position = adjacent
    move = (scene.door[0] - adjacent[0], scene.door[1] - adjacent[1])
    blocked = scene.step(move)
    assert blocked.collided and maze.position == adjacent
    scene.door_open = True
    passed = scene.step(move)
    assert passed.reached_goal and maze.position == scene.door


def test_fixture_rejects_remote_actions():
    fixture = MockWorld(Scenario("hd_lever_pull_right", "both", "drawer",
                                 n_keys=1, seed=0, p_fail=0.0, pen=False))
    drawer = LocalFixture(fixture, {fixture.pid["drawer"]},
                          f"part:{fixture.pid['drawer']}.open", set())
    assert all(action[1] == fixture.pid["drawer"] for action in drawer.actions())
    with pytest.raises(ValueError, match="out of reach"):
        drawer.execute(("push", fixture.pid["door"]))


def test_held_out_door_and_wrong_key_recovery_with_recipe_transfer():
    report = evaluate_stage4(range(20, 22), size=21)
    summary = report["summary"]
    assert summary["n"] == 6
    assert summary["cold_passed"] == summary["warm_passed"] == 6
    assert summary["wrong_key_reports"] == 6
    assert summary["wrong_key_explanation_accuracy"] == 1.0
    assert summary["warm_mean_fixture_actions"] < summary["cold_mean_fixture_actions"]
