"""Visual sensing, mobility, causal adaptation and transfer contracts."""

from pai.maze.advanced import run_advanced_episode
from pai.maze.env import MazeWorld
from pai.maze.mobility import ContinuousMobile
from pai.maze.visual import VisualExplorer, decode


def test_visual_observation_has_no_absolute_pose_and_decodes_landmarks():
    world = MazeWorld(4, size=11)
    image = world.visual_observe()
    assert not hasattr(image, "position")
    labels = decode(image.image)
    assert labels.shape == (7, 7)
    assert len(world.landmarks) > 0


def test_mobile_skill_moves_continuously_and_reports_stall():
    world = MazeWorld(2, size=11)
    body = ContinuousMobile(world, speed=2.0, dt=0.1)
    move = next(d for d in ((-1, 0), (0, 1), (1, 0), (0, -1)) if not world.grid[1 + d[0], 1 + d[1]])
    result = body.move(move)
    assert result.executed and not result.stalled and len(result.path) > 2
    assert tuple(body.xy) == world.position
    back = (-move[0], -move[1])
    body.move(back)
    blocked = body.move((-1, 0))
    assert blocked.stalled and not blocked.executed


def test_visual_mobile_explorer_closes_loops_and_completes():
    world = MazeWorld(0, size=11)
    result = run_advanced_episode(world, visual=True, mobile=True, budget=180, odom_noise=0.35)
    assert result["visited"] == result["reachable"] and result["goal_found"]
    assert result["loop_closures"] > 0
    assert result["mean_pose_error"] == 0
    assert result["mean_raw_odom_error"] > 0


def test_disturbance_explanations_and_replanning():
    world = MazeWorld(0, size=21)
    result = run_advanced_episode(world, visual=True, mobile=True, budget=600,
                                  block_at=30, shift_at=50, odom_noise=0.35)
    assert set(result["truth_causes"]) == {"passage_blocked", "camera_shift"}
    assert all(result["cause_recall"].values())
    assert result["visited"] == result["reachable"]
    assert result["goal_found"]


def test_visual_policy_does_not_receive_world_coordinates():
    world = MazeWorld(1, size=11)
    agent = VisualExplorer(size=11)
    command = agent.act(world.visual_observe())
    assert command is not None
    assert agent.localizer.position == world.start


def test_clean_visual_run_does_not_invent_causes():
    result = run_advanced_episode(MazeWorld(0, size=11), visual=True, mobile=True,
                                  budget=180, odom_noise=0.35)
    assert result["truth_causes"] == []
    assert result["cause_reports"] == []


def test_shifted_repetitive_corridor_recovers_from_body_outcomes():
    world = MazeWorld(40005, size=21, sensor_range=3, loop_probability=0.05,
                      appearance_jitter=12.0)
    result = run_advanced_episode(world, visual=True, mobile=True, budget=700,
                                  block_at=24, shift_at=42, odom_noise=0.35, seed=40005)
    assert result["goal_found"]
    assert result["visited"] == result["reachable"]
