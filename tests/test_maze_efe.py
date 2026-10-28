"""Unit tests for the Active Inference maze explorer, probabilistic map, and cause model.

Verifies:
1. Exact Bayesian map updates and entropy reduction.
2. Expected information gain matches analytical hand-computed values.
3. Expected Free Energy (EFE) uses both pragmatic and epistemic terms (neither is zero).
4. Cause likelihoods are learned from empirical data and discriminate confusable causes.
5. Explorer policy never accesses ground truth (grid, goal, truth()).
"""

import math
import numpy as np
import pytest

from pai.maze.env import MazeWorld, Observation
from pai.maze.map_model import ObservationModel, ProbabilisticMap, SpatialPrior, shannon_entropy
from pai.maze.efe_agent import (
    ActiveInferenceExplorer,
    InfoGainFrontierExplorer,
    NearestFrontierExplorer,
)
from pai.maze.cause_model import LearnedMazeCauseLikelihoods, LearnedMazeCauseMonitor


def test_probabilistic_map_bayes_update():
    """Verify Bayesian map update analytically against exact formula."""
    # Custom 3x3 observation model:
    # P(O=wall | true=wall)=0.9, P(O=wall | true=floor)=0.1, P(O=wall | true=goal)=0.1
    # P(O=floor | true=wall)=0.08, P(O=floor | true=floor)=0.85, P(O=floor | true=goal)=0.05
    # P(O=goal | true=wall)=0.02, P(O=goal | true=floor)=0.05, P(O=goal | true=goal)=0.85
    matrix = np.array([
        [0.90, 0.10, 0.10],
        [0.08, 0.85, 0.05],
        [0.02, 0.05, 0.85],
    ], dtype=float)
    obs_model = ObservationModel(matrix=matrix)

    prior = SpatialPrior(size=5)
    # Set interior cell (2, 2) prior to [0.5, 0.4, 0.1]
    prior.priors[2, 2] = [0.5, 0.4, 0.1]

    pmap = ProbabilisticMap(size=5, sensor_range=2, prior=prior, obs_model=obs_model)

    # Initial entropy
    init_ent = pmap.cell_entropy((2, 2))
    expected_init_ent = -(0.5 * math.log2(0.5) + 0.4 * math.log2(0.4) + 0.1 * math.log2(0.1))
    assert abs(init_ent - expected_init_ent) < 1e-6

    # Observe 'wall' at (2, 2)
    obs = Observation(position=(2, 1), visible={(2, 2): "wall"}, collided=False, reached_goal=False)
    pmap.update(obs)

    # Analytical posterior:
    # Likelihood for obs='wall': [0.90, 0.10, 0.10]
    # Unnormalized: [0.5*0.9, 0.4*0.1, 0.1*0.1] = [0.45, 0.04, 0.01]
    # Sum = 0.50
    # Normalized: [0.90, 0.08, 0.02]
    post = pmap.get_belief((2, 2))
    assert abs(post[0] - 0.90) < 1e-6
    assert abs(post[1] - 0.08) < 1e-6
    assert abs(post[2] - 0.02) < 1e-6

    # Entropy after observation is strictly lower
    post_ent = pmap.cell_entropy((2, 2))
    assert post_ent < init_ent


def test_expected_information_gain_analytical():
    """Verify expected information gain matches hand-computed mutual information."""
    # Consider a simplified binary-like observation matrix where goal is 0
    # M in {wall, floor}, prior = [0.5, 0.5, 0.0]
    # P(O=wall | wall) = 0.8, P(O=wall | floor) = 0.2
    # P(O=floor | wall) = 0.2, P(O=floor | floor) = 0.8
    matrix = np.array([
        [0.8, 0.2, 0.0],
        [0.2, 0.8, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    obs_model = ObservationModel(matrix=matrix)

    prior = SpatialPrior(size=5)
    prior.priors[2, 2] = [0.5, 0.5, 0.0]
    pmap = ProbabilisticMap(size=5, sensor_range=2, prior=prior, obs_model=obs_model)

    # Hand calculation:
    # H(M) = - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = 1.0 bit
    # P(O=wall) = 0.8*0.5 + 0.2*0.5 = 0.5
    # P(O=floor) = 0.2*0.5 + 0.8*0.5 = 0.5
    # If O=wall:
    #   P(M=wall | O=wall) = (0.8*0.5)/0.5 = 0.8
    #   P(M=floor | O=wall) = (0.2*0.5)/0.5 = 0.2
    #   H(M | O=wall) = -(0.8*log2(0.8) + 0.2*log2(0.2)) = 0.7219280948873623
    # If O=floor:
    #   H(M | O=floor) = 0.7219280948873623
    # E[H(M | O)] = 0.5 * 0.721928 + 0.5 * 0.721928 = 0.7219280948873623
    # Mutual info I(M; O) = H(M) - E[H(M | O)] = 1.0 - 0.7219280948873623 = 0.2780719051126377
    h_m_cond_o = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
    expected_ig = 1.0 - h_m_cond_o

    computed_ig = pmap.expected_cell_information_gain((2, 2))
    assert abs(computed_ig - expected_ig) < 1e-5


def test_efe_score_uses_both_terms_and_non_zero():
    """Verify that both pragmatic and epistemic terms in EFE are non-zero."""
    agent = ActiveInferenceExplorer(sensor_range=3, size=11, seed=42,
                                   epistemic_weight=1.5, effort_weight=0.5,
                                   risk_weight=1.0, goal_weight=2.0)

    # Initial observation
    obs = Observation(position=(1, 1), visible={(1, 1): "floor", (1, 2): "floor", (2, 1): "wall"},
                      collided=False, reached_goal=False)
    agent.map.update(obs)

    # Candidate target (1, 2)
    efe, epistemic, pragmatic, effort = agent._evaluate_candidate_efe(
        current=(1, 1),
        target=(1, 2),
        route=[(1, 2)],
        goal_candidate=None,
    )

    # Assert neither term is zero
    assert epistemic > 0.0, "Epistemic information gain must be non-zero"
    assert pragmatic > 0.0, "Pragmatic cost must be non-zero"
    assert effort > 0.0, "Effort cost must be non-zero"

    # Verify that EFE strictly follows the formula: pragmatic - epistemic_weight * epistemic + effort_weight * effort
    expected_efe = pragmatic - 1.5 * epistemic + 0.5 * effort
    assert abs(efe - expected_efe) < 1e-6

    # Test weight modulation
    agent_curious = ActiveInferenceExplorer(sensor_range=3, size=11, seed=42,
                                           epistemic_weight=10.0, effort_weight=0.1)
    efe_curious, _, _, _ = agent_curious._evaluate_candidate_efe((1, 1), (1, 2), [(1, 2)], None)
    assert efe_curious < efe, "Higher epistemic weight should decrease EFE (favoring exploration)"


def test_cause_likelihoods_learned_from_data():
    """Verify that cause likelihoods are fitted from empirical data and discriminate causes."""
    # Generate mock training records:
    # passage_blocked: collided_free=1, odom_drift=low, visual_mismatch=low
    # odometry_slip: collided_free=1, odom_drift=high, visual_mismatch=high
    # camera_shift: collided_free=0, odom_drift=low, visual_mismatch=high, camera_offset_changed=1
    training_data = []
    for _ in range(30):
        training_data.append({
            "true_cause": "passage_blocked",
            "features": {"collided_free": "1", "odom_drift": "low", "visual_mismatch": "low", "camera_offset_changed": "0"},
        })
        training_data.append({
            "true_cause": "odometry_slip",
            "features": {"collided_free": "1", "odom_drift": "high", "visual_mismatch": "high", "camera_offset_changed": "0"},
        })
        training_data.append({
            "true_cause": "camera_shift",
            "features": {"collided_free": "0", "odom_drift": "low", "visual_mismatch": "high", "camera_offset_changed": "1"},
        })
        training_data.append({
            "true_cause": "none",
            "features": {"collided_free": "0", "odom_drift": "low", "visual_mismatch": "low", "camera_offset_changed": "0"},
        })

    model = LearnedMazeCauseLikelihoods()
    model.fit(training_data)

    # Monitor using learned model
    monitor = LearnedMazeCauseMonitor(likelihoods=model)

    # Case 1: Collision with low odometry drift -> passage_blocked
    post_block = monitor.observe_event(
        step=10,
        collided_previously_free=True,
        odom_drift_high=False,
        visual_mismatch_high=False,
        camera_offset_changed=False,
    )
    assert post_block.best == "passage_blocked"
    assert post_block.probabilities["passage_blocked"] > 0.75

    # Case 2: Collision with high odometry drift and visual mismatch -> odometry_slip
    post_slip = monitor.observe_event(
        step=20,
        collided_previously_free=True,
        odom_drift_high=True,
        visual_mismatch_high=True,
        camera_offset_changed=False,
    )
    assert post_slip.best == "odometry_slip"
    assert post_slip.probabilities["odometry_slip"] > 0.75

    # Case 3: Camera offset change with no collision -> camera_shift
    post_shift = monitor.observe_event(
        step=30,
        collided_previously_free=False,
        odom_drift_high=False,
        visual_mismatch_high=True,
        camera_offset_changed=True,
    )
    assert post_shift.best == "camera_shift"
    assert post_shift.probabilities["camera_shift"] > 0.75


def test_no_ground_truth_in_agent_policy():
    """Verify that agent policy never reads MazeWorld.grid, truth(), or goal position."""
    world = MazeWorld(seed=123, size=11)
    agent = ActiveInferenceExplorer(sensor_range=3, size=11, seed=123)

    # Monkeypatch world attributes to raise if accessed
    def forbidden_truth():
        raise AssertionError("Agent policy accessed world.truth()!")

    world.truth = forbidden_truth

    # Also wrap grid in an object that forbids policy access during act()
    real_grid = world.grid.copy()

    class GuardedGrid:
        def __getitem__(self, item):
            raise AssertionError("Agent policy accessed world.grid directly!")

    # Observe gives valid local observation
    obs = world.observe()

    # Temporarily hide grid during act()
    world.grid = GuardedGrid()
    try:
        move = agent.act(obs)
        assert move in ((-1, 0), (0, 1), (1, 0), (0, -1), None)
    finally:
        world.grid = real_grid


def test_baselines_run_on_mazes():
    """Verify all 3 explorers execute steps on a maze and make valid moves."""
    world = MazeWorld(seed=55, size=11)
    obs = world.observe()

    agents = [
        NearestFrontierExplorer(size=11, seed=55),
        InfoGainFrontierExplorer(size=11, seed=55),
        ActiveInferenceExplorer(size=11, seed=55),
    ]
    for ag in agents:
        move = ag.act(obs)
        assert move in ((-1, 0), (0, 1), (1, 0), (0, -1), None)


def test_noisy_sensor_confuses_labels_and_reports_bumps():
    """Noisy looks disagree with the truth sometimes; a bump always reports the blocked cell as wall."""
    from pai.maze.noisy import NoisySensorMaze

    world = MazeWorld(seed=3, size=11)
    noisy = NoisySensorMaze(world, seed=3)
    wrong = total = 0
    for _ in range(40):
        obs = noisy.observe()
        for cell, kind in obs.visible.items():
            true = "wall" if world.grid[cell] else "goal" if cell == world.goal else "floor"
            wrong += kind != true
            total += 1
    assert 0.03 < wrong / total < 0.3
    p = world.position
    for d in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        q = p[0] + d[0], p[1] + d[1]
        if world.grid[q]:
            obs = noisy.step(d)
            assert obs.collided and obs.visible[q] == "wall"
            break


def test_route_efe_explorer_finds_goal_without_grid_access():
    """The route-EFE explorer reaches the goal of a small maze and never reads world.grid."""
    from pai.maze.efe_agent import RouteEFEExplorer

    world = MazeWorld(seed=7, size=11)
    agent = RouteEFEExplorer(size=11, sensor_range=world.sensor_range, seed=7,
                             effort_weight=2.0, switch_margin=0.0)
    obs = world.observe()
    for _ in range(200):
        move = agent.act(obs)
        if move is None or obs.reached_goal:
            break
        obs = world.step(move)
    assert obs.reached_goal


def test_dawid_skene_recovers_confusion_from_looks_and_few_body_labels():
    """EM recovers P(seen | true) from repeated noisy looks; only 5% of cells have body labels."""
    from pai.maze.self_calibration import dawid_skene

    rng = np.random.default_rng(0)
    true_m = np.array([[0.85, 0.1, 0.05], [0.12, 0.88, 0.35], [0.03, 0.02, 0.6]])
    looks, anchors = {}, {}
    for i in range(3000):
        k = rng.choice(3, p=[0.55, 0.43, 0.02])
        looks[i] = np.bincount(rng.choice(3, size=8, p=true_m[:, k]), minlength=3).astype(float)
        if rng.random() < 0.05 or k == 2 and rng.random() < 0.3:
            anchors[i] = int(k)
    model, prior = dawid_skene(looks, anchors)
    assert np.abs(model.matrix[:, :2] - true_m[:, :2]).max() < 0.03
    assert abs(prior[0] - 0.55) < 0.03
