import numpy as np
import pytest

from pai.bike import BicycleEnv, BalanceController
from pai.cognition.contracts import GoalRelation, SkillCommand
from pai.cognition.controller import CognitiveController


def test_free_frame_and_reset_sensor_orientation():
    env = BicycleEnv()
    assert env.model.nq == 11 and env.model.nv == 10 and env.model.nu == 3
    assert all(int(joint) > 0 for joint in env.model.actuator_trnid[:, 0])
    initial = env.reset(roll=.02)
    assert initial[0] == pytest.approx(.02)
    assert not env.fallen
    env.execute(SkillCommand("ride", parameters=(0, 0, 0)))
    assert env.data.time == pytest.approx(.02)
    assert np.all(env.data.xfrc_applied == 0)
    assert np.all(env.data.qfrc_applied == 0)


def test_balanced_and_uncontrolled_use_same_initial_state():
    env = BicycleEnv()
    env.reset(seed=0)
    initial = env.data.qpos.copy()
    controller = CognitiveController()
    result = controller.run(env, GoalRelation("distance_reached", "rider"), BalanceController(), budget=500)
    assert result.success and env.travelled >= 10 and not env.fallen
    env.reset(seed=0)
    np.testing.assert_array_equal(env.data.qpos, initial)
    result = controller.run(env, GoalRelation("distance_reached", "rider"),
        lambda state, adapter: None if adapter.done else SkillCommand("ride", parameters=(0, 0, 0)), budget=500)
    assert not result.success and env.fallen
    assert len(env.events) == 1 and env.events[0]["type"] == "fall"


def test_fallen_rider_cannot_succeed_or_continue():
    env = BicycleEnv()
    env.reset(roll=.8)
    assert env.fallen
    assert not env.observe().satisfies(GoalRelation("distance_reached", "rider"))
    outcome = env.execute(SkillCommand("ride", parameters=(0, 0, 0)))
    assert not outcome.executed and outcome.duration == 0


@pytest.mark.parametrize("parameters", [(0,), (0, float("nan"), 0)])
def test_invalid_controls(parameters):
    with pytest.raises(ValueError):
        BicycleEnv().execute(SkillCommand("ride", parameters=parameters))


def test_inverse_dynamics_recovers_known_servo_targets():
    import mujoco
    from pai.bike.observation_learning import infer_action
    env = BicycleEnv()
    env.data.qpos[2] = 2  # No contact forces for this analytic actuator check.
    action = np.array([1.5, .04, -.02])
    env.data.ctrl[:] = action
    mujoco.mj_forward(env.model, env.data)
    recovered = infer_action(env, env.data.qpos.copy(), env.data.qvel.copy(), env.data.qacc.copy())
    np.testing.assert_allclose(recovered, action, atol=1e-7)


def test_student_never_uses_expert_control_labels(tmp_path):
    from pai.bike.observation_learning import ObservedRidingPolicy
    rng = np.random.default_rng(1)
    sensations = rng.normal(size=(20, 6))
    inferred = rng.normal(size=(20, 3))
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, sensations=sensations, inferred_actions=inferred, audit_expert_actions=np.zeros((20, 3)))
    np.savez(b, sensations=sensations, inferred_actions=inferred, audit_expert_actions=np.full((20, 3), 1000))
    first, second = ObservedRidingPolicy.fit(a), ObservedRidingPolicy.fit(b)
    np.testing.assert_array_equal(first.weights, second.weights)
    checkpoint = tmp_path / "policy.json"
    first.save(checkpoint)
    np.testing.assert_array_equal(ObservedRidingPolicy.load(checkpoint).action(sensations[0]), first.action(sensations[0]))


def test_student_uses_decision_boundaries_not_post_action_states(tmp_path):
    from pai.bike.observation_learning import ObservedRidingPolicy
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(20, 6)), rng.normal(size=(20, 3))
    boundaries = np.arange(20) % 2 == 0
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, sensations=x, inferred_actions=y, control_boundaries=boundaries)
    x[~boundaries], y[~boundaries] = 1000, -1000
    np.savez(b, sensations=x, inferred_actions=y, control_boundaries=boundaries)
    np.testing.assert_array_equal(ObservedRidingPolicy.fit(a).weights, ObservedRidingPolicy.fit(b).weights)


def test_practice_preserves_incumbent_when_candidates_regress(monkeypatch):
    import pai.bike.practice as module
    from pai.bike.observation_learning import ObservedRidingPolicy
    initial = ObservedRidingPolicy(np.zeros((7, 3)))
    def trials(policy, seeds):
        return [], (1, -float(np.linalg.norm(policy.weights)))
    monkeypatch.setattr(module, "riding_trials", trials)
    result = module.practice(initial, [0], updates=3)
    assert result.first_all_successful_attempt == 0
    assert all(not row["accepted"] for row in result.trials[1:])
    np.testing.assert_array_equal(initial.weights, result.policy.weights)
