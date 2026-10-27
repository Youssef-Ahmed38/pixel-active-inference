"""Stage 2 locomotion: env, goals/preferences, policy I/O, body model, surprise and fall causes."""

import numpy as np
import pytest
import torch

from pai.locomotion.body import load_body, randomized_models
from pai.locomotion.env import (ACT_DIM, OBS_DIM, PROPRIO, EnvConfig, LocomotionEnv,
                                desired_foot_heights, gait_params)
from pai.locomotion.falls import (CAUSES, Calibration, LocomotionEventLog, SurpriseMonitor,
                                  posterior)
from pai.locomotion.policy import ActorCritic, LocomotionController, save_policy
from pai.locomotion.world_model import EnsembleBodyModel, ReplayBuffer, train_steps


@pytest.fixture(scope="module")
def body():
    return load_body()


def small_env(body, n=4, **kw):
    cfg = EnvConfig(nthread=1, n_variants=2, **kw)
    return LocomotionEnv(n, cfg, seed=3, body=body)


def test_body_has_twelve_learned_leg_servos_and_foot_sensors(body):
    assert len(body.leg_ctrl) == ACT_DIM == 12
    assert len(body.foot_floor_pairs) == 6
    ms = randomized_models(body, 3, np.random.default_rng(0))
    assert len({float(m.pair_friction[body.foot_floor_pairs[0], 0]) for m in ms}) == 3
    assert all(m.nq == body.model.nq for m in ms)


def test_env_step_shapes_and_command_in_observation(body):
    env = small_env(body)
    obs = env.observe()
    assert obs.shape == (4, OBS_DIM) and obs.dtype == np.float32
    env.set_command(np.arange(4), 0.7, 0.1, -0.2)
    o, r, done, info = env.step(np.zeros((4, ACT_DIM)))
    assert o.shape == (4, OBS_DIM) and r.shape == (4,) and done.shape == (4,)
    np.testing.assert_allclose(o[:, 9:12], [[0.7, 0.1, -0.2]] * 4, atol=1e-6)
    assert np.isfinite(o).all()
    assert len(PROPRIO) == 33


def test_heading_goal_becomes_a_bounded_yaw_rate(body):
    env = small_env(body, n=2)
    yaw = env.yaw()
    env.set_command([0, 1], 0.5, heading=yaw + np.array([0.3, 3.0]))
    cmd = env.effective_command()
    assert cmd[0, 2] == pytest.approx(0.3, abs=1e-6)       # proportional
    assert cmd[1, 2] == pytest.approx(1.0)                 # saturated at 1 rad/s


def test_tipped_body_is_a_fall_and_is_reset(body):
    env = small_env(body, n=2, randomize=False)
    qp = env.qpos()
    qp[0, 3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]   # lying on its side
    _, r, done, info = env.step(np.zeros((2, ACT_DIM)))
    assert info["fell"][0] and done[0] and not info["fell"][1]
    assert "final_obs" in info and env.steps[0] == 0
    assert env.qpos()[0, 2] > 0.6


def test_gait_rhythm_runs_have_flight_and_standing_keeps_feet_down():
    f_walk, d_walk = gait_params(np.array([0.5]))
    f_run, d_run = gait_params(np.array([2.0]))
    assert f_run > f_walk and d_walk[0] == pytest.approx(0.5) and d_run[0] < 0.5
    z = desired_foot_heights(np.linspace(0, 1, 50), np.full(50, 0.5), 0.08, np.zeros(50, bool))
    assert np.all(z == 0)
    z = desired_foot_heights(np.linspace(0, 1, 50), np.full(50, 0.5), 0.08, np.ones(50, bool))
    assert z.max() <= 0.08 + 1e-9 and np.all((z[:, 0] == 0) | (z[:, 1] == 0))   # one foot down


def test_preference_prefers_matching_the_commanded_velocity(body):
    env = small_env(body, n=2, randomize=False)
    env.set_command([0, 1], 1.0, 0.0, 0.0)
    env.step(np.zeros((2, ACT_DIM)))
    k = env.kinematics()
    k["v_head"] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    cmd = env.effective_command()
    moving = np.ones(2, bool)
    _, terms = env._reward(k, cmd, np.zeros((2, ACT_DIM)), moving, np.full(2, 0.5))
    assert terms["lin"][0] == pytest.approx(1.0) and terms["lin"][1] < 0.1


def test_policy_io_and_frozen_controller_roundtrip(tmp_path):
    torch.manual_seed(0)
    ac = ActorCritic(hidden=(32, 32))
    obs = torch.randn(5, OBS_DIM)
    a, logp, v = ac.act(obs)
    assert a.shape == (5, ACT_DIM) and logp.shape == (5,) and v.shape == (5,)
    save_policy(tmp_path / "p.pt", ac)
    ctrl = LocomotionController(tmp_path / "p.pt")
    out = ctrl.act(obs.numpy())
    det, _, _ = ac.act(obs, deterministic=True)
    np.testing.assert_allclose(out, det.numpy(), atol=1e-6)


def test_body_model_learns_and_disagrees_less_on_seen_data():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    s = rng.normal(size=(2000, 33)).astype(np.float32)
    a = rng.normal(size=(2000, ACT_DIM)).astype(np.float32)
    s2 = s + 0.1 * np.tanh(a @ rng.normal(size=(ACT_DIM, 33))).astype(np.float32)
    m = EnsembleBodyModel(members=3, hidden=64)
    m.fit_normalizers(*(torch.as_tensor(x) for x in (s, a, s2)))
    buf = ReplayBuffer(2000)
    buf.add(s, a, s2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = train_steps(m, opt, buf, 5, 256, rng, "cpu")
    last = train_steps(m, opt, buf, 300, 256, rng, "cpu")
    assert last < 0.5 * first
    _, dis_seen = m.surprise_and_disagreement(torch.as_tensor(s[:200]), torch.as_tensor(a[:200]))
    _, dis_new = m.surprise_and_disagreement(torch.as_tensor(s[:200] * 6), torch.as_tensor(a[:200] * 6))
    assert dis_new.mean() > dis_seen.mean()


class StillModel:
    """Predicts 'nothing changes' and no surprise: lets the test control the residuals."""

    def predict(self, s, a):
        return s

    def surprise_and_disagreement(self, s, a, s2=None):
        return ((s2 - s) ** 2).mean(-1), torch.zeros(len(s))


def _feed(mon, steps, jump_at=None, slide=0.0):
    obs = np.zeros((1, OBS_DIM), np.float32)
    h = np.array([[0.0, 0.1]])
    for t in range(steps):
        nxt = obs.copy()
        if t == jump_at:
            nxt[0, 6] += 1.0                                   # pelvis velocity jumps by 1 m/s
        v = np.zeros((1, 2, 3))
        v[0, 0, 0] = slide
        mon.update(obs, np.zeros((1, ACT_DIM)), nxt, h, v)
        obs = nxt


def test_surprise_monitor_measures_unexplained_impulse_and_sliding():
    mon = SurpriseMonitor(StillModel(), 1, window=30)
    _feed(mon, 40, jump_at=35)
    U, S = mon.features([0])[0]
    assert U == pytest.approx(1.0, abs=1e-5) and S == 0.0
    mon.reset([0])
    _feed(mon, 40, slide=0.8)
    U, S = mon.features([0])[0]
    assert U == 0.0 and S == pytest.approx(0.8)


def test_cause_posterior_separates_push_slip_and_gait():
    cal = Calibration.fit(np.exp(np.random.default_rng(1).normal([np.log(0.1), np.log(0.05)], 0.3, (200, 2))))
    p = posterior(np.array([[0.1, 0.05], [1.5, 0.05], [0.15, 1.0]]), cal)
    assert [CAUSES[i] for i in p.argmax(1)] == ["gait", "push", "slip"]
    np.testing.assert_allclose(p.sum(1), 1.0)


def test_event_log_records_contacts_slips_and_falls():
    log = LocomotionEventLog()
    v0 = np.zeros((2, 3))
    log.observe(0.02, np.array([0.1, 0.0]), v0)                # left foot lifts
    log.observe(0.04, np.array([0.0, 0.0]), v0)                # and lands
    vs = np.zeros((2, 3)); vs[1, 0] = 1.0
    for k in range(4):
        log.observe(0.06 + 0.02 * k, np.array([0.0, 0.0]), vs)  # right foot slides
    log.observe(0.2, np.array([0.0, 0.0]), v0, fell=True)
    kinds = [e["type"] for e in log.events]
    assert kinds == ["contact_end", "contact_start", "slip", "fall"]
    assert log.of_type("slip")[0]["foot"] == "right_foot"


def test_skill_api_moves_goals_into_the_observation(tmp_path, body):
    from pai.locomotion.skill import LocomotionSkill
    save_policy(tmp_path / "p.pt", ActorCritic(hidden=(16,)))
    skill = LocomotionSkill(tmp_path / "p.pt", seed=1)
    skill.move(1.2, heading=0.5)
    assert skill.env.command[0, 0] == pytest.approx(1.2) and skill.env.heading_mode[0]
    out = skill.run(0.1)
    assert set(out) >= {"x", "y", "heading", "speed", "fell"} and out["t"] == pytest.approx(0.1)
    assert skill.obs[0, 9] == pytest.approx(1.2)
    skill.stop()
    assert not skill.env.heading_mode[0] and np.all(skill.env.command[0] == 0)
