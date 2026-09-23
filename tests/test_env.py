import numpy as np
import pytest

from pai.config import load_config, REPO_ROOT
from pai.envs import PandaEnv, build_disturbances


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "configs" / "default.yaml", ["env.image_size=64"])


@pytest.fixture(scope="module")
def env(cfg):
    e = PandaEnv(cfg.env, disturbances=[])
    yield e
    e.close()


def test_reset_and_step_shapes(env):
    obs = env.reset(seed=0)
    assert obs["image"].shape == (64, 64, 3) and obs["image"].dtype == np.uint8
    assert obs["q"].shape == (env.n_active,)
    obs = env.step(np.zeros(env.n_active))
    assert np.isfinite(obs["q"]).all()


def test_velocity_action_moves_joint(env):
    obs = env.reset()
    q0 = obs["q"].copy()
    for _ in range(50):
        obs = env.step(np.array([0.5] + [0.0] * (env.n_active - 1)))
    assert obs["q"][0] - q0[0] == pytest.approx(0.5 * 50 * env.control_dt, abs=0.05)
    assert np.abs(obs["q"][1:] - q0[1:]).max() < 0.05


def test_render_at_does_not_change_state(env):
    env.reset()
    qpos = env.data.qpos.copy()
    img_home = env.render()
    img_other = env.render_at(env.sample_q(np.random.default_rng(0)))
    assert np.array_equal(env.data.qpos, qpos)
    assert np.array_equal(env.render(), img_home)
    assert not np.array_equal(img_other, img_home)


def test_sample_q_within_limits(env):
    rng = np.random.default_rng(0)
    qs = np.stack([env.sample_q(rng) for _ in range(200)])
    assert (qs >= env.joint_range[:, 0]).all() and (qs <= env.joint_range[:, 1]).all()


def test_disturbances_apply_and_reset(cfg):
    specs = [
        {"type": "lighting", "start": 0, "scale": 0.2},
        {"type": "camera_shift", "start": 0, "offset": [0, 0.1, 0]},
        {"type": "payload", "start": 0, "mass": 3.0},
        {"type": "occlusion", "start": 0, "box": [0, 0, 0.5, 0.5], "color": [255, 0, 0]},
    ]
    env = PandaEnv(cfg.env, disturbances=build_disturbances(specs))
    env.reset()
    nominal_light = env.model.light_diffuse.copy()
    nominal_cam = env.model.cam_pos.copy()
    obs = env.step(np.zeros(env.n_active))
    assert (env.model.light_diffuse < nominal_light + 1e-9).all()
    assert env.model.cam_pos[env.cam_id][1] == pytest.approx(nominal_cam[env.cam_id][1] + 0.1)
    assert (obs["image"][:32, :32] == [255, 0, 0]).all()
    env.reset()
    assert np.allclose(env.model.light_diffuse, nominal_light)
    assert np.allclose(env.model.cam_pos, nominal_cam)
    env.close()


def test_push_moves_arm(cfg):
    env = PandaEnv(cfg.env, disturbances=build_disturbances(
        [{"type": "push", "start": 0, "duration": 20, "body": "link5", "force": [0, 60, 0]}]))
    env.reset()
    ee0 = env.data.xpos[env.ee_body].copy()
    for _ in range(20):
        env.step(np.zeros(env.n_active))
    assert np.linalg.norm(env.data.xpos[env.ee_body] - ee0) > 0.005
    env.close()
