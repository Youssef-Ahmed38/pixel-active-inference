"""Every embodiment behind the common interface: builds, is stable, tracks the palm, closes, reaches."""

import numpy as np
import pytest

from pai.envs.door_scene import DoorSceneEnv, library_fixture
from pai.envs.embodiments import EMBODIMENTS, SPECS, rot_error, rotvec_to_mat
from pai.envs.scripted_reach import ScriptedReachGraspPull

PARAMS = [pytest.param(n, marks=pytest.mark.skipif(not SPECS[n].available(), reason=f"{SPECS[n].xml} missing "
                                                     "(run scripts/fetch_assets.py)")) for n in EMBODIMENTS]
_ENVS: dict = {}


def door_env(name: str) -> DoorSceneEnv:
    if name not in _ENVS:
        _ENVS[name] = DoorSceneEnv(name, "door", seed=0)
    return _ENVS[name]


def hold(n_extra: int, grasp: float = 1.0) -> np.ndarray:
    return np.r_[np.zeros(6), np.full(n_extra, grasp)]


def drive(env, obs, pos, rot, steps, grasp=1.0, kp=4.0, kr=3.0):
    for _ in range(steps):
        a = np.r_[kp * (pos - obs["palm_pos"]), kr * rot_error(obs["palm_rot"], rot), grasp]
        obs = env.step(a)
    return obs


@pytest.mark.parametrize("name", PARAMS)
def test_builds_and_is_stable(name):
    env = door_env(name)
    obs = env.reset(seed=0)
    assert obs["n_fingers"] == SPECS[name].n_fingers
    assert obs["finger_closure"].shape == (obs["n_fingers"],)
    assert obs["finger_force"].shape == (obs["n_fingers"],)
    assert obs["finger_q"].size >= obs["n_fingers"]
    start = obs["palm_pos"].copy()
    for _ in range(200):
        obs = env.step(hold(1))
        assert env.stable()
    assert np.all(np.isfinite(env.data.qpos))
    assert np.linalg.norm(obs["palm_pos"] - start) < 0.02   # holding still stays still
    assert obs["fixture"]["opening"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("name", PARAMS)
def test_palm_tracks_commanded_pose(name):
    env = door_env(name)
    obs = env.reset(seed=0)
    n = env.approach0
    target = obs["palm_pos"] - 0.04 * n + np.array([0.0, 0.03, 0.05])
    ax = n / np.linalg.norm(n)
    rot = rotvec_to_mat(0.3 * ax) @ obs["palm_rot"]
    obs = drive(env, obs, target, rot, 100)
    assert np.linalg.norm(obs["palm_pos"] - target) < 0.015
    assert np.linalg.norm(rot_error(obs["palm_rot"], rot)) < 0.1


@pytest.mark.parametrize("name", PARAMS)
def test_grasp_command_closes_fingers(name):
    env = door_env(name)
    obs = env.reset(seed=0)
    assert obs["opening"] > 0.9
    for _ in range(40):
        obs = env.step(hold(1, -1.0))
    assert obs["opening"] < 0.2
    assert np.all(obs["finger_closure"] > 0.6)   # in free air fingers may meet each other (Shadow)
    for _ in range(40):
        obs = env.step(hold(1, 1.0))
    assert obs["opening"] > 0.9


@pytest.mark.parametrize("name", PARAMS)
def test_per_finger_commands(name):
    env = door_env(name)
    obs = env.reset(seed=0)
    nf = obs["n_fingers"]
    g = np.ones(nf)
    g[0] = -1.0  # close only the first finger
    for _ in range(40):
        obs = env.step(np.r_[np.zeros(6), g])
    c = obs["finger_closure"]
    if len(SPECS[name].synergy) == 1:  # one actuator for both fingers: commands are averaged
        assert np.allclose(c, c[0], atol=0.05)
    else:
        assert c[0] > 0.7 and np.all(c[1:] < 0.3)


@pytest.mark.parametrize("name", PARAMS)
def test_scripted_reach_gets_palm_to_handle(name):
    env = door_env(name)
    obs = env.reset(seed=0)
    policy = ScriptedReachGraspPull(env)
    for _ in range(400):
        obs = env.step(policy.act(obs))
        if policy.phase == "close":
            break
    assert policy.phase == "close"
    assert np.linalg.norm(obs["palm_pos"] - obs["fixture"]["handle_pos"]) < 0.03
    assert env.stable()


def test_library_fixture_adapter():
    name = next((n for n in ("shadow", "robotiq_2f85", "allegro", "leap", "panda_2f") if SPECS[n].available()), None)
    if name is None:
        pytest.skip("no embodiment assets")
    fixtures = pytest.importorskip("pai.envs.fixtures")
    assert "drawer_bar" in fixtures.FIXTURE_TYPES
    env = DoorSceneEnv(name, library_fixture("drawer_bar"), seed=0)
    f = env.reset()["fixture"]
    assert f["opening"] == pytest.approx(0.0, abs=0.01)
    assert f["handle_approach"][0] > 0.99          # front faces the body
    assert f["open_dir"][0] < -0.99                # the drawer comes out towards it
    env.close()
