import mujoco
import numpy as np
import pytest

from pai.config import REPO_ROOT, load_config
from pai.envs.articulated import ArticulatedEnv
from pai.envs.fixtures import (CLOSED_FRAC, FAMILIES, FIXTURE_TYPES, FixtureRuntime, add_fixture, fixture_types,
                               make_fixture, sample_fixture)
from pai.envs.scripted_open import run_open

CFG = REPO_ROOT / "configs" / "articulated.yaml"


def make_env(fixtures=None, *overrides):
    cfg = load_config(CFG, ["env.image_size=64", "env.render_images=false", "env.render.msaa=0",
                            "env.render.shadow_size=0", *overrides])
    return ArticulatedEnv(cfg.env, fixtures=fixtures)


def single(type_name: str, seed: int = 0, **overrides) -> ArticulatedEnv:
    return make_env({"fixture": make_fixture(np.random.default_rng(seed), type_name, **overrides)})


@pytest.fixture(scope="module")
def env():
    e = make_env()  # the configured scene: a drawer and a hinged door
    yield e
    e.close()


def _push(env, part: str, force: float, steps: int = 60, lever_torque: float = 0.0) -> float:
    """Generalised force on the part's joint (and optionally its lever) with the arm holding still."""
    ids = env.fx.ids[part]
    for _ in range(steps):
        env.data.qfrc_applied[ids["dof"]] = force
        if lever_torque:
            env.data.qfrc_applied[env.model.jnt_dofadr[ids["lever_jnt"]]] = lever_torque
        obs = env.step(np.array([0, 0, 0, 0, 1.0]))
    env.data.qfrc_applied[:] = 0
    return obs["articulations"][part]["opening"]


def _keys(x) -> set[str]:
    if isinstance(x, dict):
        return set(x) | set().union(*(_keys(v) for v in x.values()))
    return set()


def test_every_fixture_type_builds_in_a_bare_spec():
    """The library is embodiment-independent: every type builds in an empty MjSpec, rests closed and
    does not collide with its own carcass."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    rng = np.random.default_rng(0)
    fixtures = [add_fixture(spec, spec.worldbody, make_fixture(rng, t), f"fx{i}", pos=[0, 3.0 * i, 0])
                for i, t in enumerate(FIXTURE_TYPES)]
    model = spec.compile()
    data = mujoco.MjData(model)
    rt = FixtureRuntime(model, fixtures)
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert all(a["opening"] < CLOSED_FRAC for a in rt.observe(data).values())
    assert data.ncon == 0  # only the floor, which the fixtures do not touch
    assert {f.spec.family for f in fixtures} == set(FAMILIES)


def test_registry_holds_out_test_types():
    train, test = set(fixture_types(split="train")), set(fixture_types(split="test"))
    assert train and test and not train & test
    for family in FAMILIES:
        assert fixture_types(family, "train") and fixture_types(family, "test"), family
    rng = np.random.default_rng(0)
    assert all(sample_fixture(rng, split="test").type in test for _ in range(20))
    assert all(sample_fixture(rng, "cabinet_door", "train").family == "cabinet_door" for _ in range(5))
    assert not any(FIXTURE_TYPES[sample_fixture(rng, tabletop=True).type].family == "room_door" for _ in range(20))


def test_scene_observations_and_predicates(env):
    obs = env.reset(seed=0)
    assert set(obs["articulations"]) == {"drawer", "door"}
    for name, a in obs["articulations"].items():
        assert a["opening"] == pytest.approx(0.0, abs=0.01)
        assert ("closed", name) in obs["predicates"] and ("open", name) not in obs["predicates"]
        assert np.linalg.norm(a["pull_dir"]) == pytest.approx(1.0) and a["handle_pos"].shape == (3,)
        assert a["pull_dir"][0] < -0.5  # both open towards the robot
    assert {"image", "q", "ee_pos", "state", "predicates"} <= set(obs)
    # Hidden causes are ground truth only: never in observations.
    env.reset(seed=0, causes={"drawer": "locked"})
    obs = env._observe()
    assert not {"cause", "locked", "causes"} & _keys(obs)
    assert env.info["fixtures"][0]["parts"][0]["cause"] == "locked"


def test_reset_is_deterministic_per_seed():
    e = make_env(None, "env.causes={none: 0.5, locked: 0.25, stuck: 0.25}")
    runs = []
    for seed in (4, 4, 5):
        obs = e.reset(seed=seed)
        runs.append((e.info, obs["articulations"]["drawer"]["handle_pos"], obs["state"]["red"]["pos"]))
    e.close()
    assert runs[0][0] == runs[1][0]
    assert np.allclose(runs[0][1], runs[1][1]) and np.allclose(runs[0][2], runs[1][2])
    assert not np.allclose(runs[0][1], runs[2][1])


def test_objects_avoid_fixtures(env):
    for seed in range(5):
        obs = env.reset(seed=seed)
        for name, spec in env.objects.items():
            assert not env._in_keepout(obs["state"][name]["pos"][:2], 0.0), (seed, name)


def test_joints_move_under_force_unless_locked(env):
    for part, force in (("drawer", 15.0), ("door", 1.5)):
        env.reset(seed=1)
        assert _push(env, part, force) > 0.3, part
        env.reset(seed=1, causes={part: "locked"})
        assert _push(env, part, force) < 0.02, part


def test_stuck_and_blocked(env):
    env.reset(seed=1, causes={"drawer": "stuck"})
    assert _push(env, "drawer", 15.0) < 0.05
    env.reset(seed=1, causes={"drawer": "blocked"})
    assert _push(env, "drawer", 15.0, steps=100) == pytest.approx(0.3, abs=0.03)


def test_latch_releases_only_when_the_lever_is_turned():
    e = single("door_lever_down_right", 1)
    part = e.fixtures[0].target
    e.reset(seed=0, causes={part: "latched"})
    assert _push(e, part, 1.5) < 0.02
    e.reset(seed=0, causes={part: "latched"})
    assert _push(e, part, 1.5, lever_torque=0.6) > 0.3
    e.reset(seed=0)  # no latch: the same pull opens it without the lever
    assert _push(e, part, 1.5) > 0.3
    e.close()


@pytest.mark.parametrize("type_name,seed", [("drawer_bar", 0), ("drawer_knob", 1), ("door_bar_left", 1),
                                            ("door_knob_right", 2)])
def test_scripted_opener_opens_unlocked_and_fails_locked(type_name, seed):
    e = single(type_name, seed)
    part = e.fixtures[0].target
    e.reset(seed=seed)
    r = run_open(e, part)
    assert r["opened"] and r["result"] == "opened", r
    assert ("open", part) in e._observe()["predicates"]
    assert [ev["type"] for ev in e.events.events].count("articulation_opened") == 1
    assert any(ev["type"] == "grasp" and ev["object"] == part for ev in e.events.events)
    e.reset(seed=seed, causes={part: "locked"})
    r = run_open(e, part)
    assert not r["opened"] and r["peak"] < 0.05, r
    reset_event = e.events.events[0]
    assert reset_event["type"] == "fixture_reset" and reset_event["fixtures"][0]["parts"][0]["locked"]
    e.close()


def test_scripted_opener_turns_a_latched_lever():
    e = single("door_lever_up_left", 1)
    part = e.fixtures[0].target
    e.reset(seed=1, causes={part: "latched"})
    assert run_open(e, part)["opened"]
    e.close()


def test_self_closing_door_logs_opened_then_closed():
    e = single("door_bar_left", 0, spring=True, stiffness=0.35)
    part = e.fixtures[0].target
    e.reset(seed=0)
    r = run_open(e, part, hold_steps=80)
    assert r["opened"] and r["final"] < CLOSED_FRAC
    kinds = [ev["type"] for ev in e.events.of_type("articulation_opened", "articulation_closed")]
    assert kinds == ["articulation_opened", "articulation_closed"]
    rel = [ev["relation"] for ev in e.events.of_type("relation_true")]
    assert ["open", part] in rel
    e.close()


def test_segmentation_labels_fixtures(env):
    env.reset(seed=0)
    labels = set(np.unique(env.render_segmentation("top")))
    base = 2 + len(env.objects)
    assert {base + env.fixture_labels.index(n) for n in ("drawer", "door")} <= labels
