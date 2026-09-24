"""Generic skills (pai.skills) on an arm (panda_2f) and a dexterous hand (shadow): reach, grasp, turn, pull,
release, retreat on the door_scene test door and drawer, stall on a drawer held shut, insert with a peg."""

import numpy as np
import pytest

from pai.envs.door_scene import DoorSceneEnv, add_test_drawer, library_fixture
from pai.envs.embodiments import SPECS, rot_error
from pai.skills import SKILLS, SkillContext, StallMonitor, Target, handle_target, options, run_skill
from pai.skills.scenes import BLOCK, PEG, SLOT, hand_peg, let_go_holder, locked, with_peg


def _body(n):
    return pytest.param(n, marks=pytest.mark.skipif(not SPECS[n].available(), reason=f"{SPECS[n].xml} missing "
                                                     "(run scripts/fetch_assets.py)"))


BODIES = [_body("panda_2f"), _body("shadow")]           # an arm with a parallel gripper, a 5-finger hand
GRIPPERS = [_body("panda_2f"), _body("robotiq_2f85")]  # insert: hold a loose peg (see test_insert)
_ENVS: dict = {}


def env_for(body: str, scene: str) -> DoorSceneEnv:
    if (body, scene) not in _ENVS:
        adder = {"door": "door", "drawer": "drawer", "locked": locked(add_test_drawer),
                 "lib_locked": library_fixture("drawer_bar"), "peg": with_peg(add_test_drawer)}[scene]
        _ENVS[(body, scene)] = DoorSceneEnv(body, adder, seed=0)
    return _ENVS[(body, scene)]


def start(body: str, scene: str):
    env = env_for(body, scene)
    ctx = SkillContext(env, env.reset(seed=0))
    return env, ctx, handle_target(env.fixture)


def reach_and_grasp(ctx, h):
    r = run_skill(ctx, "reach", h)
    g = run_skill(ctx, "grasp", h)
    return r, g


# ---------------------------------------------------------------------------------------------- per skill
@pytest.mark.parametrize("scene", ["door", "drawer"])
@pytest.mark.parametrize("body", BODIES)
def test_reach_gets_palm_to_handle(body, scene):
    env, ctx, h = start(body, scene)
    r = run_skill(ctx, "reach", h, tol=0.01)
    assert r.success and not r.stalled, r.row()
    assert r.errors["pos_err"] < 0.01
    # the skill's own error agrees with the privileged handle position
    assert np.linalg.norm(ctx.obs["palm_pos"] - env.fixture_state()["handle_pos"]) < 0.01
    assert env.stable()


@pytest.mark.parametrize("body", BODIES)
def test_grasp_closes_on_handle(body):
    env, ctx, h = start(body, "drawer")
    _, g = reach_and_grasp(ctx, h)
    assert g.success, g.row()
    assert g.errors["fingers_in_contact"] >= 2
    assert g.errors["closure_change"] > 0.5
    assert env.handle_contact_fingers() >= 2


@pytest.mark.parametrize("body", BODIES)
def test_grasp_in_free_air_reports_no_contact(body):
    env, ctx, _ = start(body, "door")
    g = run_skill(ctx, "grasp")
    assert not g.success and g.reason == "no_contact"
    assert g.errors["closure"] > 0.5


@pytest.mark.parametrize("body", BODIES)
def test_turn_changes_palm_angle(body):
    env, ctx, h = start(body, "door")
    reach_and_grasp(ctx, h)
    R0 = ctx.obs["palm_rot"].copy()
    t = run_skill(ctx, "turn", h, angle=0.4)
    axis = env.fixture_state()["handle_axis"]
    turned = float(np.dot(rot_error(R0, ctx.obs["palm_rot"]), axis))
    assert t.success and not t.stalled, t.row()
    assert turned == pytest.approx(0.4, abs=0.1)
    assert t.errors["pivot_drift"] < 0.03        # rotated about the grasped point, not swung away


@pytest.mark.parametrize("body", BODIES)
def test_pull_opens_unlocked_drawer(body):
    env, ctx, h = start(body, "drawer")
    reach_and_grasp(ctx, h)
    p = run_skill(ctx, "push_pull", h, direction="pull", distance=0.15)
    assert p.success and not p.stalled, p.row()
    f = env.fixture_state()
    assert f["opened"] and f["opening"] > 0.12


@pytest.mark.parametrize("body", BODIES)
def test_pull_on_locked_drawer_stalls(body):
    env, ctx, h = start(body, "locked")
    _, g = reach_and_grasp(ctx, h)
    assert g.success, g.row()
    p = run_skill(ctx, "push_pull", h, direction="pull", distance=0.15)
    assert p.stalled and not p.success, p.row()
    assert p.errors["target_moved"] < 0.005
    assert p.duration < 2.0                      # noticed early, not after the whole pull
    assert abs(env.fixture_state()["opening"]) < 0.005


@pytest.mark.parametrize("body", BODIES)
def test_pull_on_library_locked_cause_stalls(body):
    """The fixture library's own "locked" cause: its joint-equality lock, switched on for the episode."""
    env = env_for(body, "lib_locked")
    obs = env.reset(seed=0)
    env.data.eq_active[env.model.equality(f"{env.fixture.joint}_lock").id] = 1
    ctx = SkillContext(env, obs)
    h = handle_target(env.fixture)
    try:
        _, g = reach_and_grasp(ctx, h)
        assert g.success, g.row()
        p = run_skill(ctx, "push_pull", h, direction="pull", distance=0.15)
        assert p.stalled and not p.success, p.row()
    finally:
        env.data.eq_active[env.model.equality(f"{env.fixture.joint}_lock").id] = 0


@pytest.mark.parametrize("body", BODIES)
def test_release_then_retreat(body):
    env, ctx, h = start(body, "door")
    reach_and_grasp(ctx, h)
    r = run_skill(ctx, "release")
    assert r.success and r.errors["opening"] > 0.7, r.row()
    b = run_skill(ctx, "retreat", distance=0.12)
    assert b.success and not b.stalled, b.row()
    assert b.errors["contact_force"] < 1.0
    assert env.handle_contact_fingers() == 0


@pytest.mark.parametrize("body", GRIPPERS)
def test_insert_peg_into_slot_and_stall_on_block(body):
    """insert: a peg taken from a holder goes into a slot; pushed onto a solid block, it stalls. Tested on
    the grippers: the dexterous hands' synergies (tuned for fixed bars) do not yet hold a loose peg
    steadily enough to thread a 9 mm clearance."""
    env = env_for(body, "peg")
    for target, should_go_in in ((SLOT, True), (BLOCK, False)):
        env.reset(seed=0)
        hand_peg(env)
        ctx = SkillContext(env)
        g = run_skill(ctx, "grasp", PEG)
        let_go_holder(env)
        assert g.success and ctx.held is PEG, g.row()
        r = run_skill(ctx, "insert", target, depth=0.03)
        if should_go_in:
            assert r.success and not r.stalled, r.row()
            assert r.errors["depth"] > 0.025 and r.errors["lateral_err"] < 0.01
        else:
            assert r.stalled and not r.success, r.row()
            assert r.errors["depth"] < 0.005


def test_insert_without_held_object_fails_cleanly():
    body = next((n for n in ("panda_2f", "shadow", "robotiq_2f85") if SPECS[n].available()), None)
    if body is None:
        pytest.skip("no embodiment assets")
    env, ctx, _ = start(body, "door")
    r = run_skill(ctx, "insert", Target("x", pos=(0.3, 0.0, 0.7)))
    assert not r.success and r.reason == "nothing_held" and r.steps == 0


# ---------------------------------------------------------------------------------------------- registry
def test_registry_enumerates_skill_target_pairs():
    handle = Target("handle", site="h")
    key = Target("key", site="k", kind="object")
    opts = options([handle, key])
    labels = {o.label for o in opts}
    assert "insert" not in {o.skill for o in opts}          # nothing held
    assert "turn(handle, angle=0.6)" in labels and "turn(key, angle=-0.6)" in labels
    assert "push_pull(handle, direction=pull)" in labels and "push_pull(handle, direction=push)" in labels
    assert "release(-)" in labels and "retreat(-)" in labels
    # reach, grasp: 1 variant; turn, push_pull: 2 variants; per target; plus release and retreat
    assert len(opts) == 2 * (1 + 1 + 2 + 2) + 2
    held = options([handle, key], holding=True)
    assert [o.target.name for o in held if o.skill == "insert"] == ["handle"]   # insert into parts only
    assert set(SKILLS) == {"reach", "grasp", "release", "turn", "push_pull", "insert", "retreat"}
    with pytest.raises(ValueError):
        run_skill(None, "reach")


def test_stall_monitor():
    m = StallMonitor(window=10, min_cmd=0.01, ratio=0.25)
    assert not any(m.update(0.002, 0.002) for _ in range(20))      # moving as commanded
    assert any(m.update(0.002, 0.0) for _ in range(10))            # commanded, not moving
    m = StallMonitor(window=10, min_cmd=0.01)
    assert not any(m.update(0.0005, 0.0) for _ in range(20))       # too little commanded to judge
