"""Home doors: registry, latch, geometric deadbolts, key capture / bitting, hidden placement, events, oracle."""

import warnings

import mujoco
import numpy as np
import pytest

from pai.envs.embodiments import SPECS
from pai.envs.home_doors import (HOME_DOOR_TYPES, THUMB_RANGE, home_door_fixture, home_door_types, make_home_door,
                                 sample_home_door)

pytestmark = pytest.mark.skipif(not SPECS["robotiq_2f85"].available(), reason="Menagerie assets missing")
warnings.filterwarnings("ignore", message="Attach conflict")
HOLD = np.r_[np.zeros(6), 1.0]
_ENVS: dict = {}


def home_env(type_name: str, **overrides):
    from pai.envs.door_scene import DoorSceneEnv
    key = (type_name, tuple(sorted(overrides.items())))
    if key not in _ENVS:
        _ENVS[key] = DoorSceneEnv("robotiq_2f85", home_door_fixture(type_name, **overrides), seed=0)
    return _ENVS[key]


def run(env, n, door=0.0, handle=0.0, thumb=None):
    rt, d = env.runtime, env.data
    for _ in range(n):
        d.qfrc_applied[:] = 0
        d.qfrc_applied[rt.door["dof"]] = door
        d.qfrc_applied[env.model.jnt_dofadr[rt.handle_jnt]] = handle
        if thumb is not None:
            d.qfrc_applied[env.model.jnt_dofadr[rt.thumb["jnt"]]] = thumb
        obs = env.step(HOLD)
    d.qfrc_applied[:] = 0
    return obs


def _keys(x) -> set:
    if isinstance(x, dict):
        return set(x) | set().union(*(_keys(v) for v in x.values()))
    return set()


def test_registry_splits_and_sampling():
    train, test = set(home_door_types("train")), set(home_door_types("test"))
    assert train and test and not train & test and train | test == set(HOME_DOOR_TYPES)
    rng = np.random.default_rng(0)
    assert all(sample_home_door(rng, "test").type in test for _ in range(10))
    fx = make_home_door(np.random.default_rng(1), "hd_knob_pull_left")
    assert fx.params["bitting"] in fx.params["key_bittings"] and len(set(fx.params["key_bittings"])) == fx.params["n_keys"]
    handles = {t.fixed["handle"] for t in HOME_DOOR_TYPES.values()}
    assert {"knob_round", "knob_ball", "lever", "push_bar"} <= handles
    assert {t.fixed["swing"] for t in HOME_DOOR_TYPES.values()} == {"push", "pull"}


@pytest.mark.parametrize("type_name", list(HOME_DOOR_TYPES))
def test_every_type_builds_rests_closed_and_hides_its_state(type_name):
    env = home_env(type_name)
    obs = env.reset(seed=0)
    obs = run(env, 50)
    assert env.stable()
    assert obs["articulations"]["door"]["opening"] < 0.01
    hidden = {"bitting", "lock_state", "locked", "bolt", "latch", "key_place", "matches", "cause", "carrier"}
    assert not hidden & _keys({k: v for k, v in obs.items() if k in ("articulations", "keys", "fixture")})
    assert env.info["lock_state"] in env.runtime.available_states()


def test_reset_is_deterministic_per_seed():
    env = home_env("hd_knob_pull_left")
    a = env.reset(seed=3) and env.info
    b = env.reset(seed=3) and env.info
    assert a == b
    infos = [env.reset(seed=s) and (env.info["lock_state"], env.info["key_place"]) for s in range(12)]
    assert len(set(infos)) > 3


@pytest.mark.parametrize("type_name,torque", [("hd_knob_pull_left", 1.0), ("hd_lever_pull_right", 1.5),
                                              ("hd_pushbar_push_left", 20.0), ("hd_knob_push_right", 1.0)])
def test_latch_holds_until_the_handle_is_turned(type_name, torque):
    env = home_env(type_name)
    env.reset(seed=0, lock_state="unlocked")
    assert run(env, 60, door=25.0)["articulations"]["door"]["q"] < 0.01
    env.reset(seed=0, lock_state="unlocked")
    assert run(env, 100, door=25.0, handle=torque)["articulations"]["door"]["q"] > 1.0
    kinds = [e["type"] for e in env.runtime.events]
    assert {"knob_turned", "latch_released", "door_opened"} <= set(kinds)


def test_self_closing_door_relatches():
    env = home_env("hd_lever_pull_right")               # strong closer
    env.reset(seed=0, lock_state="unlocked")
    run(env, 60, door=25.0, handle=1.5)
    obs = run(env, 250)
    assert obs["articulations"]["door"]["q"] < 0.02                        # swung shut
    assert run(env, 60, door=25.0)["articulations"]["door"]["q"] < 0.03    # and latched again


def test_thumb_deadbolt_blocks_by_geometry_and_is_visible():
    env = home_env("hd_knob_pull_left")
    obs = env.reset(seed=0, lock_state="deadbolt")
    assert obs["articulations"]["thumb_turn"]["q"] == pytest.approx(THUMB_RANGE, abs=0.02)   # visible
    assert env.runtime.bolt_thrown("thumb")
    q = run(env, 100, door=25.0, handle=1.0)["articulations"]["door"]["q"]
    assert 0.0 < q < 0.03                   # the bolt stops against the jamb pocket: it rattles, it does not open
    d, m = env.data, env.model
    bolt = m.body("home_thumb_bolt").id
    pocket = [g for g in range(m.ngeom) if m.geom(g).name.startswith("pocket")]
    assert any(d.contact[i].geom1 in pocket or d.contact[i].geom2 in pocket for i in range(d.ncon)
               if bolt in (m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]))
    env.reset(seed=0, lock_state="deadbolt")
    run(env, 60, thumb=-0.5)                # turn the thumb back
    assert not env.runtime.bolt_thrown("thumb")
    assert run(env, 100, door=25.0, handle=1.0)["articulations"]["door"]["q"] > 1.0
    assert "bolt_retracted" in [e["type"] for e in env.runtime.events]


def _insert(env, ki: int, twist: float, steps=80):
    """Put key ki at the keyhole mouth, aligned, push it in, then twist it (applied force/torque)."""
    rt, m, d = env.runtime, env.model, env.data
    k = rt.keys[ki]
    pb = rt.kl["plug_body"]
    R = d.xmat[pb].reshape(3, 3).copy()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    d.qpos[k["qpos"]:k["qpos"] + 3] = d.xpos[pb] - 0.004 * R[:, 0]
    d.qpos[k["qpos"] + 3:k["qpos"] + 7] = q
    d.qvel[k["qvel"]:k["qvel"] + 6] = 0
    mujoco.mj_forward(m, d)
    b = k["body"]
    lift = 9.81 * m.body_subtreemass[b]
    for i in range(40 + steps):
        d.xfrc_applied[b] = 0
        d.xfrc_applied[b, :3] = 3.0 * R[:, 0] + [0, 0, lift]
        if i >= 40:
            d.xfrc_applied[b, 3:] = twist * R[:, 0]
        obs = env.step(HOLD)
    d.xfrc_applied[:] = 0
    return obs


def test_only_the_matching_key_turns_the_lock():
    env = home_env("hd_knob_pull_left", n_keys=2)
    rt = env.runtime
    good = next(i for i, k in enumerate(rt.keys) if k["bitting"] == rt.rec.key_lock["bitting"])
    bad = 1 - good
    env.reset(seed=0, lock_state="key", key_place="table")
    obs = _insert(env, bad, 0.6)
    assert rt.inserted and abs(obs["articulations"]["key_lock"]["q"]) < 0.05 and rt.bolt_thrown("key")
    assert run(env, 60, door=25.0, handle=1.0)["articulations"]["door"]["q"] < 0.03
    ev = [e for e in rt.events if e["type"] == "key_inserted"]
    assert ev and ev[0]["matches"] is False
    env.reset(seed=0, lock_state="key", key_place="table")
    obs = _insert(env, good, 0.6)
    assert obs["articulations"]["key_lock"]["q"] > 1.3 and not rt.bolt_thrown("key")
    assert {"key_inserted", "bolt_retracted", "key_turned"} <= {e["type"] for e in rt.events}
    assert run(env, 100, door=25.0, handle=1.0)["articulations"]["door"]["q"] > 1.0


def test_key_placement_is_hidden_in_the_drawer_and_unsolvable_in_the_other_room():
    env = home_env("hd_knob_pull_left")
    obs = env.reset(seed=0, lock_state="key", key_place="drawer")
    name = next(n for n, k in env.info["keys"].items() if k["matches"])
    assert name not in obs["keys"]                                   # inside the closed drawer: not visible
    d = env.data
    d.qpos[env.runtime.drawer["q"]] = 0.24
    mujoco.mj_forward(env.model, d)
    assert name in env.step(HOLD)["keys"]
    env.reset(seed=0, lock_state="key", key_place="other_room")
    assert env.info["solvable"] is False
    env.reset(seed=0, lock_state="deadbolt", key_place="other_room")
    assert env.info["solvable"] is True                               # the thumb-turn needs no key


def test_oracle_finds_the_key_in_the_drawer_and_opens_the_door():
    from pai.envs.home_oracle import run_oracle
    env = home_env("hd_knob_push_right")
    r = run_oracle(env, seed=0, lock_state="key", key_place="drawer")
    assert r["opened"], r["why"]
    assert r["events"].index("drawer_opened") < r["events"].index("key_inserted") < r["events"].index("key_turned") \
        < r["events"].index("door_opened")
    r = run_oracle(env, seed=0, lock_state="key", key_place="other_room")
    assert not r["opened"] and "other room" in r["why"]


# ---------------------------------------------------------------------------------------- decoys
def _decoy_names(m) -> set:
    return {m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith("dec_")} | \
        {m.joint(j).name for j in range(m.njnt) if m.joint(j).name.startswith("dec_")}


def test_decoys_are_off_by_default_and_every_type_builds_stable_with_them():
    from pai.envs.home_doors import DECOYS
    assert not _decoy_names(home_env("hd_knob_pull_left").model)
    for type_name in HOME_DOOR_TYPES:
        env = home_env(type_name, decoys=True)
        rec = env.runtime.rec
        assert set(rec.decoys) == set(DECOYS) - {"peg"} and [pr["body"] for pr in rec.props] == ["dec_peg"]
        env.reset(seed=0)
        obs = run(env, 60)
        assert env.stable() and obs["articulations"]["door"]["opening"] < 0.01
        peg = env.runtime.props[0]
        bid, pos, _ = peg["rest"]
        rest = env.data.xpos[bid] + env.data.xmat[bid].reshape(3, 3) @ pos
        assert np.linalg.norm(env.data.xpos[peg["body"]] - rest) < 0.01        # it lies still on its posts
    env = home_env("hd_knob_pull_left", decoys=("cab_disc",))
    assert set(env.runtime.rec.decoys) == {"cab_disc"} and not env.runtime.props
    with pytest.raises(ValueError):
        home_env("hd_knob_pull_left", decoys=("trapdoor",))


def test_decoys_are_connected_to_nothing():
    """No equality refers to a decoy, and turning every decoy joint to its stop (torques on the joints)
    leaves bolts, latch and lock as they were: the door stays shut under a pull and nothing is logged."""
    env = home_env("hd_knob_pull_left", decoys=True)
    m, d, rt = env.model, env.data, env.runtime
    dec_j = [j for j in range(m.njnt)
             if m.joint(j).name.startswith("dec_") and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    dec_b = {m.body(b).id for b in range(m.nbody) if m.body(b).name.startswith("dec_")}
    assert len(dec_j) == 4 and len(dec_b) == 5
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT:
            assert m.eq_obj1id[e] not in dec_j and m.eq_obj2id[e] not in dec_j
        elif m.eq_objtype[e] == mujoco.mjtObj.mjOBJ_BODY:
            assert m.eq_obj1id[e] not in dec_b and m.eq_obj2id[e] not in dec_b
    for state in ("both", "unlocked"):
        for sign in (1.0, -1.0):
            env.reset(seed=0, lock_state=state, key_place="table")
            bolts = {k: rt.bolt_thrown(k) for k in rt.bolt_q}
            q0 = d.qpos[[m.jnt_qposadr[j] for j in dec_j]].copy()
            for _ in range(80):
                d.qfrc_applied[:] = 0
                d.qfrc_applied[[m.jnt_dofadr[j] for j in dec_j]] = sign * 0.5
                d.qfrc_applied[rt.door["dof"]] = 25.0
                env.step(HOLD)
            d.qfrc_applied[:] = 0
            moved = np.abs(d.qpos[[m.jnt_qposadr[j] for j in dec_j]] - q0)
            assert np.all(moved > 0.5), moved                                  # they do turn
            assert env.stable() and d.qpos[rt.door["q"]] < 0.03                 # the latch (and bolts) hold the door
            assert {k: rt.bolt_thrown(k) for k in rt.bolt_q} == bolts
            assert abs(d.qpos[rt.kl["plug_q"]]) < 0.05 and not rt.inserted
            assert [e["type"] for e in rt.events] == ["home_reset"]


def test_the_peg_is_never_taken_for_a_key():
    """Held at the keyhole mouth, aligned, and pushed in, the peg is not captured (only keys are)."""
    env = home_env("hd_knob_pull_left", decoys=True)
    rt, m, d = env.runtime, env.model, env.data
    env.reset(seed=0, lock_state="key", key_place="table")
    peg = rt.props[0]
    pb = rt.kl["plug_body"]
    R = d.xmat[pb].reshape(3, 3).copy()
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    d.qpos[peg["qpos"]:peg["qpos"] + 3] = d.xpos[pb] - 0.03 * R[:, 0]
    d.qpos[peg["qpos"] + 3:peg["qpos"] + 7] = q
    d.qvel[peg["qvel"]:peg["qvel"] + 6] = 0
    mujoco.mj_forward(m, d)
    lift = 9.81 * m.body_subtreemass[peg["body"]]
    for _ in range(80):
        d.xfrc_applied[peg["body"]] = 0
        d.xfrc_applied[peg["body"], :3] = 3.0 * R[:, 0] + [0, 0, lift]
        env.step(HOLD)
    d.xfrc_applied[:] = 0
    assert env.stable() and rt.welded is None and not rt.inserted and rt.bolt_thrown("key")
    assert "key_inserted" not in [e["type"] for e in rt.events]


def test_the_oracle_opens_a_push_door_with_decoys_as_without():
    """The look-alikes stand in nobody's way: the scripted solver opens a push door (its leaf swings over
    the far table) with the key in the drawer, decoys on."""
    from pai.envs.home_oracle import run_oracle
    env = home_env("hd_knob_push_right", decoys=True)
    r = run_oracle(env, seed=0, lock_state="key", key_place="drawer")
    assert r["opened"] and not r["why"], r["why"]
    assert r["events"].index("drawer_opened") < r["events"].index("key_inserted") < r["events"].index("door_opened")
