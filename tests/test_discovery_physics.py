"""PhysicsWorld (pai.discovery.physics): interface shapes, opaque ids, no look that tells which key fits,
the same vocabulary as the MockWorld, hidden keys, probes on locked and unlocked doors, the goal check,
what the hand can do while it holds something."""

import re
import warnings

import pytest

from pai.discovery.interface import MANIPULATE_KINDS, PROBE_KINDS, Entity, Outcome
from pai.discovery.mock import MockWorld, Scenario
from pai.envs.embodiments import SPECS

pytestmark = pytest.mark.skipif(not SPECS["robotiq_2f85"].available(), reason="Menagerie assets missing")
warnings.filterwarnings("ignore", message="Attach conflict")
SEMANTIC = re.compile(r"key|lock|knob|handle|door|drawer|thumb|bolt|latch|lever|cylinder|plug|cab", re.I)
_WORLDS: dict = {}


def world(type_name: str = "hd_knob_pull_left", **overrides):
    from pai.discovery.physics import PhysicsWorld
    key = (type_name, tuple(sorted(overrides.items())))
    if key not in _WORLDS:
        _WORLDS[key] = PhysicsWorld(type_name, seed=0, **overrides)
    return _WORLDS[key]


def ids_of(w) -> dict:
    """internal name -> opaque id (from the ground truth; tests only)."""
    return {v: k for k, v in w.truth()["ids"].items()}


def matching(w) -> str:
    return ids_of(w)[next(n for n, k in w.truth()["keys"].items() if k["matches"])]


def goal_holds(w) -> bool:
    f = w.features()
    return all(f.get(k) == v for k, v in w.goal().items())


def _strings(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def test_entities_features_actions_have_the_interface_shapes():
    w = world(n_keys=3)
    w.reset(0, lock_state="both", key_place="table")
    ents = w.entities()
    assert all(isinstance(e, Entity) for e in ents)
    parts = [e for e in ents if e.kind == "part"]
    objs = [e for e in ents if e.kind == "object"]
    assert len(parts) == 5 and objs                       # door, handle, thumb-turn, cylinder, drawer
    assert all(re.fullmatch(r"p\d+", e.id) for e in parts) and all(re.fullmatch(r"o\d+", e.id) for e in objs)
    assert {e.joint for e in parts} <= {"hinge", "slide", "none"} and all(e.joint == "none" for e in objs)
    assert all(set(e.attrs) == {"shape", "size", "colour"} and len(e.pos) == 3 for e in ents)
    assert all(isinstance(e.attrs["colour"], tuple) and len(e.attrs["colour"]) == 3 for e in ents)
    f = w.features()
    ids = {e.id for e in ents}
    assert f["body:hand"] == "empty"
    for k, v in f.items():
        if k == "body:hand":
            continue
        m = re.fullmatch(r"(part|obj):([po]\d+)\.(open|angle|where)", k)
        assert m and m.group(2) in ids, k
        allowed = {"open": {"closed", "ajar", "open"}, "angle": {"rest", "pos", "neg"}, "where": {"free", "held"}}
        assert v in allowed[m.group(3)] or (m.group(3) == "where" and v.startswith("in:p")), (k, v)
    assert f[next(iter(w.goal()))] == "closed" and not w.goal_reached()
    acts = w.actions()
    assert acts and all(a[0] in PROBE_KINDS + MANIPULATE_KINDS and a[1] in ids for a in acts)
    graspable = {e.id for e in parts if e.graspable}
    assert {a[1] for a in acts if a[0] in PROBE_KINDS and a[0] != "push"} == graspable
    assert {a[1] for a in acts if a[0] == "push"} == {e.id for e in parts}          # non-graspable: push only
    assert {a[1] for a in acts if a[0] == "pick"} == {e.id for e in objs}
    assert not any(a[0] in ("insert", "release", "turn_held") for a in acts)       # nothing held, nothing in


def test_ids_and_attributes_carry_no_semantics_and_are_shuffled_by_seed():
    w = world(n_keys=3)
    maps = []
    for seed in range(6):
        w.reset(seed, lock_state="key", key_place="table")
        shown = [(e.id, e.attrs) for e in w.entities()] + [list(w.features().items()), w.actions(), w.goal()]
        assert not [s for s in _strings(shown) if SEMANTIC.search(s)]
        maps.append(ids_of(w))
    assert len({tuple(sorted(m.items())) for m in maps}) > 1        # the door is not always p1
    w.reset(3, lock_state="key", key_place="table")
    assert ids_of(w) == maps[3]                                     # stable per seed


def test_the_look_of_a_key_does_not_tell_which_one_fits():
    """The fixture and each key body's bitting live as long as the world: the look must not. Across
    resets the fitting key's colour changes every time, and no colour is always the fitting one's."""
    for type_name in ("hd_knob_pull_left", "hd_ball_pull_left"):
        w = world(type_name, n_keys=3)
        fit, other = [], []
        for seed in range(12):
            w.reset(seed, lock_state="key", key_place="table")
            good = matching(w)
            for e in w.entities():
                if e.kind == "object":
                    (fit if e.id == good else other).append(e.attrs["colour"])
            assert len({e.attrs["shape"] for e in w.entities() if e.kind == "object"}) == 1
            assert len({e.attrs["size"] for e in w.entities() if e.kind == "object"}) == 1
        assert len(set(fit)) == len(fit) == 12                     # a new colour every episode
        assert not set(fit) & set(other)
        w.reset(4, lock_state="key", key_place="table")
        c = {e.id: e.attrs["colour"] for e in w.entities() if e.kind == "object"}
        w.reset(4, lock_state="key", key_place="table")
        assert c == {e.id: e.attrs["colour"] for e in w.entities() if e.kind == "object"}   # stable per seed


def test_the_same_vocabulary_as_the_mock_world():
    """Joint, graspable and shape of each part, the objects' shape, the colour format and the feature
    names agree with the MockWorld for the same door type, and every action the mock offers on the
    parts they share is offered here too."""
    for type_name in ("hd_knob_pull_left", "hd_ball_pull_left", "hd_lever_pull_right", "hd_pushbar_push_left"):
        w = world(type_name, n_keys=2)
        w.reset(0, lock_state="unlocked", key_place="table")
        mw = MockWorld(Scenario(type_name, "latch", "table", n_keys=2, seed=0, p_fail=0.0))
        pe = {e.id: e for e in w.entities()}
        me = {e.id: e for e in mw.entities()}
        pid, mid = ids_of(w), mw.pid
        shared = set(pid) & set(mid)
        assert {"door", "handle", "drawer"} <= shared
        for n in shared:
            a, b = pe[pid[n]], me[mid[n]]
            assert (a.joint, a.graspable, a.attrs["shape"]) == (b.joint, b.graspable, b.attrs["shape"]), (type_name, n)
        for d in (pe, me):
            assert {e.attrs["shape"] for e in d.values() if e.kind == "object"} == {"elongated"}
            assert all(isinstance(e.attrs["colour"], tuple) and len(e.attrs["colour"]) == 3 for e in d.values())
        kind = lambda f: re.sub(r"[po]\d+", "#", f)     # noqa: E731
        assert {kind(k) for k in w.features()} == {kind(k) for k in mw.features()}
        to_phys = {mid[n]: pid[n] for n in shared}
        mock_acts = {(a[0], to_phys[a[1]], *a[2:]) for a in mw.actions() if a[1] in to_phys}
        assert mock_acts <= set(w.actions()), mock_acts - set(w.actions())


def test_a_key_in_the_drawer_appears_only_once_the_drawer_is_open():
    w = world()
    w.reset(0, lock_state="key", key_place="drawer")
    ids = ids_of(w)
    good = matching(w)
    assert good not in {e.id for e in w.entities()} and not any(k.startswith(f"obj:{good}.") for k in w.features())
    o = w.execute(("pull", ids["drawer"]))
    assert isinstance(o, Outcome) and o.executed and not o.stalled
    assert good in {e.id for e in o.revealed}
    assert o.changed[f"part:{ids['drawer']}.open"] == ("closed", "open")
    assert w.features()[f"obj:{good}.where"] == "free" and ("pick", good) in w.actions()
    assert o.cost > 0


def test_a_probe_on_a_locked_door_stalls():
    w = world()
    w.reset(0, lock_state="deadbolt")
    ids = ids_of(w)
    assert w.features()[f"part:{ids['thumb']}.angle"] == "pos"     # the thrown thumb-turn, as it looks
    o = w.execute(("turn_pull", ids["handle"], 1))
    assert o.executed and o.stalled and not w.goal_reached()
    assert f"part:{ids['door']}.open" not in o.changed
    o = w.execute(("push", ids["cylinder"]))                       # a flat face: pressed, not grasped
    assert o.executed and o.stalled and not w.goal_reached()


def test_the_unlocked_door_opens_with_the_right_probe_only_and_the_goal_follows_the_features():
    w = world()
    w.reset(0, lock_state="unlocked")
    ids = ids_of(w)
    o = w.execute(("pull", ids["handle"]))            # the latch holds while the knob is not turned
    assert o.executed and o.stalled and not w.goal_reached() and not w.truth()["opened"]
    o = w.execute(("turn_pull", ids["handle"], 1))
    assert o.executed and not o.stalled and w.goal_reached() and goal_holds(w)
    assert o.changed[f"part:{ids['door']}.open"] == ("closed", "open")
    assert {"latch_released", "door_opened"} <= set(w.truth()["events"])
    w.execute(("pull", ids["drawer"]))                # the hand lets go of the door first: it may swing back
    assert w.goal_reached() == goal_holds(w) and w.truth()["opened"]


def test_holding_an_object_changes_what_can_be_done():
    w = world(n_keys=3)
    w.reset(0, lock_state="key", key_place="table")
    ids = ids_of(w)
    good = matching(w)
    o = w.execute(("pick", good))
    assert o.executed and o.changed[f"obj:{good}.where"] == ("free", "held")
    assert w.features()["body:hand"] == good
    acts = w.actions()
    assert ("insert", good, ids["cylinder"]) in acts and ("release", good) in acts and ("turn_held", good, 1) in acts
    assert not any(a[0] in PROBE_KINDS for a in acts)                    # one hand
    o = w.execute(("turn_held", good, -1))                               # in the hand: nothing happens
    assert o.executed and not o.stalled and not o.changed and w.features()[f"obj:{good}.where"] == "held"
    o = w.execute(("insert", good, ids["cylinder"]))
    assert w.features()[f"obj:{good}.where"] == f"in:{ids['cylinder']}" and ("turn_held", good, 1) in w.actions()
