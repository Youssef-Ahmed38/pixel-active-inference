"""Discovery (mock world only, fast): the symbolic home door, the causal-rule belief, the agent against
random and novelty baselines, no-solution detection, explanations against the ground truth, recipe reuse."""

import dataclasses
import re

import numpy as np
import pytest

from pai.discovery.agent import AGENTS, DiscoveryAgent, Locality, _Episode, compare, format_table
from pai.discovery.belief import ActionModel, Belief, context, score_actions
from pai.discovery.explain import explain, score
from pai.discovery.mock import LOCK_STATES, MockWorld, Scenario, available_states, sample_scenario, scenario_types
from pai.discovery.recipes import RecipeBook

SEMANTIC = re.compile(r"door|knob|lever|key|lock|thumb|bolt|latch|drawer|cylinder|handle|hook|cabinet|pen\b|dial")
_LOGS: list = []      # (log, truth) of the episodes run below, reused by the explanation test


def world(type_name="hd_knob_pull_left", lock="both", place="drawer", n_keys=2, seed=3, **kw) -> MockWorld:
    return MockWorld(Scenario(type_name, lock, place, n_keys=n_keys, seed=seed, p_fail=0.0, **kw))


def run(w, a):
    assert a in w.actions(), (a, w.actions())
    return w.execute(a)


# ---------------------------------------------------------------------------------------- the mock world
def test_mock_is_opaque_and_shuffled():
    ids = set()
    for seed in range(6):
        w = MockWorld(sample_scenario(np.random.default_rng(seed)))
        for e in w.entities():
            assert re.fullmatch(r"[po]\d+", e.id)
            assert set(e.attrs) <= {"shape", "size", "colour"} and not SEMANTIC.search(str(e.attrs))
        assert not any(SEMANTIC.search(f + v) for f, v in w.features().items())
        assert not any(SEMANTIC.search(str(a)) for a in w.actions())
        ids.add(w.truth()["parts"]["door"])
    assert len(ids) > 1                     # the door is a different id in different scenes


def test_mock_scenarios_follow_the_home_door_registry():
    assert set(scenario_types("train")) and set(scenario_types("test"))
    assert not set(scenario_types("train")) & set(scenario_types("test"))
    rng = np.random.default_rng(0)
    for st in LOCK_STATES:
        sc = sample_scenario(rng, "test", lock_state=st)
        assert st in available_states(sc.type_name) and sc.type_name in scenario_types("test")
    assert "deadbolt" not in available_states("hd_knob_push_right")      # no thumb-turn on that type
    assert not sample_scenario(rng, lock_state="key", key_place="other_room").solvable


def test_mock_latch_deadbolt_key_mechanics():
    w = world()
    t = w.truth()
    P, dr = t["parts"], t["unlock_sign"]
    knob, thumb, cyl, drawer = P["handle"], P["thumb"], P["cylinder"], P["drawer"]
    key = t["matching_key"]
    wrong = next(o for o, k in t["keys"].items() if not k["matches"])
    assert key not in {e.id for e in w.entities()}                           # hidden in the drawer
    assert run(w, ("pull", knob)).stalled                                    # latched (and bolted)
    assert run(w, ("turn", knob, 1)).changed == {}                           # the knob springs back
    assert run(w, ("turn_pull", knob, 1)).stalled                            # bolted
    out = run(w, ("turn", thumb, -1))
    assert out.changed == {f"part:{thumb}.angle": ("pos", "rest")}
    assert run(w, ("turn_pull", knob, 1)).stalled                            # still key-locked
    out = run(w, ("pull", drawer))
    assert key in [e.id for e in out.revealed] and not out.stalled
    run(w, ("pick", wrong))
    assert run(w, ("insert", wrong, knob)).stalled                           # not a keyhole
    assert not run(w, ("insert", wrong, cyl)).stalled
    assert run(w, ("turn_held", wrong, dr)).stalled                          # wrong key: will not turn
    run(w, ("release", wrong))
    assert w.features()[f"obj:{wrong}.where"] == f"in:{cyl}"                 # stays in when let go
    run(w, ("pick", wrong))
    assert w.features()[f"obj:{wrong}.where"] == "held"                      # pulled back out at home
    run(w, ("release", wrong))
    run(w, ("pick", key))
    run(w, ("insert", key, cyl))
    assert run(w, ("turn_held", key, -dr)).stalled                           # already thrown that way
    out = run(w, ("turn_held", key, dr))
    assert not out.stalled and out.changed[f"part:{cyl}.angle"][1] != "rest"
    run(w, ("release", key))
    out = run(w, ("pick", key))                                              # a turned key stays in
    assert w.features()[f"obj:{key}.where"] == f"in:{cyl}" and out.notes == "grasped in place"
    run(w, ("release", key))
    assert run(w, ("pull", knob)).stalled                                    # the latch still holds
    out = run(w, ("turn_pull", knob, -1))                                    # two-way knob
    assert not out.stalled and w.goal_reached()
    kinds = [e["type"] for e in w.events]
    assert kinds.index("bolt_retracted") < kinds.index("drawer_opened") < kinds.index("key_turned") < kinds.index("door_opened")


def test_mock_push_door_one_way_knob_and_noise():
    w = world("hd_ball_push_right", "latch", "table", n_keys=1)
    knob = w.truth()["parts"]["handle"]
    assert run(w, ("turn_pull", knob, 1)).stalled                            # wrong way to swing
    assert run(w, ("turn_push", knob, -1)).stalled                           # one-way knob
    assert not run(w, ("turn_push", knob, 1)).stalled and w.goal_reached()
    w = MockWorld(Scenario("hd_knob_pull_left", "latch", "table", seed=1, p_fail=0.5))
    outs = [w.execute(("turn", w.truth()["parts"]["handle"], 1)) for _ in range(40)]
    assert 5 < sum(not o.executed for o in outs) < 35


def test_mock_door_mounted_distractors_carry_the_door():
    w = world("hd_knob_pull_left", "unlocked", "table", n_keys=1, door_dial=True, door_hook=True)
    P = w.truth()["parts"]
    dd = P["door_dial"]
    assert run(w, ("turn", dd, 1)).changed == {f"part:{dd}.angle": ("rest", "pos")}   # persistent
    assert not run(w, ("pull", P["door_hook"])).stalled and w.goal_reached()           # on the leaf
    w = world("hd_knob_pull_left", "latch", "table", n_keys=1, door_dial=True)
    assert run(w, ("turn_pull", w.truth()["parts"]["door_dial"], 1)).stalled           # does not release the latch


def test_mock_no_solution_key_behind_the_door():
    w = world(lock="key", place="other_room", n_keys=2)
    t = w.truth()
    assert not t["solvable"] and t["matching_key"] not in {e.id for e in w.entities()}


# ---------------------------------------------------------------------------------------- belief and model
def test_belief_rules_out_and_grows_its_vocabulary():
    goal = {"part:p1.open": "open"}
    probes = [("pull", "p2"), ("turn_pull", "p2", 1)]
    s0 = {"part:p1.open": "closed", "part:p2.angle": "rest", "part:p3.angle": "pos"}
    b = Belief(goal, probes)
    b.update_vocab(s0, ["p1", "p2", "p3"])
    assert ("part:p1.open", "open") not in b.lits and ("part:p3.angle", "rest") in b.lits
    e0 = b.eig_probes(s0)
    assert (e0 > 0).all()
    b.observe(("pull", "p2"), s0, False)
    b.observe(("turn_pull", "p2", 1), s0, False)
    s1 = dict(s0, **{"part:p3.angle": "rest"})
    assert b.eig_probes(s1).max() > b.eig_probes(s0).max()       # a change makes probes informative again
    n = len(b.lits)
    s2 = dict(s1, **{"obj:o1.where": "free", "obj:o1.angle": "rest"})
    assert b.update_vocab(s2, ["p1", "p2", "p3"]) and len(b.lits) > n
    assert ("obj:o1.where", "in:p3") in b.lits
    b.observe(("turn_pull", "p2", 1), s1, True)
    m = b.map((b.h_probe == b.pidx[("turn_pull", "p2", 1)]) & b.holds(s1) & (b.h_err == 0))
    assert m["literals"] == {"part:p3.angle": "rest"}              # the simplest rule that fits


def test_action_model_contexts_include_what_is_inside():
    m = ActionModel()
    s = {"obj:o1.where": "held", "obj:o2.where": "in:p5", "part:p5.angle": "rest", "body:hand": "o1"}
    a = ("insert", "o1", "p5")
    assert ("obj:o2.where", "in:p5") in context(a, s)                # the occupant is part of the context
    m.observe_available([a], s)
    assert m.untested(a, s) and not m.exhausted()
    m.record(a, s, s, True, True, 0)
    assert not m.untested(a, s) and m.exhausted()
    s2 = dict(s, **{"obj:o2.where": "free"})
    assert m.untested(a, s2)                                          # emptied: worth trying again
    assert m.predict(a, s) == ({}, True)
    b = Belief({"part:p1.open": "open"}, [("pull", "p2")])
    b.update_vocab(s, ["p1", "p2", "p5"])
    rows = score_actions(b, m, s, [a, ("release", "o1")], lambda x: 3.0)
    assert all(r["rate"] >= 0 for r in rows)


# ---------------------------------------------------------------------------------------- the agent
@pytest.mark.parametrize("lock", LOCK_STATES)
def test_agent_opens_each_lock_state(lock):
    rng = np.random.default_rng(100 + LOCK_STATES.index(lock))
    for i, split in enumerate(("train", "test")):
        sc = sample_scenario(rng, split, lock_state=lock, key_place="drawer" if i == 0 else "shelf")
        w = MockWorld(sc)
        log = DiscoveryAgent(seed=i).run(w, 500)
        _LOGS.append((log, w.truth()))
        assert log.reached_goal and log.stopped_reason == "goal", (lock, sc, log.n_actions)
        assert log.n_actions == len(log.actions) == len(log.outcomes) and log.time_cost > 0
        assert log.snapshots and log.final_map is not None


def test_merged_state_forgets_objects_that_went_out_of_view():
    w = world(lock="latch", place="drawer", n_keys=1)
    P, key = w.truth()["parts"], w.truth()["matching_key"]
    ep = _Episode(w)
    ep.step(("pull", P["drawer"]), "test")
    assert f"obj:{key}.where" in ep.state
    ep.step(("push", P["drawer"]), "test")                    # shut over the key: out of view again
    assert not any(f.startswith(f"obj:{key}.") for f in ep.state)


def test_no_false_give_up_while_a_reachable_rule_is_untested():
    # a scene where the old rule (sweep 80 actions, then compare the reachable mass with UNKNOWN) gave up
    sc = Scenario("hd_lever_pull_right", "both", "shelf", n_keys=3, seed=570530607, dial=False, cabinet=True, pen=False)
    log = DiscoveryAgent(seed=3).run(MockWorld(sc), 1000)
    assert log.reached_goal and log.stopped_reason == "goal", (log.n_actions, log.stopped_reason)


@pytest.mark.parametrize("lock", ["deadbolt", "both"])
def test_agent_beats_random_with_distractors_on_the_door(lock):
    """A persistent dial and a hook on the leaf, closer to the door's centre than the handle: nearness to
    the door is no longer a hint of what matters."""
    rng = np.random.default_rng(42)
    scen = []
    while len(scen) < 12:
        sc = sample_scenario(rng, "train", lock_state=lock)
        if sc.solvable:
            scen.append(dataclasses.replace(sc, door_dial=True, door_hook=True))
    res = {}
    for name in ("discovery", "random"):
        logs = [AGENTS[name](seed=i).run(MockWorld(sc), 1000) for i, sc in enumerate(scen)]
        res[name] = (np.mean([lg.reached_goal for lg in logs]),
                     float(np.mean([lg.n_actions if lg.reached_goal else 1000 for lg in logs])))
    print(f"\n{lock}, dial and hook on the door, 12 scenes (success, mean actions):", res)
    assert res["discovery"][0] == 1.0 and res["discovery"][1] < res["random"][1]


def test_agent_beats_random_and_novelty():
    rows = compare(n_seeds=50, budget=1000, seed=0, processes=8, verbose=False)
    print("\n" + format_table(rows, "50 solvable train scenes, any lock state, budget 1000"))
    d, n, r = rows["discovery"], rows["novelty"], rows["random"]
    assert d["success"] >= 0.96
    assert d["actions"] < 0.5 * r["actions"] and d["actions"] < n["actions"]
    assert d["time"] < r["time"]


@pytest.mark.parametrize("lock", ["key", "both"])
def test_agent_detects_no_solution(lock):
    rng = np.random.default_rng(7)
    sc = sample_scenario(rng, "train", lock_state=lock, key_place="other_room")
    w = MockWorld(sc)
    log = DiscoveryAgent(seed=0).run(w, 800)
    _LOGS.append((log, w.truth()))
    assert log.stopped_reason == "no_solution" and not log.reached_goal, (log.n_actions, log.stopped_reason)
    e = explain(log)
    assert e.claims["no_solution"] and "could not open" in e.text
    print(f"\n{lock}: gave up after {log.n_actions} actions: {e.text}")


def test_explanation_and_score_of_a_false_give_up():
    """The no-solution text names the object that did turn the lock, and the score does not reward it."""
    w = world(lock="key", place="table", n_keys=2)
    t = w.truth()
    P, key, dr = t["parts"], t["matching_key"], t["unlock_sign"]
    wrong = next(o for o, k in t["keys"].items() if not k["matches"])
    ep = _Episode(w)
    for a in [("pick", wrong), ("insert", wrong, P["cylinder"]), ("turn_held", wrong, dr), ("release", wrong), ("pick", wrong),
              ("release", wrong), ("pick", key), ("insert", key, P["cylinder"]), ("turn_held", key, dr),
              ("turn_held", key, -dr), ("release", key)]:
        ep.step(a, "test")
    log = ep.finish("no_solution")
    e = explain(log)
    assert e.claims["turned"] == [key] and e.claims["wrong_keys"] == [wrong]
    assert f"{key} went into" in e.text and "turned it" in e.text
    s = score(e.claims, t, log)
    assert not s["opened"] and not s["no_solution"] and not s["latched"] and not s["key_used"]
    assert s["key_part"] and s["wrong_keys"]                                  # it did find the lock
    assert s["relevant"] < 0.5 and s["accuracy"] <= 0.5


def test_explanations_match_the_ground_truth():
    assert _LOGS, "run with the agent tests"
    accs, rel, false = [], [], 0
    for log, truth in _LOGS:
        e = explain(log)
        s = score(e.claims, truth, log)
        accs.append(s["accuracy"])
        rel.append(s["relevant"])
        false += s["false_claims"]
        assert e.text
    print(f"\nexplanation accuracy over {len(accs)} episodes: {np.mean(accs):.3f} (on the mechanisms present: "
          f"{np.mean(rel):.3f}; false claims: {false})")
    print(explain(_LOGS[-5][0]).text)
    assert np.mean(accs) >= 0.9 and np.mean(rel) >= 0.85


def test_explanation_of_a_locked_door_names_key_and_part():
    w = world(lock="key", place="drawer", n_keys=2, seed=11)
    log = DiscoveryAgent(seed=3).run(w, 500)
    t = w.truth()
    c = explain(log).claims
    assert log.reached_goal
    assert c["key_used"] == t["matching_key"] and c["key_part"] == t["parts"]["cylinder"]
    assert c["key_found_in"] == t["parts"]["drawer"] and c["latched"]
    assert all(not t["keys"][k]["matches"] for k in c["wrong_keys"])


def test_recipe_reuse_reduces_actions_on_a_new_scene():
    res = {}
    for lock in ("latch", "deadbolt"):
        cold, warm = [], []
        for i in range(8):
            rng = np.random.default_rng(3000 + i)
            first = sample_scenario(rng, "train", lock_state=lock, key_place="table")
            second = sample_scenario(rng, "test", lock_state=lock, key_place="table")
            book = RecipeBook()
            r = book.learn(DiscoveryAgent(seed=i).run(MockWorld(first), 400))
            assert r is not None and r.released                              # it learned the latch
            cold.append(DiscoveryAgent(seed=i).run(MockWorld(second), 400).n_actions)
            warm.append(DiscoveryAgent(seed=i).run(MockWorld(second), 400, recipes=book).n_actions)
        res[lock] = (float(np.mean(cold)), float(np.mean(warm)))
    print("\nactions on a held-out door type, without / with the recipe of a first scene:",
          {k: f"{a:.1f} / {b:.1f}" for k, (a, b) in res.items()})
    assert all(b < a for a, b in res.values())


# ---------------------------------------------------------------------------------------- locality ablation
def test_locality_switches_and_defaults():
    assert DiscoveryAgent().locality == Locality() and DiscoveryAgent(radius=0.5).locality.radius == 0.5
    off = Locality.off()
    assert (off.radius, off.probe_prior, off.lit_prior, off.explain) == (None, 0.0, 0.0, "evidence")
    assert Locality().without("radius") == Locality(radius=None)
    w = world(lock="latch", place="table", n_keys=1)
    drawer = w.truth()["parts"]["drawer"]
    ag = DiscoveryAgent(seed=0)
    ag.run(w, 1)
    assert not any(p[1] == drawer for p in ag.belief.probes)                 # far from the door: not a probe
    w = world(lock="latch", place="table", n_keys=1)
    ag = DiscoveryAgent(seed=0, locality=Locality.off())
    log = ag.run(w, 1)
    assert ("pull", drawer) in ag.belief.probes and log.locality["explain"] == "evidence"
    lb = ag.belief.log_prior
    assert np.ptp(lb[ag.belief.h_size == 0]) == 0                           # no distance penalty on probes


@pytest.mark.parametrize("lock", ["latch", "deadbolt", "key"])
def test_agent_opens_doors_without_locality(lock):
    rng = np.random.default_rng(500 + LOCK_STATES.index(lock))
    sc = sample_scenario(rng, "train", lock_state=lock, key_place="drawer", shared_shapes=True)
    w = MockWorld(sc)
    log = DiscoveryAgent(seed=1, locality=Locality.off()).run(w, 1000)
    assert log.reached_goal, (lock, sc, log.n_actions)
    s = score(explain(log).claims, w.truth(), log)
    print(f"\n{lock}, no locality, shared shapes: {log.n_actions} actions, explanation {s['accuracy']:.2f}")


def _hand_log(att_dial: float):
    """deadbolt: probe, turn the cabinet dial, probe, turn the thumb back, probe (opens); the belief's rule
    is the thumb and it gives `att_dial` of its mass to rules that also need the dial."""
    w = world("hd_knob_pull_left", "deadbolt", "table", n_keys=1, dial=True)
    P = w.truth()["parts"]
    probe = ("turn_pull", P["handle"], 1)
    ep = _Episode(w)
    for a in (probe, ("turn", P["dial"], 1), probe, ("turn", P["thumb"], -1), probe):
        ep.step(a, "test")
    log = ep.finish("goal")
    th, dl = f"part:{P['thumb']}.angle", f"part:{P['dial']}.angle"
    log.final_map = {"probe": probe, "literals": {th: "rest"}, "p": 0.5}
    log.consistent = [{"literals": {th: "rest"}, "p": 1 - att_dial}, {"literals": {th: "rest", dl: "pos"}, "p": att_dial}]
    return log, P


def test_evidence_filter_replaces_nearness():
    log, P = _hand_log(0.1)
    assert log.reached_goal
    for mode in ("near", "evidence"):
        c = explain(log, mode=mode).claims
        assert c["deadbolt_part"] == P["thumb"] and not c["bolt_alternatives"], mode   # the dial only came along
    log, P = _hand_log(0.6)                  # the belief attributes the far dial: only the evidence rule keeps it
    assert not explain(log, mode="near").claims["bolt_alternatives"]
    c = explain(log, mode="evidence").claims
    assert c["deadbolt_part"] == P["thumb"] and c["bolt_alternatives"] == [P["dial"]]
    log.locality = {"explain": "evidence"}   # the mode the agent ran with is the default
    assert explain(log).claims["bolt_alternatives"] == [P["dial"]]


# ---------------------------------------------------------------------------------------- shared shapes
def test_mock_shared_shapes_decoys():
    for shared in (False, True):
        w = world("hd_knob_pull_left", "both", "table", n_keys=1, door_dial=True, shared_shapes=shared)
        P, ents = w.truth()["parts"], {e.id: e for e in w.entities()}
        look = lambda n, *k: (ents[P[n]].joint, ents[P[n]].graspable, *(ents[P[n]].attrs[x] for x in k))   # noqa: E731
        same = lambda n, *k: [m for m in P if m != n and look(m, *k) == look(n, *k)]              # noqa: E731
        if not shared:   # (the round knob is round like the dials, but not of their size)
            assert not same("cylinder", "shape") and not same("thumb", "shape") and not same("handle", "shape", "size")
            continue
        assert same("cylinder", "shape") == ["cab_disc"] and same("thumb", "shape") == ["door_wing"]
        assert same("handle", "shape", "size") == ["door_dial"]
        wing = P["door_wing"]
        assert run(w, ("turn", wing, 1)).changed == {f"part:{wing}.angle": ("rest", "pos")}   # persistent
        assert run(w, ("turn_pull", P["handle"], 1)).stalled                                 # bolts nothing
        key = w.truth()["matching_key"]
        run(w, ("pick", key))
        assert run(w, ("insert", key, P["cab_disc"])).stalled                                 # not a keyhole
    w = world("hd_knob_pull_left", "unlocked", "table", n_keys=1, shared_shapes=True)
    assert not run(w, ("pull", w.truth()["parts"]["door_wing"])).stalled and w.goal_reached()   # on the leaf
    sc = sample_scenario(np.random.default_rng(1), "train", shared_shapes=True)
    plain = sample_scenario(np.random.default_rng(1), "train")
    assert sc.door_dial and dataclasses.replace(sc, shared_shapes=False, door_dial=False) == \
        dataclasses.replace(plain, door_dial=False)


def test_recipe_reuse_with_shared_shapes():
    """The decoys share the mechanisms' shapes, so the shape cue of a recipe cannot pick the mechanism;
    reuse still helps, with the shape cue and without it."""
    res = {}
    for lock in ("latch", "deadbolt"):
        cold, warm, blind = [], [], []
        for i in range(8):
            rng = np.random.default_rng(3000 + i)
            first = sample_scenario(rng, "train", lock_state=lock, key_place="table", shared_shapes=True)
            second = sample_scenario(rng, "test", lock_state=lock, key_place="table", shared_shapes=True)
            first_log = DiscoveryAgent(seed=i).run(MockWorld(first), 400)
            book, book0 = RecipeBook(), RecipeBook(b_shape=0.0)
            assert book.learn(first_log) is not None and book0.learn(first_log) is not None
            cold.append(DiscoveryAgent(seed=i).run(MockWorld(second), 400).n_actions)
            warm.append(DiscoveryAgent(seed=i).run(MockWorld(second), 400, recipes=book).n_actions)
            blind.append(DiscoveryAgent(seed=i).run(MockWorld(second), 400, recipes=book0).n_actions)
        res[lock] = (float(np.mean(cold)), float(np.mean(warm)), float(np.mean(blind)))
    print("\nshared shapes, held-out type, actions without / with the recipe / with it but no shape cue:",
          {k: "{:.1f} / {:.1f} / {:.1f}".format(*v) for k, v in res.items()})
    assert all(b < a and c < a for a, b, c in res.values())
