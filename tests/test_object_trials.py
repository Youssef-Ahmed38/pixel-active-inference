"""A revealed object is tested before it is put away."""

from types import SimpleNamespace

from pai.discovery.object_trials import ObjectTrialController


def test_revealed_object_trial_carries_through_to_goal_probe():
    obj = SimpleNamespace(id="o2", kind="object", attrs={"size": 0.1}, pos=(0, 0, 0))
    disc = SimpleNamespace(id="p4", kind="part", attrs={"size": 0.04}, pos=(0.2, 0, 0))
    bar = SimpleNamespace(id="p5", kind="part", attrs={"size": 0.12}, pos=(0.1, 0, 0))
    door = SimpleNamespace(id="p2", kind="part", attrs={"size": 2}, pos=(0, 0, 0))
    ep = SimpleNamespace(state={"body:hand": "empty", "obj:o2.where": "free"},
                         ents={e.id: e for e in (obj, disc, bar, door)}, goal_part="p2",
                         log=SimpleNamespace(actions=[], outcomes=[]), cost=lambda action: 1.0)
    trial = ObjectTrialController()
    trial.observe(ep, ("pull", "p1"), SimpleNamespace(revealed=[obj], executed=True,
                                                      stalled=False, changed={}), [])
    assert trial.choose(ep, [("pick", "o2")], []) == ("pick", "o2")
    ep.state.update({"body:hand": "o2", "obj:o2.where": "held"})
    trial.observe(ep, ("pick", "o2"), SimpleNamespace(revealed=[], executed=True,
                                                      stalled=False, changed={}), [])
    inserts = [("insert", "o2", "p5"), ("insert", "o2", "p4")]
    assert trial.choose(ep, inserts + [("release", "o2")], []) == ("insert", "o2", "p4")
    ep.state["obj:o2.where"] = "in:p4"
    trial.observe(ep, ("insert", "o2", "p4"), SimpleNamespace(revealed=[], executed=True,
                                                               stalled=False, changed={}), [])
    assert trial.choose(ep, [("turn_held", "o2", 1), ("turn_held", "o2", -1)], []) == ("turn_held", "o2", 1)
    trial.observe(ep, ("turn_held", "o2", 1), SimpleNamespace(revealed=[], executed=True,
                                                               stalled=False, changed={"part:p4.angle": ("rest", "pos")}), [])
    assert trial.choose(ep, [("release", "o2")], []) == ("release", "o2")
    ep.state["body:hand"] = "empty"
    probe = ("turn_pull", "p5", 1)
    trial.observe(ep, ("release", "o2"), SimpleNamespace(revealed=[], executed=True,
                                                          stalled=False, changed={}), [probe])
    assert trial.choose(ep, [probe], [probe]) == probe


def test_disconfirmed_object_is_withdrawn_before_next_candidate():
    o1 = SimpleNamespace(id="o1", kind="object", attrs={}, pos=(0, 0, 0))
    o2 = SimpleNamespace(id="o2", kind="object", attrs={}, pos=(0, 0, 0))
    ep = SimpleNamespace(state={"body:hand": "empty", "obj:o2.where": "in:p4",
                                "obj:o1.where": "free"}, ents={"o1": o1, "o2": o2},
                         goal_part="p2", log=SimpleNamespace(actions=[], outcomes=[]),
                         cost=lambda action: 1.0)
    trial = ObjectTrialController()
    trial.attempted.add("o2")
    trial.probe_object = "o2"  # its observed turn did not open the goal after all probes
    assert trial.choose(ep, [("pick", "o2"), ("pick", "o1")], []) == ("pick", "o2")
    ep.state.update({"body:hand": "o2", "obj:o2.where": "held"})
    trial.observe(ep, ("pick", "o2"), SimpleNamespace(revealed=[], executed=True,
                                                      stalled=False, changed={}), [])
    assert trial.choose(ep, [("release", "o2")], []) == ("release", "o2")
    ep.state.update({"body:hand": "empty", "obj:o2.where": "free"})
    trial.observe(ep, ("release", "o2"), SimpleNamespace(revealed=[], executed=True,
                                                          stalled=False, changed={}), [])
    assert trial.choose(ep, [("pick", "o1"), ("pick", "o2")], []) == ("pick", "o1")


def test_post_turn_probe_uses_observed_rotation_near_goal():
    positions = {"p2": (0, 0, 0), "p5": (0.1, 0, 0), "p3": (0.8, 0, 0)}
    entities = {key: SimpleNamespace(id=key, pos=pos) for key, pos in positions.items()}
    near, far = ("turn_pull", "p5", 1), ("turn_pull", "p3", 1)
    ep = SimpleNamespace(state={"body:hand": "empty"}, ents=entities, goal_part="p2",
                         log=SimpleNamespace(actions=[far, near],
                                             outcomes=[SimpleNamespace(notes="moved=0 turned=0.68"),
                                                       SimpleNamespace(notes="moved=0 turned=0.68")]),
                         cost=lambda action: 1.0)
    trial = ObjectTrialController()
    trial.active = "o1"
    trial.successful_turn = True
    trial.observe(ep, ("release", "o1"), SimpleNamespace(revealed=[], executed=True,
                                                          stalled=False, changed={}), [far, near])
    assert trial.choose(ep, [far, near], [far, near]) == near


def test_stalled_probe_restores_changed_part_then_retries():
    probe = ("turn_pull", "p1", 1)
    ep = SimpleNamespace(state={"body:hand": "empty", "obj:o2.where": "in:p3",
                                "part:p4.angle": "pos"},
                         world=SimpleNamespace(goal_reached=lambda: False))
    trial = ObjectTrialController({"part:p4.angle": "rest"})
    trial.probe_object = "o2"
    trial.primary_probe = probe
    trial.probes = [("push", "p2")]
    trial.observe(ep, probe, SimpleNamespace(revealed=[], executed=True, stalled=True,
                                             changed={}), [probe])
    restore = ("turn", "p4", -1)
    assert trial.choose(ep, [restore, probe], [probe]) == restore
    ep.state["part:p4.angle"] = "rest"
    assert trial.choose(ep, [probe], [probe]) == probe


def test_stalled_probe_retracts_an_initially_engaged_part():
    probe = ("turn_pull", "p1", 1)
    ep = SimpleNamespace(state={"body:hand": "empty", "obj:o2.where": "in:p3",
                                "part:p3.angle": "pos", "part:p4.angle": "pos"},
                         world=SimpleNamespace(goal_reached=lambda: False))
    trial = ObjectTrialController({"part:p3.angle": "pos", "part:p4.angle": "pos"})
    trial.probe_object = "o2"
    trial.primary_probe = probe
    trial.observe(ep, probe, SimpleNamespace(revealed=[], executed=True, stalled=True,
                                             changed={}), [probe])
    assert trial.choose(ep, [("turn", "p4", -1), probe], [probe]) == ("turn", "p4", -1)
