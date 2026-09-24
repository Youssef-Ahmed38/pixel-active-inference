import numpy as np
import torch

from pai.causes.inference import StepEvidence, SurpriseMonitor, infer_cause
from pai.goals.relations import RelationalGoal
from pai.envs.predicates import ObjectSpec
from pai.planning.mppi import MPPIPlanner
from pai.world.entities import DYNAMIC, FORCE, GRIP, POS, TOKEN_DIM
from pai.world.model import EnsembleWorldModel, load_world_model

SIG = np.r_[np.full(3, 0.001), np.full(3, 0.3), np.full(6, 0.001)]


def _window(n=20, rng=None, push=None, sag=0.0, holding=False, slip=None, camera=None):
    """Synthetic evidence. push=(t0, t1, offset on the 9 arm channels); slip=(t0, object offset,
    force change); camera=(t0, jump of everything seen)."""
    rng = rng or np.random.default_rng(0)
    evs = []
    for t in range(n):
        r = rng.normal(0, 1, 12) * SIG
        a = np.array([0, 0, 0.2 if holding else 0.0, -1.0])
        if push and push[0] <= t < push[1]:
            r[:9] += push[2]
        if holding:
            r[2] -= sag
            r[8] -= sag  # the held object sags with the hand
            r[5] += sag * 1000  # 1 mm of sag comes with ~1 N of extra pull
        if slip and holding and t >= slip[0]:
            r[6:9] += slip[1]
            r[3:6] += slip[2]
        if camera and t == camera[0]:
            r[6:9] += camera[1]
            r[9:12] += camera[1]
        evs.append(StepEvidence(t, r, SIG, a, holding))
    return evs


def test_cause_inference_identifies_each_cause():
    assert infer_cause(_window(), 0.1).best == "none"
    push = np.r_[0.004, -0.002, 0.0, 20.0, -10.0, 0.0, 0.0, 0.0, 0.0]
    assert infer_cause(_window(push=(6, 11, push)), 0.1).best == "push"
    assert infer_cause(_window(sag=0.004, holding=True), 0.1).best == "heavier_object"
    slip = (8, np.array([0.0, 0.0, -0.006]), np.array([0.0, 0.0, -2.0]))
    assert infer_cause(_window(holding=True, slip=slip), 0.1).best == "slippery_object"
    rep = infer_cause(_window(camera=(9, np.array([-0.04, 0.02, 0.0]))), 0.1)
    assert rep.best == "camera_shift"
    assert np.allclose(rep.params["camera_shift"]["camera_offset_mm"], [40, -20, 0], atol=2)


def test_heavier_object_impossible_without_holding():
    rep = infer_cause(_window(sag=0.004, holding=False), 0.1)
    assert rep.posterior["heavier_object"] == 0.0


def test_surprise_monitor_triggers_only_on_spikes():
    mon = SurpriseMonitor(0.1)
    for ev in _window(n=40):
        assert mon.add(ev) is None
    for ev in _window(n=40, push=(10, 16, np.r_[0.01, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
        mon.add(ev)
    assert mon.reports and mon.episode_verdict().best == "push"


def test_world_model_shapes_and_checkpoint(tmp_path):
    wm = EnsembleWorldModel(4, n_members=2, d_model=32, n_layers=1, n_heads=2)
    x, a = torch.randn(3, 7, TOKEN_DIM), torch.randn(3, 4)
    p = wm.predict(x, a)
    nxt, mean, alea, epi = p.next_tokens, p.delta, p.aleatoric, p.epistemic
    assert nxt.shape == x.shape and mean.shape == (3, 7, DYNAMIC.stop) and (alea > 0).all() and (epi >= 0).all()
    assert p.force.shape == (3, 3) and (p.force_var > 0).all()
    assert torch.equal(nxt[..., DYNAMIC.stop:FORCE.start], x[..., DYNAMIC.stop:FORCE.start])  # kind, colour
    x2 = x.clone()
    x2[..., FORCE] += 100.0
    assert torch.allclose(wm.predict(x2, a).force, p.force)  # force is predicted, never read
    torch.save({"model": wm.state_dict(), "model_config": wm.config()}, tmp_path / "wm.pt")
    wm2 = load_world_model(tmp_path / "wm.pt")
    assert torch.allclose(wm2.predict(x, a).delta, mean)


class _PointMass:
    """Toy world model: the gripper moves by action * dt; used to check that MPPI optimises."""

    members = [None]

    def predict(self, x, a, member=None):
        from pai.world.model import Prediction

        nxt = x.clone()
        nxt[..., 0, POS] = x[..., 0, POS] + a[..., :3] * 0.1
        z = torch.zeros(*x.shape[:-1], DYNAMIC.stop)
        f = torch.zeros(*x.shape[:-2], 3)
        return Prediction(nxt, z, z + 1e-6, z, f, f + 1e-6)


def test_mppi_moves_towards_goal():
    target = torch.tensor([0.1, -0.05, 0.2])
    cost = lambda x: ((x[..., 0, POS] - target) ** 2).sum(-1)
    # The toy's costs are ~100x smaller than the task's, so the smoothness prior is switched off here.
    planner = MPPIPlanner(_PointMass(), horizon=8, samples=128, iterations=3, smooth_weight=0.0)
    x = torch.zeros(2, TOKEN_DIM)
    for _ in range(15):
        a, _ = planner.act(x, cost)
        x = _PointMass().predict(x[None], a[None])[0][0]
    assert float((x[0, POS] - target).norm()) < 0.02


def test_goal_subgoals_advance_in_order():
    objects = {"red": ObjectSpec("red", "block", 0.02, 0.04), "plate": ObjectSpec("plate", "plate", 0.07, 0.012)}
    goal = RelationalGoal("on", "red", "plate", ["gripper", "red", "plate"], objects)
    x = torch.zeros(3, TOKEN_DIM)
    x[1, POS] = torch.tensor([0.5, 0.0, 0.02])
    x[2, POS] = torch.tensor([0.5, 0.2, 0.006])
    x[0, GRIP] = 0.08
    x[0, POS] = x[1, POS] + torch.tensor([0, 0, 0.10])
    goal.update(x)
    assert goal.current.name == "at_object"
    x[0, POS] = x[1, POS] + torch.tensor([0, 0, 0.006])
    goal.update(x)
    assert goal.current.name == "grasped"


def test_episode_report_mentions_explanation_and_truth():
    from pai.memory.episodic import EpisodeRecord
    from pai.memory.report import episode_report

    rec = EpisodeRecord(0, "on(red, plate)", True, 50, {"grasped": 10}, [1.0, 30.0, 2.0], "push",
                        {"none": 0.05, "push": 0.9, "heavier_object": 0.05}, {"push": {"onset": 1}},
                        [{"t": 60, "type": "disturbance_start", "disturbance": "push"}])
    text = episode_report(rec)
    assert "something pushed my arm" in text and "probability 0.90" in text and "push at 1.2 s" in text


def test_masks_to_patches_fractions():
    from pai.perception.features import masks_to_patches

    m = np.zeros((1, 28, 28), np.uint8)
    m[0, :14, :7] = 2  # half of the top-left patch
    p = masks_to_patches(m, 3).astype(np.float32)
    assert p.shape == (1, 2, 2, 3)
    assert np.isclose(p[0, 0, 0, 2], 0.5) and np.isclose(p[0, 0, 0, 0], 0.5) and np.allclose(p.sum(-1), 1)


def test_goal_falls_back_to_regrasp_when_object_is_dropped():
    objects = {"red": ObjectSpec("red", "block", 0.02, 0.04), "plate": ObjectSpec("plate", "plate", 0.07, 0.012)}
    goal = RelationalGoal("on", "red", "plate", ["gripper", "red", "plate"], objects)
    x = torch.zeros(3, TOKEN_DIM)
    x[1, POS] = torch.tensor([0.5, 0.0, 0.08])   # red held in the air
    x[2, POS] = torch.tensor([0.5, 0.2, 0.006])
    x[0, POS] = x[1, POS]
    x[0, GRIP] = 0.04
    goal.index = 3  # lifted
    goal.update(x)
    assert not goal.fell_back
    x[1, POS] = torch.tensor([0.5, 0.0, 0.02])   # dropped: fell out of the hand
    goal.update(x)
    assert goal.fell_back and goal.current.name in ("above_object", "at_object")


def test_calibration_sets_floor_and_threshold():
    from pai.causes.inference import calibrate

    rng = np.random.default_rng(0)
    res = rng.normal(0, 1, (2000, 12)) * np.r_[[0.003] * 3, [1.0] * 3, [0.003] * 6]  # real errors: 3 mm, 1 N
    sig = np.tile(np.r_[[0.0005] * 3, [0.2] * 3, [0.0005] * 6], (2000, 1))           # while it claims 0.5 mm, 0.2 N
    cal = calibrate(res, sig)
    assert np.allclose(cal.floor[:3], 0.003, rtol=0.1) and np.allclose(cal.floor[3:6], 0.98, rtol=0.1)
    assert 1.0 < cal.threshold < 20.0


def test_calibration_per_step_keeps_quiet_steps_sensitive():
    from pai.causes.inference import calibrate

    rng = np.random.default_rng(0)
    n = 1000
    steps = ["grasped"] * n + ["lifted"] * n
    noise = np.r_[[0.001] * 3, [5.0] * 3, [0.001] * 6], np.r_[[0.001] * 3, [0.05] * 3, [0.001] * 6]   # grasping is noisy, carrying is quiet
    res = np.vstack([rng.normal(0, 1, (n, 12)) * noise[0], rng.normal(0, 1, (n, 12)) * noise[1]])
    sig = np.full((2 * n, 12), 1e-4)
    cal = calibrate(res, sig, steps)
    assert cal.floor_for("grasped")[5] > 4.0 and cal.floor_for("lifted")[5] < 0.4  # 0.4 = floor minimum 0.3 N
    assert cal.floor_for("never_seen")[5] == cal.floor[5]


def test_planner_respects_gripper_mode_from_goal_level():
    cost = lambda x: ((x[..., 0, POS] - torch.tensor([0.1, 0.0, 0.0])) ** 2).sum(-1)
    planner = MPPIPlanner(_PointMass(), horizon=4, samples=32, iterations=2, smooth_weight=0.0)
    a, _ = planner.act(torch.zeros(2, TOKEN_DIM), cost, grip=-1.0)
    assert float(a[3]) == -1.0 and (planner.plan[:, 3] == -1.0).all()


def test_counterfactual_credit_finds_the_necessary_factor():
    """Toy world: the gripper must travel to x = 0.3. Skipping the move breaks success; slowing does too;
    skipping a phase where nothing happens does not."""
    from pai.causes.credit import assign_credit

    T = 10
    tokens = np.zeros((T + 1, 2, TOKEN_DIM), np.float32)
    actions = np.zeros((T, 4), np.float32)
    actions[:6, 0] = 0.5  # move for 6 steps at 0.5 m/s * 0.1 s = 0.3 m
    phases = {"move": (0, 6), "wait": (6, 10)}
    goal_cost = lambda x: float((x[0, POS][0] - 0.3) ** 2)
    base, results = assign_credit(_PointMass(), tokens, actions, goal_cost, phases, success_threshold=1e-3)
    by = {r.name: r for r in results}
    assert base < 1e-6
    assert by["skip_move"].necessary and by["slow_down"].necessary
    assert not by["skip_wait"].necessary
    assert results[0].name == "skip_move"  # most important first


def test_memory_saves_trajectories_beside_jsonl(tmp_path):
    from pai.memory.episodic import EpisodeRecord, EpisodicMemory

    mem = EpisodicMemory(tmp_path / "mem.jsonl")
    rec = EpisodeRecord(3, "on(red, plate)", True, 5, {}, [], "none", None, None, [],
                        tokens=np.zeros((6, 2, TOKEN_DIM), np.float32), actions=np.zeros((5, 4), np.float32),
                        phases={"lifted": (1, 3)})
    mem.add(rec)
    row = __import__("json").loads((tmp_path / "mem.jsonl").read_text())
    assert row["trajectory"] == "mem_ep0003.npz" and "tokens" not in row
    assert np.load(tmp_path / "mem_ep0003.npz")["actions"].shape == (5, 4)
    assert mem.successes() == [rec]


def test_unknown_wins_only_when_no_cause_fits():
    # an oscillating error on every channel: large, but shaped like none of the known causes
    rng = np.random.default_rng(1)
    evs = []
    for t in range(20):
        r = rng.normal(0, 1, 12) * SIG * 8 * (1 if t % 2 else -1)
        evs.append(StepEvidence(t, r, SIG, np.array([0, 0, 0, -1.0]), holding=True))
    assert infer_cause(evs, 0.1).best == "unknown"
    assert infer_cause(_window(), 0.1).posterior["unknown"] < 0.05  # plain noise is not "unknown"


def test_lowered_waits_until_object_is_still():
    objects = {"red": ObjectSpec("red", "block", 0.02, 0.04), "plate": ObjectSpec("plate", "plate", 0.07, 0.012)}
    goal = RelationalGoal("on", "red", "plate", ["gripper", "red", "plate"], objects)
    lowered = next(g for g in goal.subgoals if g.name == "lowered")
    x = torch.zeros(3, TOKEN_DIM)
    x[2, POS] = torch.tensor([0.5, 0.2, 0.006])
    x[1, POS] = torch.tensor([0.5, 0.2, 0.012 + 0.02 + 0.025])  # at the release height over the plate
    x[0, POS], x[0, GRIP] = x[1, POS], 0.04
    from pai.world.entities import VEL
    x[1, VEL] = torch.tensor([0.0, 0.0, -0.2])  # still falling
    assert not lowered.done(x)
    x[1, VEL] = 0.0
    assert lowered.done(x)


def test_recipe_extraction_and_proposal_transfer_to_a_new_layout():
    """A recipe stores where the hand went relative to the roles, so in a new layout its proposal
    heads for the new object, not for where the old one was."""
    from types import SimpleNamespace

    from pai.memory.recipes import check_preconditions, extract_recipe, proposal

    names = ["gripper", "red", "plate"]
    objects = {"red": SimpleNamespace(kind="block"), "plate": SimpleNamespace(kind="plate")}
    T = 6
    tokens = np.zeros((T + 1, 3, TOKEN_DIM), np.float32)
    tokens[:, 1, POS] = [0.5, 0.1, 0.02]      # the object starts here
    tokens[:, 2, POS] = [0.4, -0.2, 0.006]    # the plate
    tokens[:3, 0, GRIP] = 0.08                # hand open until the grasp
    tokens[2, 0, POS] = [0.5, 0.1, 0.12]      # end of above_object: 10 cm above the object
    tokens[5, 0, POS] = [0.4, -0.2, 0.06]     # end of over_target: above the plate
    rec = SimpleNamespace(tokens=tokens, phases={"above_object": (0, 2), "over_target": (3, 5)}, episode=7)
    r = extract_recipe(rec, "on", "red", "plate", names, objects, ["above_object", "over_target"])
    assert [s.reference for s in r.steps] == ["object", "target"]
    assert np.allclose(r.steps[0].offset, [0, 0, 0.10], atol=1e-6)
    assert "hand open" in r.preconditions and check_preconditions(r, tokens[0], 1, 2) == []

    new = tokens[0].copy()
    new[1, POS] = [0.3, 0.3, 0.02]            # new layout
    prop = proposal(r, "above_object", new, tokens[0, 1, POS], 2, horizon=20, dt=0.1, i_o=1)
    end = new[0, POS] + (prop[:, :3] * 0.1).sum(0)
    assert np.allclose(end, [0.3, 0.3, 0.12], atol=1e-4)
    assert proposal(r, "released", new, new[1, POS], 2, horizon=5, dt=0.1) is None


def test_push_while_holding_is_not_a_slip():
    """A sideways push on a hand that holds the object moves the object and the wrist force
    sideways; slipping is gravity-driven and cannot explain that."""
    push = np.r_[0.001, 0.0, 0.0, 25.0, -15.0, 0.0, 0.003, 0.0, 0.0]
    assert infer_cause(_window(holding=True, push=(6, 9, push)), 0.1).best == "push"
