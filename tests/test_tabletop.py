import numpy as np
import pytest

from pai.config import REPO_ROOT, Config, load_config
from pai.envs import TabletopEnv, build_disturbances
from pai.envs.scripted import PickPlace

CFG = REPO_ROOT / "configs" / "tabletop.yaml"


def make_env(*overrides, disturbances=None):
    cfg = load_config(CFG, ["env.image_size=64", "env.render_images=false", *overrides])
    return TabletopEnv(cfg.env, disturbances=disturbances)


@pytest.fixture(scope="module")
def env():
    e = make_env()
    yield e
    e.close()


def test_config_wraps_dicts_inside_lists():
    cfg = Config({"cams": [{"name": "a", "pos": [1, 2]}]})
    assert cfg.cams[0].name == "a"
    assert cfg.to_dict() == {"cams": [{"name": "a", "pos": [1, 2]}]}


def test_objects_rest_on_table(env):
    obs = env.reset(seed=0)
    for name, spec in env.objects.items():
        z = obs["state"][name]["pos"][2]
        expected = spec.height / 2 if spec.kind != "bowl" else 0.0
        assert z == pytest.approx(expected, abs=0.004), name
    assert {("upright", b) for b in ("red", "green", "blue", "yellow")} <= obs["predicates"]


def test_layout_is_reproducible_and_respected(env):
    a = env.reset(seed=3)["state"]["red"]["pos"]
    b = env.reset(seed=3)["state"]["red"]["pos"]
    assert np.allclose(a, b)
    c = env.reset(seed=3, layout={"red": [0.5, 0.1]})["state"]["red"]["pos"]
    assert np.allclose(c[:2], [0.5, 0.1], atol=0.005)


def test_segmentation_labels_every_object(env):
    env.reset(seed=0)
    labels = set(np.unique(env.render_segmentation("top")))
    assert {2 + i for i in range(len(env.objects))} <= labels  # every object is visible from above
    assert 1 in labels  # the robot


def test_ee_control_moves_hand(env):
    obs = env.reset(seed=0)
    start = obs["ee_pos"].copy()
    for _ in range(40):
        obs = env.step(np.array([0.1, 0.0, 0.0, 0.0, 1.0]))
    moved = obs["ee_pos"] - start
    assert moved[0] > 0.05 and abs(moved[1]) < 0.02 and abs(moved[2]) < 0.02


def test_scripted_pick_place_and_event_log(env):
    for seed in range(3):
        obs = env.reset(seed=seed)
        pp = PickPlace(env, "red", "plate")
        for _ in range(900):
            obs = env.step(pp.act(obs))
            if pp.done():
                break
        for _ in range(20):
            obs = env.step(np.array([0, 0, 0, 0, 1.0]))
        assert ("on", "red", "plate") in obs["predicates"], seed
        types = [e["type"] for e in env.events.events]
        assert types.count("grasp") == 1 and types.count("release") == 1
        on_events = [e for e in env.events.of_type("relation_true") if e["relation"] == ["on", "red", "plate"]]
        assert on_events and on_events[0]["t"] > env.events.of_type("grasp")[0]["t"]


def test_disturbances_are_logged_and_reset():
    specs = [
        {"type": "push", "start": 5, "duration": 10, "body": "link5", "force": [0, 40, 0]},
        {"type": "friction", "start": 0, "body": "red", "scale": 0.1},
        {"type": "payload", "start": 0, "body": "red", "mass": 0.5},
    ]
    env = make_env(disturbances=build_disturbances(specs))
    env.reset(seed=0)
    red_geom = env.model.geom("red_geom").id
    nominal_friction = env._nominal["geom_friction"][red_geom, 0]
    for _ in range(20):
        env.step(np.array([0, 0, 0, 0, 1.0]))
    assert env.model.geom_friction[red_geom, 0] == pytest.approx(0.1 * nominal_friction)
    assert env.model.body_mass[env.model.body("red").id] > env._nominal["body_mass"][env.model.body("red").id]
    logged = {(e["type"], e["disturbance"]) for e in env.events.of_type("disturbance_start", "disturbance_end")}
    assert {("disturbance_start", "push"), ("disturbance_end", "push"), ("disturbance_start", "friction")} <= logged
    env.reset(seed=0)
    assert env.model.geom_friction[red_geom, 0] == pytest.approx(nominal_friction)
    env.close()


def test_compliant_arm_yields_more_to_pushes():
    def deflection(stiffness):
        push = [{"type": "push", "start": 5, "duration": 15, "body": "link5", "force": [0, 40, 0]}]
        env = make_env(f"env.stiffness={stiffness}", disturbances=build_disturbances(push))
        ee0 = env.reset(seed=0)["ee_pos"].copy()
        worst = 0.0
        for _ in range(40):
            worst = max(worst, float(np.linalg.norm(env.step(np.array([0, 0, 0, 0, 1.0]))["ee_pos"] - ee0)))
        env.close()
        return worst

    assert deflection(0.15) > 3 * deflection(1.0)


def test_perceived_relations_match_truth_when_perception_is_exact():
    """The agent's relations come from its own tokens, not the simulator's predicates. With exact
    perception they agree with the ground truth (except 'upright', which tokens cannot show)."""
    import numpy as np
    from pai.config import load_config
    from pai.envs import TabletopEnv
    from pai.envs.predicates import perceived_predicates
    from pai.world.entities import encode, entity_names

    cfg = load_config("configs/tabletop.yaml", [])
    env = TabletopEnv(type(cfg.env)({**cfg.env, "render_images": False}), disturbances=[])
    names = entity_names(env)
    for seed in range(4):
        obs = env.reset(seed=seed)
        truth = {p for p in obs["predicates"] if p[0] != "upright"}
        assert perceived_predicates(env.objects, encode(env, obs), names) == truth
    # holding: fingers closed on the object at the hand -> grasped, from the body itself
    tokens = encode(env, obs)
    tokens[0, :3] = tokens[1, :3] + np.array([0.0, 0.0, 0.01])
    tokens[0, 8] = 0.04  # finger opening (GRIP slot): closed on a 4 cm block
    assert ("grasped", names[1]) in perceived_predicates(env.objects, tokens, names)
    env.close()
