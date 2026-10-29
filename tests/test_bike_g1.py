"""The G1 on the realistic bicycle: it is seated with hands and feet on the bike, and it rides."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")


def test_g1_is_seated_with_hands_on_grips_and_feet_on_pedals():
    from pai.bike.g1_rider import G1BikeEnv

    env = G1BikeEnv()
    assert env.ik_gap.max() < 0.03                      # the solved posture reaches grips and pedals
    m, d = env.model, env.data
    for a, b in (("g1_left_palm", "grip_l"), ("g1_right_palm", "grip_r"),
                 ("g1_left_foot", "pedal_l"), ("g1_right_foot", "pedal_r")):
        gap = np.linalg.norm(d.site_xpos[m.site(a).id] - d.site_xpos[m.site(b).id])
        assert gap < 0.02


def test_g1_rides_ten_metres_from_an_easy_start():
    from pai.bike.env import BalanceController
    from pai.bike.g1_rider import G1BikeEnv

    env = G1BikeEnv(distance=10.0, max_seconds=10.0)
    s = env.reset(seed=0, speed=3.0, roll=0.0)
    ctrl = BalanceController(target_speed=3.0)
    while not env.done:
        s = env.step(ctrl.action(s))
    assert not env.fallen and env.travelled >= 10.0


def test_realistic_bicycle_has_the_named_parts():
    from pai.bike.real_bike import build_spec

    m = build_spec(scenery=False).compile()
    for name in ("root", "rear_spin", "front_spin", "steering", "crank"):
        m.joint(name)
    for name in ("saddle", "grip_l", "grip_r", "pedal_l", "pedal_r"):
        m.site(name)
    assert 9 < float(m.body_mass[m.body("frame").id:].sum()) < 14     # about 11.4 kg


def test_arms_can_steer_the_handlebar():
    """With arm_steer the fork servo is off; the arm/waist table keeps the palms on the grips
    within +-0.3 rad, and a held bike's handlebar follows commanded angles through the arms."""
    import numpy as np
    from pai.bike.g1_rider import STEER_TABLE, G1BikeEnv
    env = G1BikeEnv(scenery=False, max_seconds=1e9, arm_steer=True, arm_steer_kp=80.0)
    assert env.model.actuator_gainprm[env.a_steer, 0] == 0.0
    inner = np.abs(STEER_TABLE) <= 0.3 + 1e-9
    assert env.arm_table_gap[inner].max() < 0.005
    env.reset(seed=0, speed=0.0, roll=0.0)
    d = env.data
    root = d.qpos[:7].copy()
    for _ in range(40):
        env.step(np.array([0.0, 0.25, 0.0]))
        d.qpos[:7] = root
        d.qvel[:6] = 0.0
    assert abs(d.qpos[env.steer_q] - 0.25) < 0.03
