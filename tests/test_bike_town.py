import numpy as np

from pai.bike.delivery import blocked_prior, plan_delivery
from pai.bike.navigation import StreetBelief, avoid_cars, plan
from pai.bike.riding import CORNER_R, curvature, lane_path
from pai.bike.town import Town


def test_town_layout_and_hidden_obstacles():
    t = Town(n=4, seed=4300)
    assert len(t.nodes) == 16 and len(t.edges) == 24 and len(t.houses) == 8
    from pai.bike.town import scoring
    with scoring():
        assert all(0 not in o.edge for o in t.obstacles)      # the depot's streets stay open
        assert t.truth["blocked"] and t.truth["partial"]


def test_rounded_lane_stays_within_its_radius():
    t = Town(n=4, seed=0)
    p = lane_path(t.nodes, [0, 1, 5, 4], radius=CORNER_R)
    assert np.abs(curvature(p)).max() <= 1 / CORNER_R + 1e-3
    assert np.linalg.norm(np.diff(p, axis=0), axis=1).max() < 0.7


def test_planner_avoids_streets_believed_blocked():
    t = Town(n=4, seed=0)
    b = StreetBelief(prior=0.1)
    r = plan(t, b, 0, 3)
    assert r == [0, 1, 2, 3]
    b.update((1, 2), True)
    r2 = plan(t, b, 0, 3)
    assert (1, 2) not in set(zip(r2[:-1], r2[1:])) and r2[-1] == 3


def test_delivery_route_ends_at_the_house_and_respects_a_seen_barrier():
    t = Town(n=4, seed=4300)
    b = StreetBelief(prior=blocked_prior(range(4000, 4004)))
    for house in t.houses:
        r = plan_delivery(t, b, 0, house)
        assert r[-1] == "pad" and r[0] == 0
    (a, c), _, _, frac = t.houses[3]
    # a barrier between end ``a`` and the pad forces the approach from the other end
    pa, pc = np.array(t.nodes[a]), np.array(t.nodes[c])
    b.update((a, c), True, tuple(pa + 0.5 * frac * (pc - pa)))
    r = plan_delivery(t, b, 0, 3)
    assert r[-2] == c


def test_lane_change_around_a_parked_car():
    t = Town(n=4, seed=0)
    p = lane_path(t.nodes, [0, 1, 2])
    car = (12.5, -1.0)                       # in the rider's (right-hand) lane of street 0-1
    q = avoid_cars(p, [car])
    k = int(np.argmin(np.linalg.norm(p - car, axis=1)))
    assert q[k, 1] > 0.5                     # moved into the other lane next to the car
    assert np.allclose(q[0], p[0]) and np.allclose(q[-1], p[-1])


def test_blocked_prior_is_counted_from_practice_towns():
    p = blocked_prior(range(4000, 4004))
    assert 0.05 < p < 0.4


def test_dead_end_risk_counts_the_unchecked_ways_on():
    from pai.bike.navigation import dead_end_risk
    t = Town(n=4, seed=0)
    b = StreetBelief(prior=0.125)
    assert abs(dead_end_risk(t, b, 2, 3) - 0.125) < 1e-9          # town corner: one way on
    assert abs(dead_end_risk(t, b, 0, 1) - 0.125 ** 2) < 1e-9     # edge: two ways on
    b.update((3, 7), False)
    assert dead_end_risk(t, b, 2, 3) < 0.05                       # the way on was seen clear


def test_u_turn_loop_stays_tight_but_bounded():
    t = Town(n=4, seed=0)
    p = lane_path(t.nodes, [0, 1, 2, 1, 5], radius=CORNER_R)
    k = np.abs(curvature(p))
    assert 0.35 < k.max() < 0.7                                    # a ~2.4 m loop, no sharp reversal
    assert np.linalg.norm(np.diff(p, axis=0), axis=1).max() < 0.7


def test_fast_vector_env_matches_the_reference():
    import json
    from pathlib import Path
    from pai.bike.ppo_rider import BikeVecEnv, FastBikeVecEnv
    g = np.array(json.loads(Path("results/bike_g1_turns_v3.json").read_text())["practised"])
    a = BikeVecEnv(4, towns=range(5000, 5002), seed=5, nthread=2, residual_gains=g)
    b = FastBikeVecEnv(4, towns=range(5000, 5002), seed=5, nthread=2, residual_gains=g)
    rng = np.random.default_rng(0)
    for _ in range(20):
        act = rng.normal(0, 0.3, (4, 2))
        oa, ra, da, _ = a.step(act)
        ob, rb, db, _ = b.step(act)
        assert np.allclose(oa, ob, atol=1e-5) and np.allclose(ra, rb, atol=1e-5) and (da == db).all()


def test_truth_guard_blocks_the_rider_from_hidden_obstacles():
    import pytest
    from pai.bike.town import TruthAccessError, scoring
    t = Town(n=4, seed=4300)
    with pytest.raises(TruthAccessError):
        _ = t.truth
    with pytest.raises(TruthAccessError):
        _ = t.obstacles
    with scoring():
        assert t.truth["blocked"]


def test_traffic_is_deterministic_and_drivers_yield_to_the_rider():
    from pai.bike.traffic import Traffic
    t = Town(n=4, seed=4300)
    a, b = Traffic(t, n=4, seed=1), Traffic(t, n=4, seed=1)
    for _ in range(50):
        a.step(0.02, (-99.0, -99.0))
        b.step(0.02, (-99.0, -99.0))
    assert all(np.allclose(x.at()[0], y.at()[0]) for x, y in zip(a.vehicles, b.vehicles))
    v = a.vehicles[0]
    c, yaw = v.at()
    rider = c + 5.0 * np.array([np.cos(yaw), np.sin(yaw)])      # 5 m ahead in its lane
    s0 = v.s
    a.step(0.1, rider)
    assert v.s - s0 < 0.25 * v.speed * 0.1 + 1e-9              # slows strongly behind the rider


def test_tracker_estimates_velocity_and_speed_choice_avoids_a_crossing_vehicle():
    from pai.bike.traffic_perception import TrafficTracker
    tr = TrafficTracker()
    for k in range(4):                                         # an object moving at 3 m/s in +y, 6 m ahead
        tr.update([np.array([6.0, -6.0 + 0.6 * k])], 0.2 * k)
    assert abs(tr.tracks[0]["v"][1] - 3.0) < 0.3 and abs(tr.tracks[0]["v"][0]) < 0.3
    path = np.stack([np.linspace(0, 30, 61), np.zeros(61)], 1)  # the rider rides +x along y = 0
    v, clears = tr.choose_speed(path, 0, np.array([0.0, 0.0]), 0.6, None)
    assert v is not None and clears[v] == max(clears.values())


def test_head_camera_sees_a_moved_vehicle():
    import mujoco

    from pai.bike.navigation import HeadCamera
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1])
    spec.worldbody.add_body(name="g1_torso_link", pos=[0, 0, 0.8]).add_geom(size=[0.05, 0, 0])
    car = spec.worldbody.add_body(name="vehicle_0", mocap=True, pos=[0, 30, 0.7])
    car.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[2.0, 0.85, 0.7])
    model = spec.compile()
    data = mujoco.MjData(model)
    data.mocap_pos[0] = [6.0, 0.0, 0.7]                        # the vehicle has moved in front of the rider
    mujoco.mj_forward(model, data)
    cam = HeadCamera(model)
    _, pts, z = cam.look(data, 0.0)
    cam.close()
    p = pts.reshape(-1, 3)[(z.reshape(-1) < 20) & (pts.reshape(-1, 3)[:, 2] > 0.3)]
    assert len(p) > 20 and abs(np.median(p[:, 0]) - 4.0) < 0.5   # its near face, 4 m ahead


def test_drivers_wait_short_of_a_junction_the_rider_is_at():
    from pai.bike.traffic import Traffic
    t = Town(n=4, seed=4300)
    tr = Traffic(t, n=4, seed=1)
    for v in tr.vehicles:                                      # find a vehicle 5.5-6.5 m before a junction
        for _ in range(2000):
            c, yaw = v.at()
            d = np.array([np.cos(yaw), np.sin(yaw)])
            rj = tr.junctions - c
            along, across = rj @ d, rj @ np.array([-d[1], d[0]])
            ok = (along > 5.5) & (along < 6.5) & (np.abs(across) < 2.5)
            if ok.any():
                break
            v.s += 0.25
        if ok.any():
            break
    j = tr.junctions[int(np.argmax(ok))]
    others = [w for w in tr.vehicles if w is not v]
    for w in others:                                           # only this vehicle moves
        w.speed = 0.0
    s0 = v.s
    tr.step(0.1, j + np.array([0.0, 0.0]) + 3.0 * np.array([-d[1], d[0]]))   # the rider at the junction, to the side
    slowed = v.s - s0
    v.s = s0
    tr.step(0.1, (-99.0, -99.0))                               # the rider far away
    assert slowed < 0.3 * (v.s - s0)
