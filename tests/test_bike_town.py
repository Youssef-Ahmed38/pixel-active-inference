import numpy as np

from pai.bike.delivery import blocked_prior, plan_delivery
from pai.bike.navigation import StreetBelief, avoid_cars, plan
from pai.bike.riding import CORNER_R, curvature, lane_path
from pai.bike.town import Town


def test_town_layout_and_hidden_obstacles():
    t = Town(n=4, seed=4300)
    assert len(t.nodes) == 16 and len(t.edges) == 24 and len(t.houses) == 8
    assert all(0 not in o.edge for o in t.obstacles)          # the depot's streets stay open
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
