"""One parcel delivery on the bicycle: depot -> house, with looking and replanning.

``Delivery.run`` rides the G1 from the depot to a house's drop-off pad. The route is planned on the
street map with the current belief (``pai/bike/navigation.py``). Before every intersection the rider
looks down the street it plans to take next (10 m before it when that street is straight ahead, 4 m
when it is around a corner); a barrier makes it replan from that intersection (avoiding a U-turn when
another way exists, else looping round the intersection), a parked car makes it plan a lane change
around it. The ``blind`` rider is the baseline: same steering, same shortest route, no looking.

Collisions are checked geometrically against the hidden obstacles (the visual geoms do not collide):
the rider's position inside a barrier's line across the street or inside a car's footprint (plus a
0.3 m margin) ends the delivery as a collision.
"""

from __future__ import annotations

import math

import numpy as np

from pai.bike.navigation import DETOUR, HeadCamera, StreetBelief, avoid_cars, inspect_street, plan
from pai.bike.riding import CORNER_R, LeanFollower, lane_path
from pai.bike.town import LANE, ROAD_HALF, Town

LOOK_M = 4.0           # streets around a corner (corner buildings hide them from further back)
LOOK_STRAIGHT_M = 10.0  # the street straight ahead
RANGE_M = 35.0
PAD_R = 1.5
OFF_ROAD_M = 3.0


def blocked_prior(seeds) -> float:
    """The rate of blocked streets the rider met in its practice towns (counted, not given)."""
    n = k = 0
    for s in seeds:
        t = Town(n=4, seed=s)
        n += sum(0 not in e for e in t.edges)
        k += len(t.truth["blocked"])
    return k / n


def plan_delivery(town: Town, belief: StreetBelief, start: int, house: int, banned=()):
    """Best route from ``start`` to the house pad: a node list ending with 'pad', or None."""
    (a, b), _, _, t = town.houses[house]
    e = (a, b)
    L = town.block
    best = None
    for end, frac in ((a, t), (b, 1 - t)):
        r = [start] if start == end else plan(town, belief, start, end, banned)
        if r is None or (len(r) > 1 and tuple(sorted(r[-2:])) == e):
            continue
        cost = sum(float(np.linalg.norm(np.subtract(town.nodes[u], town.nodes[v]))) + belief.get((u, v)) * DETOUR
                   for u, v in zip(r[:-1], r[1:]))
        # the last leg into the house street: blocked only if a barrier stands between the end and the pad
        if e in belief.barriers:
            from_end = np.linalg.norm(np.subtract(belief.barriers[e], town.nodes[end]))
            pb = 0.97 if from_end < frac * L else 0.03
        elif e in belief.looked:
            pb = 0.03
        else:
            pb = belief.prior * frac
        if pb > 0.9:
            continue
        cost += frac * L + pb * DETOUR
        if best is None or cost < best[0]:
            best = (cost, r + ["pad"])
    return None if best is None else best[1]


class Delivery:
    def __init__(self, town: Town, env, gains, prior: float, camera: HeadCamera | None):
        self.town, self.env, self.gains, self.cam = town, env, np.asarray(gains), camera
        self.prior = prior

    # ---------------------------------------------------------------- geometry helpers
    def _nodes(self, house):
        (a, b), _, _, t = self.town.houses[house]
        nodes = dict(self.town.nodes)
        nodes["pad"] = tuple(np.add(nodes[a], t * np.subtract(nodes[b], nodes[a])))
        return nodes

    @staticmethod
    def _node_index(nodes, route, path):
        """Path index nearest each route node (searched in order along the path)."""
        idx, k = [0], 0
        for v in route[1:]:
            seg = path[k:]
            k = k + int(np.argmin(np.linalg.norm(seg[: 120] - np.asarray(nodes[v]), axis=1)))
            idx.append(k)
        return idx

    def _collision(self, xy):
        for o in self.town.obstacles:
            a, b = o.edge
            pa, pb = np.array(self.town.nodes[a], float), np.array(self.town.nodes[b], float)
            d = (pb - pa) / np.linalg.norm(pb - pa)
            left = np.array([-d[1], d[0]])
            c = np.array(o.pos)
            if o.kind == "partial":
                c = c + left * o.lane * LANE
                rel = xy - c
                if abs(rel @ d) < 2.0 + 0.3 and abs(rel @ left) < 0.85 + 0.3:
                    return o
            else:
                rel = xy - c
                if abs(rel @ d) < 0.12 + 0.6 + 0.3 and abs(rel @ left) < ROAD_HALF:
                    return o
        return None

    # ---------------------------------------------------------------- one delivery
    def run(self, house: int, seed: int, blind: bool = False, max_t: float = 240.0, record: bool = False):
        town, env = self.town, self.env
        nodes = self._nodes(house)
        belief = StreetBelief(prior=self.prior)
        depot = town.depot[0]
        route = plan_delivery(town, belief, depot, house)
        events, frames = [], []

        def build(route, at=None, xy=None):
            path = avoid_cars(lane_path(nodes, route, radius=CORNER_R), belief.cars.values())
            f = LeanFollower(path, self.gains)
            nidx = self._node_index(nodes, route, path)
            if at is not None:          # continue from where the rider is, just before route node ``at``
                lo = max(0, nidx[at] - 40)
                f.i = lo + int(np.argmin(np.linalg.norm(path[lo:nidx[at] + 5] - xy, axis=1)))
            return path, f, nidx

        path, f, nidx = build(route)
        d = path[min(5, len(path) - 1)] - path[0]
        s = env.reset(seed=seed, speed=2.5, roll=0.0, start_xy=path[0], yaw=float(np.arctan2(d[1], d[0])))
        looked_at = set()
        head, cross = [], []
        outcome, hit, replans, looks, wrong_looks = "timeout", None, 0, 0, 0
        while env.data.time < max_t:
            xy = env.data.qpos[:2].copy()
            # --- look ahead before each intersection on the route
            if not blind:
                for i in range(len(route) - 1):
                    if i in looked_at:
                        continue
                    a = route[i]
                    b = route[i + 1] if route[i + 1] != "pad" else None
                    if b is None:       # the house street: look toward its far end
                        e = town.houses[house][0]
                        b = e[1] if e[0] == a else e[0]
                    # a street straight ahead is seen from further back than one around a corner
                    straight = i > 0 and np.allclose(
                        np.sign(np.subtract(town.nodes[b], town.nodes[a])),
                        np.sign(np.subtract(town.nodes[a], town.nodes[route[i - 1]])))
                    reach = (LOOK_STRAIGHT_M if straight else LOOK_M) + LANE
                    if f.i > nidx[i] or np.linalg.norm(xy - np.asarray(nodes[a])) > reach:
                        continue
                    looked_at.add(i)
                    blocked, where, img, npts = inspect_street(town, self.cam, env.data, a, b, max_range=RANGE_M)
                    looks += 1
                    e = tuple(sorted((a, b)))
                    truth = "blocking" if e in town.truth["blocked"] else "partial" if e in town.truth["partial"] else "clear"
                    seen = "blocking" if blocked else "partial" if where is not None else "clear"
                    wrong_looks += truth != seen
                    belief.update(e, blocked, where)
                    look_yaw = math.atan2(*(np.subtract(town.nodes[b], xy))[::-1])
                    ev = {"t": round(env.data.time, 2), "at": a, "street": [a, b], "seen": seen, "points": npts,
                          "look_yaw": round(look_yaw, 3), "where": None if where is None else [round(w, 2) for w in where],
                          "truth": truth,      # scoring only; not used by the rider
                          "from_node_m": round(float(np.linalg.norm(xy - np.asarray(town.nodes[a]))), 2)}
                    if seen != "clear":
                        prev = route[i - 1] if i > 0 else None
                        new = None
                        if prev is not None:
                            new = plan_delivery(town, belief, a, house, banned={tuple(sorted((prev, a)))})
                        if new is None:
                            new = plan_delivery(town, belief, a, house)
                        if new is None:
                            outcome = "no_route"
                            break
                        new_route = route[:i] + new
                        changed = new_route != route
                        if changed or seen == "partial":
                            replans += changed
                            ev["decision"] = ("replan via " + "-".join(map(str, new))) if changed else "change lane"
                            route = new_route
                            path, f, nidx = build(route, at=i, xy=xy)
                            # looks at streets already passed stay done; a new route is looked at again
                            # from this intersection at once
                            looked_at = {j for j in looked_at if j < i or (j == i and not changed)}
                    events.append(ev)
                    if record:
                        frames.append({"t": ev["t"], "pov": img})
                    if i in looked_at:
                        break
                if outcome == "no_route":
                    break
            a_, info = f.action(s, xy)
            s = env.step(a_)
            head.append(abs(info["heading_error"]))
            cross.append(info["cross_track"])
            if record:
                frames.append({"t": round(env.data.time, 3), "qpos": env.data.qpos.copy(), "speed": float(s[2]),
                               "roll": float(s[0]), "steer": float(s[3]), "lean_ref": info["lean_ref"],
                               "route": list(route), "target_speed": info["target_speed"]})
            if env.fallen:
                outcome = "fell"
                break
            if info["cross_track"] > OFF_ROAD_M:
                outcome = "off_road"
                break
            hit = self._collision(env.data.qpos[:2])
            if hit is not None:
                outcome = "collision"
                break
            if np.linalg.norm(env.data.qpos[:2] - path[-1]) < PAD_R:
                outcome = "delivered"
                break
        ridden = float(np.sum(np.linalg.norm(np.diff(path[: f.i + 1], axis=0), axis=1)))
        res = {"house": house, "blind": blind, "outcome": outcome, "time_s": round(env.data.time, 2),
               "distance_m": round(ridden, 1), "replans": replans, "looks": looks, "wrong_looks": wrong_looks,
               "hit": None if hit is None else hit.kind,
               "mean_heading_error_deg": round(float(np.degrees(np.mean(head))), 2) if head else None,
               "mean_cross_track_m": round(float(np.mean(cross)), 3) if cross else None,
               "events": events}
        if record:
            res["frames"] = frames
        return res
