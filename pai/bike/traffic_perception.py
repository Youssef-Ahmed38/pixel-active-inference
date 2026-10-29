"""Seeing and avoiding moving traffic with the head camera: track, predict, choose a safe speed.

Every ``LOOK_EVERY_S`` the rider looks ahead along its own path. Depth points standing 0.3-2 m above the
road, within the road width around the path ahead and within ``RANGE_M``, that are not a parked car or
barrier it already knows about, are grouped into objects (points within ``JOIN_M`` of each other). Each
object is matched to the nearest track from the previous look (within ``MATCH_M`` of where that track
was predicted to be); the track's velocity is the smoothed change of position. Nothing about the vehicles'
routes or speeds is given: they are hidden state inferred from the camera.

The response is a choice of action by expected risk: for each candidate speed in the bicycle's stable
range (it cannot stop), the rider's own positions along its path over the next ``HORIZON_S`` are compared
with every track's extrapolated positions, and the speed with the largest predicted clearance is chosen
(the rule's own speed when every candidate keeps ``SAFE_M`` of clearance, or when the speed makes no
difference of ``MATTERS_M`` to the clearance: riding slower than practised through a turn costs balance
and buys nothing then).
"""

from __future__ import annotations

import math

import numpy as np

LOOK_EVERY_S = 0.2
RANGE_M = 25.0
JOIN_M = 1.0
MATCH_M = 2.5
HORIZON_S = 4.0
SAFE_M = 3.5
SPEEDS = (1.9, 2.3, 2.8, 3.2)
KNOWN_M = 1.6              # points this close to a known parked car or barrier belong to it (its footprint;
                           # wider would hide a vehicle passing right beside it)
MATTERS_M = 0.3           # a speed choice must change the predicted clearance by this much
MOVING_MPS = 0.5           # slower tracks are static; static obstacles are the route planner's job, except
                           # a vehicle seen moving that has stopped (it is not on the planner's map)
MOVED_M = 2.0              # a track is a vehicle once it has travelled this far from where it was first seen
                           # (a parked car's centroid wanders as more of it comes into view)


def objects_from_points(pts: np.ndarray) -> list[np.ndarray]:
    """Group 2-D points into objects (single linkage within JOIN_M); returns their centroids."""
    if len(pts) == 0:
        return []
    pts = pts[np.argsort(pts[:, 0])]
    left = list(range(len(pts)))
    groups = []
    while left:
        seed = left.pop(0)
        group, frontier = [seed], [seed]
        while frontier:
            k = frontier.pop()
            near = [j for j in left if np.linalg.norm(pts[j] - pts[k]) < JOIN_M]
            for j in near:
                left.remove(j)
            group += near
            frontier += near
        if len(group) >= 6:
            groups.append(pts[group].mean(0))
    return groups


class TrafficTracker:
    def __init__(self):
        self.tracks: list[dict] = []          # {"p": xy, "v": vxy, "t": last seen}

    def update(self, centroids: list[np.ndarray], t: float):
        new = []
        used = set()
        for c in centroids:
            best, bd = None, MATCH_M
            for k, tr in enumerate(self.tracks):
                if k in used:
                    continue
                pred = tr["p"] + tr["v"] * (t - tr["t"])
                d = float(np.linalg.norm(c - pred))
                if d < bd:
                    best, bd = k, d
            if best is None:
                new.append({"p": c, "v": np.zeros(2), "t": t, "n": 1, "moved": False, "p0": c})
            else:
                tr = self.tracks[best]
                used.add(best)
                dt = max(t - tr["t"], 1e-3)
                v_now = (c - tr["p"]) / dt
                v = v_now if tr["n"] == 1 else 0.5 * tr["v"] + 0.5 * v_now
                moved = tr["moved"] or (float(np.linalg.norm(v)) >= MOVING_MPS
                                        and float(np.linalg.norm(c - tr["p0"])) > MOVED_M)
                new.append({"p": c, "v": v, "t": t, "n": tr["n"] + 1, "moved": moved, "p0": tr["p0"]})
        # keep tracks unseen for a short while (they may be briefly hidden)
        for k, tr in enumerate(self.tracks):
            if k not in used and t - tr["t"] < 1.0:
                new.append(tr)
        self.tracks = new

    def choose_speed(self, path: np.ndarray, k: int, xy: np.ndarray, t: float, default: float):
        """Speed with the largest predicted clearance to every track; ``default`` when all are safe."""
        if not self.tracks:
            return None, None
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path[k:], axis=0), axis=1))])
        taus = np.linspace(0.0, HORIZON_S, 16)
        best, best_clear, clears = None, -1.0, {}
        for v in SPEEDS:
            s = v * taus
            idx = np.minimum(np.searchsorted(arc, s), len(arc) - 1)
            mine = path[k:][idx]
            clear = math.inf
            for tr in self.tracks:
                moving = tr["n"] >= 2 and float(np.linalg.norm(tr["v"])) >= MOVING_MPS
                if not (moving or tr["moved"]):
                    continue
                vel = tr["v"] if moving else np.zeros(2)
                theirs = tr["p"][None] + vel[None] * (taus[:, None] + (t - tr["t"]))
                clear = min(clear, float(np.min(np.linalg.norm(mine - theirs, axis=1))))
            clears[v] = clear
            if clear > best_clear:
                best, best_clear = v, clear
        if all(c >= SAFE_M for c in clears.values()):
            return None, clears
        if max(clears.values()) - min(clears.values()) < MATTERS_M:
            return None, clears
        return best, clears


def standing_points_ahead(pts: np.ndarray, z: np.ndarray, path: np.ndarray, k: int, known: list) -> np.ndarray:
    """Depth points standing in the road around the path ahead, minus those of known static obstacles."""
    ok = (z > 0.3) & (z < RANGE_M)
    p = pts[ok]
    p = p[(p[:, 2] > 0.3) & (p[:, 2] < 2.0)]
    if len(p) == 0:
        return np.zeros((0, 2))
    ahead = path[k:min(len(path), k + int(RANGE_M / 0.5))]
    if len(ahead) == 0:
        return np.zeros((0, 2))
    xy = p[:, :2]
    dmin = np.min(np.linalg.norm(xy[:, None, :] - ahead[None, ::2, :], axis=2), axis=1)
    xy = xy[dmin < 2.4]
    for q in known:
        xy = xy[np.linalg.norm(xy - np.asarray(q), axis=1) > KNOWN_M]
    return xy
