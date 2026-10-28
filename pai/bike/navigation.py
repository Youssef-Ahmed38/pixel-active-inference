"""Delivering on the bicycle: a street-map belief, the head camera, and replanning.

The rider knows the street layout (a map app) but not the obstacles. Per street it keeps a belief
P(blocked), starting from the rate it met in its practice towns. Approaching an intersection it turns
its head toward the street it plans to take and looks (head camera, colour and depth): depth points
standing above the road inside that street's corridor are an obstacle, blocking if they span the street
and partial if they cover one lane. A clear view lowers P(blocked). The route minimises expected cost
(length plus P(blocked) x a detour penalty) and is replanned whenever the belief changes, before the
rider commits at the intersection. Partial obstacles are passed in the other lane.

Given, not learned: the street map, the corridor geometry used to read depth points, the detour
penalty. Learned: the steering law (``pai/bike/riding.py``, by practice) and the blocked-street prior
(counted in practice towns).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from pai.bike.town import LANE, ROAD_HALF, Town

DETOUR = 60.0          # expected extra metres if a planned street turns out blocked
U_TURN_RISK_M = 150.0  # the cost (in metres) of being forced into a late U-turn at a dead end


@dataclass
class StreetBelief:
    prior: float
    p: dict = field(default_factory=dict)          # (a, b) sorted -> P(blocked)
    cars: dict = field(default_factory=dict)       # (a, b) -> world (x, y) of a one-lane obstacle
    barriers: dict = field(default_factory=dict)   # (a, b) -> world (x, y) of a barrier
    looked: set = field(default_factory=set)

    def get(self, e):
        return self.p.get(tuple(sorted(e)), self.prior)

    def update(self, e, blocked: bool, where=None):
        e = tuple(sorted(e))
        self.looked.add(e)
        # a look down a street is strong evidence either way (the camera check: 48/48 streets)
        self.p[e] = 0.97 if blocked else 0.03
        if where is not None:
            (self.barriers if blocked else self.cars)[e] = where


def dead_end_risk(town: Town, belief: StreetBelief, came_from, v) -> float:
    """P(every way on from intersection ``v`` is blocked) for a rider arriving from ``came_from``:
    then the only way on is a U-turn, decided late (a side street is seen only 4 m before)."""
    return float(np.prod([belief.get((v, x)) for x in town.graph[v] if x != came_from]))


def step_cost(town: Town, belief: StreetBelief, u, v, goal) -> float:
    length = float(np.linalg.norm(np.subtract(town.nodes[u], town.nodes[v])))
    c = length + belief.get((u, v)) * DETOUR
    if v != goal:
        c += dead_end_risk(town, belief, u, v) * U_TURN_RISK_M
    return c


def plan(town: Town, belief: StreetBelief, start: int, goal: int, banned=()) -> list[int] | None:
    """Expected-cost shortest route over the street graph: length, the expected detour if a street
    turns out blocked, and the expected cost of being forced into a U-turn at a dead end."""
    g = town.graph
    dist, prev = {start: 0.0}, {}
    pq = [(0.0, start)]
    while pq:
        c, u = heapq.heappop(pq)
        if u == goal:
            break
        if c > dist.get(u, math.inf):
            continue
        for v in g[u]:
            e = tuple(sorted((u, v)))
            if e in banned or belief.get(e) > 0.9:
                continue
            nc = c + step_cost(town, belief, u, v, goal)
            if nc < dist.get(v, math.inf):
                dist[v], prev[v] = nc, u
                heapq.heappush(pq, (nc, v))
    if goal not in dist:
        return None
    route = [goal]
    while route[-1] != start:
        route.append(prev[route[-1]])
    return route[::-1]


class HeadCamera:
    """The G1's head camera: colour and depth along a chosen heading, from the head's position."""

    def __init__(self, model, width=160, height=120):
        self.model = model
        self.data = mujoco.MjData(model)
        self.rgb = mujoco.Renderer(model, height, width)
        self.depth = mujoco.Renderer(model, height, width)
        self.depth.enable_depth_rendering()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.head = model.body("g1_torso_link").id

    def close(self):
        self.rgb.close()
        self.depth.close()

    def look(self, data, yaw: float):
        self.data.qpos[:] = data.qpos
        mujoco.mj_forward(self.model, self.data)
        eye = self.data.xpos[self.head] + np.array([0, 0, 0.45])
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        self.cam.lookat[:] = eye + fwd
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.0, math.degrees(yaw), -8.0
        self.rgb.update_scene(self.data, self.cam)
        img = self.rgb.render().copy()
        self.depth.update_scene(self.data, self.cam)
        z = self.depth.render().copy()
        gl = self.depth.scene.camera[0]
        cpos, cfwd, cup = np.array(gl.pos), np.array(gl.forward), np.array(gl.up)
        cright = np.cross(cfwd, cup)
        h, w = z.shape
        half_w = gl.frustum_width if gl.frustum_width > 0 else gl.frustum_top * w / h
        xs = gl.frustum_center + (np.arange(w) + 0.5) / w * 2 * half_w - half_w
        ys = gl.frustum_top - (np.arange(h) + 0.5) / h * (gl.frustum_top - gl.frustum_bottom)
        X, Y = np.meshgrid(xs, ys)
        dirs = (cfwd[None, None] * gl.frustum_near + cright[None, None] * X[..., None]
                + cup[None, None] * Y[..., None]) / gl.frustum_near
        pts = cpos[None, None] + z[..., None] * dirs
        return img, pts, z


def inspect_street(town: Town, cam: HeadCamera, data, a: int, b: int, max_range=26.0):
    """Look from near intersection ``a`` down street a->b.

    Returns (blocked, where, image, points): ``where`` is the world (x, y) of the obstacle seen (a
    barrier when ``blocked``, else a one-lane obstacle) or None; ``points`` counts the depth points read
    as standing in the street.
    """
    pa, pb = np.array(town.nodes[a], float), np.array(town.nodes[b], float)
    d = (pb - pa) / np.linalg.norm(pb - pa)
    left = np.array([-d[1], d[0]])
    here = data.qpos[:2]
    img, pts, z = cam.look(data, math.atan2(*(pb - here)[::-1]))
    ok = (z > 0.1) & (z < max_range)
    p = pts[ok]
    rel = p[:, :2] - pa
    along, lateral = rel @ d, rel @ left
    inside = (along > 3.0) & (along < np.linalg.norm(pb - pa) - 3.0) & (np.abs(lateral) < ROAD_HALF - 0.1)
    high = inside & (p[:, 2] > 0.2) & (p[:, 2] < 2.0)
    n = int(high.sum())
    if n < 15:
        return False, None, img, n
    lat = lateral[high]
    blocked = bool(np.quantile(lat, 0.95) - np.quantile(lat, 0.05) > 2.3)
    # a one-lane obstacle lies wholly on one side of the centre line; seen at an angle mostly its near
    # side shows, so its lane is taken from the side the points are on and it is placed at that lane
    side = float(np.median(lat)) if blocked else float(np.sign(np.median(lat))) * LANE
    # along the street the near end shows; a parked car is taken to be about 4 m long
    centre = float(np.median(along[high])) if blocked else float(np.quantile(along[high], 0.05)) + 2.0
    where = tuple(pa + centre * d + side * left)
    return blocked, where, img, n


def avoid_cars(path: np.ndarray, cars, ramp: float = 3.5, span: float = 6.5) -> np.ndarray:
    """Move the lane path around known one-lane obstacles, with smooth ramps along the path.

    Per car, at the path point closest to it: a car within a lane width of the path stands in the
    rider's lane, so the path moves into the other lane; a car in the other lane gets 0.5 m more room.
    """
    path = path.copy()
    seg = np.gradient(path, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True) + 1e-9
    left = np.stack([-seg[:, 1], seg[:, 0]], 1)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for car in cars:
        k = int(np.argmin(np.linalg.norm(path - np.asarray(car), axis=1)))
        lat0 = float((np.asarray(car) - path[k]) @ left[k])
        w = np.clip((span - np.abs(s - s[k])) / ramp, 0, 1)
        if abs(lat0) < 1.2:
            path += left * (2 * LANE * w)[:, None]
        elif 1.2 <= lat0 < 3.0:
            path -= left * (0.5 * w)[:, None]
    return path
