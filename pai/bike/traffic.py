"""Moving traffic in a town: vehicles driving along the streets in their right-hand lane.

Each vehicle drives along a chain of 3-4 streets, turns round at the ends of the chain (it simply
reverses onto the other lane) and drives back, forever, at its own cruising speed (2-4 m/s, about the
bicycle's). Its driver keeps a gap behind whatever is ahead in its lane, the rider included (it slows
down behind a cyclist and stops 4 m behind if needed), and it does not enter a junction the rider is
about to occupy (it waits short of the junction while the rider is within ``JUNCTION_M`` of it), as real
drivers do; the rider cannot see behind itself, so nothing else would stop a faster vehicle from running
into it, and a vehicle already in the junction keeps going: avoiding it is the rider's job. The vehicles are stepped
with the simulation and are deterministic given the ride, so every ride is reproducible. They are
hidden state for the rider: it is not told where vehicles are or where they go. In the MuJoCo
scene each vehicle is a moving (mocap) car body that the head camera sees; contact with the rider is
checked geometrically, like the parked cars.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from pai.bike.riding import CORNER_R, lane_path
from pai.bike.town import Town

CAR_HALF = (2.0, 0.85, 0.7)          # half length, half width, half height (m), as the parked cars
CAR_RGBA = [0.75, 0.15, 0.12, 1.0]


@dataclass
class Vehicle:
    path: np.ndarray                 # closed loop: there along the right lane, back along the other
    arc: np.ndarray
    speed: float                     # cruising speed
    phase: float                     # metres along the loop at the start
    s: float = 0.0                   # current position along the loop (m)

    def at(self, t: float = 0.0) -> tuple[np.ndarray, float]:
        """Position and heading now (``t`` is kept for the interface; the state is ``s``)."""
        s = self.s % self.arc[-1]
        k = int(np.searchsorted(self.arc, s, side="right")) - 1
        k = min(max(k, 0), len(self.path) - 2)
        w = (s - self.arc[k]) / max(self.arc[k + 1] - self.arc[k], 1e-9)
        xy = self.path[k] + w * (self.path[k + 1] - self.path[k])
        d = self.path[k + 1] - self.path[k]
        return xy, float(np.arctan2(d[1], d[0]))


class Traffic:
    def __init__(self, town: Town, n: int = 6, seed: int = 0, avoid_depot: bool = True):
        rng = np.random.default_rng(seed + 424242)
        g = town.graph
        self.junctions = np.array([town.nodes[k] for k in sorted(town.nodes)], dtype=float)
        self.vehicles: list[Vehicle] = []
        while len(self.vehicles) < n:
            start = int(rng.choice([k for k in town.nodes if not (avoid_depot and k == town.depot[0])]))
            chain = [start]
            for _ in range(int(rng.integers(3, 5))):
                nxt = [x for x in g[chain[-1]] if x not in chain]
                if not nxt:
                    break
                chain.append(int(rng.choice(nxt)))
            if len(chain) < 3:
                continue
            there = lane_path(town.nodes, chain, radius=CORNER_R)
            back = lane_path(town.nodes, chain[::-1], radius=CORNER_R)
            loop = np.vstack([there, back, there[:1]])
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
            ph = float(rng.uniform(0, arc[-1]))
            self.vehicles.append(Vehicle(loop, arc, float(rng.uniform(2.0, 4.0)), ph, ph))

    GAP_M = 4.0           # a driver stops this far behind what is ahead in its lane
    LOOK_M = 12.0         # and starts slowing when it is this close
    JUNCTION_M = 8.0      # a driver waits short of a junction while the rider is this close to its centre
    WAIT_M = 5.0          # ... with its centre this far from the junction centre (front 1 m before the road edge)

    def reset(self):
        for v in self.vehicles:
            v.s = v.phase

    def step(self, dt: float, rider_xy) -> None:
        """Advance every vehicle by ``dt``; each driver slows for the rider ahead in its lane."""
        for v in self.vehicles:
            c, yaw = v.at()
            d = np.array([np.cos(yaw), np.sin(yaw)])
            rel = np.asarray(rider_xy) - c
            ahead, side = float(rel @ d), float(rel @ np.array([-d[1], d[0]]))
            speed = v.speed
            if 0.0 < ahead < self.LOOK_M and abs(side) < 1.6:
                speed = min(speed, max(0.0, v.speed * (ahead - self.GAP_M) / (self.LOOK_M - self.GAP_M)))
            rj = self.junctions - c
            along, across = rj @ d, rj @ np.array([-d[1], d[0]])
            near = (along > self.WAIT_M - 0.5) & (along < self.LOOK_M) & (np.abs(across) < 2.5)
            if near.any():
                j = int(np.argmin(np.where(near, along, np.inf)))
                if np.linalg.norm(np.asarray(rider_xy) - self.junctions[j]) < self.JUNCTION_M:
                    speed = min(speed, max(0.0, v.speed * (along[j] - self.WAIT_M) / (self.LOOK_M - self.WAIT_M)))
            v.s += speed * dt

    def states(self, t: float):
        return [v.at(t) for v in self.vehicles]

    def hit(self, xy: np.ndarray, t: float, margin: float = 0.3) -> int | None:
        """Index of a vehicle whose footprint (plus margin) contains the rider at time t, or None."""
        for k, (c, yaw) in enumerate(self.states(t)):
            d = np.array([np.cos(yaw), np.sin(yaw)])
            rel = np.asarray(xy) - c
            if abs(rel @ d) < CAR_HALF[0] + margin and abs(rel @ np.array([-d[1], d[0]])) < CAR_HALF[1] + margin:
                return k
        return None

    # ------------------------------------------------------------------ the MuJoCo scene
    def build(self, spec: mujoco.MjSpec) -> None:
        for k in range(len(self.vehicles)):
            b = spec.worldbody.add_body(name=f"vehicle_{k}", mocap=True, pos=[0, 0, CAR_HALF[2]])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(CAR_HALF), rgba=CAR_RGBA, contype=0, conaffinity=0)

    def place(self, model, data, t: float) -> None:
        """Move the vehicle bodies to their positions at time t (for the camera and videos)."""
        for k, (c, yaw) in enumerate(self.states(t)):
            mid = model.body_mocapid[model.body(f"vehicle_{k}").id]
            data.mocap_pos[mid] = [c[0], c[1], CAR_HALF[2]]
            data.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def town_traffic_scene(town: Town, traffic: Traffic):
    """A scenery callable for ``pai/bike/real_bike.build_spec``: the town and its vehicles."""
    def build(spec, rng):
        town.build(spec, rng)
        traffic.build(spec)
    return build
