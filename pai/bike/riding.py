"""Steering the G1's bicycle along a route: lanes, corners and a path-following balance law.

A route is a list of intersections; ``lane_path`` turns it into a polyline in the right-hand lane. The
rider's steering law (gains learned by practice, ``scripts/bike_g1_turn_practice.py``) combines
balance (steer against roll and roll rate, damp the steering angle) with path following: the target
heading points at a look-ahead point on the path (pure pursuit), and the target speed drops before a
corner. Measures follow CycleRL (Liu et al. 2026): completion, heading error, cross-track error, falls.
"""

from __future__ import annotations

import math

import numpy as np

from pai.bike.town import LANE, ROAD_HALF

# gains: roll, roll rate, heading, steer damping, speed, look-ahead (m), cruise (m/s), corner (m/s)
DEFAULT = np.array([3.2, 0.75, 0.6, 0.0, 6.8, 3.0, 3.0, 2.0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def lane_path(nodes: dict, route: list, step: float = 0.5, radius: float = 0.0) -> np.ndarray:
    """Dense polyline through the right-hand lane of each street of the route.

    With ``radius`` > 0 every corner is rounded by a circular arc of that radius (as a rider takes
    a corner); U-turns stay sharp.
    """
    corners = []
    for a, b in zip(route[:-1], route[1:]):
        pa, pb = np.array(nodes[a], float), np.array(nodes[b], float)
        d = (pb - pa) / np.linalg.norm(pb - pa)
        right = np.array([d[1], -d[0]]) * LANE
        corners.append((pa + right, pb + right))
    # the lane polyline's vertices: start, the intersections of consecutive lane lines, end
    verts = [corners[0][0]]
    for (p0, p1), (q0, q1) in zip(corners[:-1], corners[1:]):
        d1, d2 = p1 - p0, q1 - q0
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9 and d1 @ d2 < 0:    # a U-turn: loop around the intersection
            verts += u_turn(p1, q0, d1 / np.linalg.norm(d1), step)
        elif abs(den) < 1e-9:                  # straight on: join the lane ends
            verts += [p1] if np.allclose(p1, q0) else [p1, q0]
        else:
            t = ((q0 - p0)[0] * d2[1] - (q0 - p0)[1] * d2[0]) / den
            verts.append(p0 + t * d1)
    verts.append(corners[-1][1])
    pts = [verts[0]]
    for i in range(1, len(verts)):
        a, b = pts[-1], verts[i]
        if i < len(verts) - 1 and radius > 0:
            c = verts[i + 1]
            u1, u2 = (b - a) / (np.linalg.norm(b - a) + 1e-9), (c - b) / (np.linalg.norm(c - b) + 1e-9)
            theta = math.acos(float(np.clip(u1 @ u2, -1, 1)))
            T = radius * math.tan(theta / 2) if theta < 2.5 else 0.0
            T = min(T, 0.45 * np.linalg.norm(b - a), 0.45 * np.linalg.norm(c - b))
            if T > 1e-3:
                s0, s1 = b - u1 * T, b + u2 * T
                n = max(2, int(np.linalg.norm(s0 - a) / step))
                pts += list(a + np.outer(np.linspace(0, 1, n, endpoint=False)[1:], s0 - a)) + [s0]
                # circular arc from s0 to s1 (radius R about a centre on the inside of the turn)
                left = u1[0] * u2[1] - u1[1] * u2[0] > 0
                n1 = np.array([-u1[1], u1[0]]) * (1 if left else -1)
                R = T / math.tan(theta / 2)
                c0 = s0 + n1 * R
                a0 = math.atan2(*(s0 - c0)[::-1])
                m = max(3, int(R * theta / step))
                for w in np.linspace(0, 1, m)[1:]:
                    ang = a0 + (theta if left else -theta) * w
                    pts.append(c0 + R * np.array([math.cos(ang), math.sin(ang)]))
                continue
        n = max(2, int(np.linalg.norm(b - a) / step))
        pts += list(a + np.outer(np.linspace(0, 1, n + 1)[1:], b - a))
    return np.asarray(pts)


U_TURN_R = 2.4      # a circle about the intersection's centre: stays on the asphalt (kerb corners 2.8 m)


def u_turn(p_in, p_out, u, step=0.5, ramp=6.0):
    """Vertices of a U-turn at an intersection: from the end of the incoming lane ``p_in`` (heading
    ``u``) swing out to a circle of radius ``U_TURN_R`` about the intersection, go round it to the left
    by 180 degrees and come back in to the outgoing lane ``p_out``."""
    left = np.array([-u[1], u[0]])
    c = (p_in + p_out) / 2
    pts = [p_in - u * ramp]
    th0 = math.atan2(*(-left)[::-1])
    m = max(6, int(math.pi * U_TURN_R / step))
    pts += [c + U_TURN_R * np.array([math.cos(th0 + math.pi * w), math.sin(th0 + math.pi * w)])
            for w in np.linspace(0, 1, m)]
    pts.append(p_out - u * ramp)
    return pts


def curvature(path: np.ndarray) -> np.ndarray:
    """Signed curvature (1/m, positive = turning left) at every point of a polyline."""
    seg = np.diff(path, axis=0)
    h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    ds = np.linalg.norm(seg, axis=1)
    k = np.gradient(h) / np.maximum(ds, 1e-6)
    k = np.convolve(np.append(k, k[-1]), np.ones(3) / 3, mode="same")
    return np.clip(k, -1.0, 1.0)


class PathFollower:
    """Steering and speed toward a polyline; keeps its progress index along the path."""

    def __init__(self, path: np.ndarray, gains=DEFAULT):
        self.path = path
        self.g = np.asarray(gains, float)
        self.i = 0
        # heading change ahead of every path point (for corner speed)
        seg = np.diff(path, axis=0)
        h = np.arctan2(seg[:, 1], seg[:, 0])
        self.turn = np.abs(np.array([wrap(h[min(k + 6, len(h) - 1)] - h[k]) for k in range(len(h))] + [0.0]))

    def progress(self, xy: np.ndarray) -> tuple[int, float]:
        lo, hi = self.i, min(len(self.path), self.i + 40)
        k = lo + int(np.argmin(np.linalg.norm(self.path[lo:hi] - xy, axis=1)))
        self.i = k
        return k, float(np.linalg.norm(self.path[k] - xy))

    def action(self, s: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, dict]:
        k_roll, k_rate, k_head, k_steer, k_speed, look, cruise, corner = self.g
        roll, rate, speed, steer, _, yaw = s
        k, cross = self.progress(xy)
        j = min(len(self.path) - 1, k + max(1, int(look / 0.5)))
        target = self.path[j]
        want = math.atan2(target[1] - xy[1], target[0] - xy[0])
        err = wrap(want - yaw)
        ahead = self.turn[k:min(len(self.turn), k + int(2 * look / 0.5) + 1)]
        v_target = corner if ahead.size and ahead.max() > 0.5 else cruise
        torque = np.clip(k_speed * (v_target - speed), -12, 12)
        steering = np.clip(-k_roll * roll - k_rate * rate + k_head * err - k_steer * steer, -0.6, 0.6)
        info = {"progress": k / max(1, len(self.path) - 1), "cross_track": cross, "heading_error": err,
                "target_speed": v_target}
        return np.array([torque, steering, 0.0]), info

    @property
    def finished(self) -> bool:
        return self.i >= len(self.path) - 3


# lean-target gains: roll, roll rate, heading->lean, steering feedforward, speed, look-ahead (m),
# cruise (m/s), corner (m/s)
LEAN_DEFAULT = np.array([3.2, 0.75, 0.3, 1.0, 6.8, 3.0, 3.0, 2.0])
WHEELBASE, G = 0.98, 9.81
CORNER_R = 3.4          # the inside of a right turn just reaches the kerb corner


class LeanFollower(PathFollower):
    """Cornering as a rider does it: lean into the curve ahead, then steer to hold the lean.

    The path's curvature a short preview ahead sets the lean a steady turn needs (this bike rolls to
    positive angles when it turns right, so a left turn needs negative roll), plus a correction from
    the heading error to the look-ahead point. The steering drives the roll to that target (which by
    itself produces the countersteer) plus the kinematic steering for the curvature,
    ``k_ff * wheelbase * curvature``.
    """

    def __init__(self, path, gains=LEAN_DEFAULT, max_lean=0.35):
        super().__init__(path, gains)
        self.max_lean = max_lean
        self.kappa = curvature(path)
        self.turn = np.array([np.abs(self.kappa[k:k + 12]).max() * CORNER_R for k in range(len(path))])

    def action(self, s, xy):
        k_roll, k_rate, k_head, k_ff, k_speed, look, cruise, corner = self.g[:8]
        tight = self.g[8] if len(self.g) > 8 else corner       # speed in U-turns (curvature > 1/2.9 m)
        roll, rate, speed, steer, _, yaw = s
        k, cross = self.progress(xy)
        j = min(len(self.path) - 1, k + max(1, int(look / 0.5)))
        target = self.path[j]
        err = wrap(math.atan2(target[1] - xy[1], target[0] - xy[0]) - yaw)
        v = max(speed, 0.5)
        kap = float(self.kappa[min(len(self.kappa) - 1, k + max(1, int(0.4 * v / 0.5)))])
        ahead = self.turn[k:min(len(self.turn), k + int(2 * look / 0.5) + 1)]
        v_target = cruise
        if ahead.size and ahead.max() > 0.5:
            v_target = tight if ahead.max() > 1.17 else corner
        if getattr(self, "v_override", None) is not None:      # set by the traffic logic
            v_target = self.v_override
        lean_ref = float(np.clip(-(math.atan(v * v * kap / G) + k_head * err), -self.max_lean, self.max_lean))
        torque = np.clip(k_speed * (v_target - speed), -12, 12)
        steering = np.clip(k_ff * WHEELBASE * kap - k_roll * (roll - lean_ref) - k_rate * rate, -0.6, 0.6)
        info = {"progress": k / max(1, len(self.path) - 1), "cross_track": cross, "heading_error": err,
                "target_speed": v_target, "lean_ref": lean_ref, "curvature": kap}
        return np.array([torque, steering, 0.0]), info
