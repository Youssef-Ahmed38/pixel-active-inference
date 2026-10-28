"""The maze agent sees through the G1's own head camera (Part C, step 1).

``EgoCameraMaze`` wraps a ``WalkerMaze`` and replaces the grid sensor: after every move the robot
looks around with a head-height camera (four renders, 90 degrees apart, colour and depth), turns the
depth pixels into 3-D points and sorts them into maze cells. A cell with points well above the floor
reads ``wall``, a cell seen only at floor level reads ``floor``, a cell whose floor pixels have the
goal disc's gold colour reads ``goal``. Cells further than ``max_range_m`` are not reported. The
agent receives the same ``Observation`` as from the grid sensor, so the Bayesian map, the sensor
model learned from its own looks and body labels (``pai/maze/self_calibration.py``), the learned
repeat-look weight and touch all apply unchanged; the camera's errors (edges, grazing views, the
robot's own body in view) are what that machinery has to absorb.

What is still given in this step: the cell size and wall height used to sort points, the goal's
colour, and the camera's pose, which comes from the simulated body (step 2 replaces it with the
robot's own odometry).
"""

from __future__ import annotations

import mujoco
import numpy as np

from pai.locomotion.body import STATE_SPEC
from pai.maze.env import Cell, Observation
from pai.maze.noisy import TouchObservation
from pai.maze.walker import WalkerMaze, cell_xy



class EgoCameraMaze:
    """Grid-maze interface whose observations come from the walker's head camera."""

    def __init__(self, walker: WalkerMaze, width: int = 160, height: int = 120,
                 eye_height: float = 1.2, max_range_m: float = 3.5, wall_z: float = 0.25,
                 floor_z: float = 0.06, min_points: int = 4):
        self.walker = walker
        self.world = walker.world
        m = walker.env.models[0]
        self.model = m
        self.data = mujoco.MjData(m)
        self.renderer = mujoco.Renderer(m, height, width)
        self.depth = mujoco.Renderer(m, height, width)
        self.depth.enable_depth_rendering()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.eye_height, self.max_range = eye_height, max_range_m
        self.wall_z, self.floor_z, self.min_points = wall_z, floor_z, min_points
        self.pelvis = m.body("pelvis").id
        self.last_views: list[np.ndarray] = []      # the four colour images of the last look (videos)
        self.map_ref = None        # the agent's own map, for localization (set by the experiment)
        self.localize = True

    def close(self):
        """Free the two OpenGL renderers; call before building the next one in the same process
        (renderers left open break later ones on this machine)."""
        for r in (self.renderer, self.depth):
            r.close()

    def __getattr__(self, name):
        if name in ("walker", "world"):
            raise AttributeError(name)
        return getattr(self.walker, name)

    # ------------------------------------------------------------------ seeing
    def _look(self, yaw: float):
        """One head-camera view along ``yaw``: colour image and world points of every pixel."""
        d = self.data
        pos = d.xpos[self.pelvis]
        eye = np.array([pos[0], pos[1], self.eye_height])
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.cam.lookat[:] = eye + fwd
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.0, np.degrees(yaw), -20.0
        self.renderer.update_scene(d, self.cam)
        rgb = self.renderer.render().copy()
        self.depth.update_scene(d, self.cam)
        z = self.depth.render().copy()                      # metres along the viewing axis
        gl = self.depth.scene.camera[0]
        cpos, cfwd, cup = (np.array(gl.pos), np.array(gl.forward), np.array(gl.up))
        cright = np.cross(cfwd, cup)
        h, w = z.shape
        # pixel grid on the near plane (OpenGL frustum), rows top to bottom
        # frustum_width 0 means "from the aspect ratio" (MuJoCo's convention)
        half_w = gl.frustum_width if gl.frustum_width > 0 else gl.frustum_top * w / h
        xs = gl.frustum_center + (np.arange(w) + 0.5) / w * 2 * half_w - half_w
        ys = gl.frustum_top - (np.arange(h) + 0.5) / h * (gl.frustum_top - gl.frustum_bottom)
        X, Y = np.meshgrid(xs, ys)
        dirs = (cfwd[None, None] * gl.frustum_near + cright[None, None] * X[..., None]
                + cup[None, None] * Y[..., None])
        dirs /= gl.frustum_near                              # unit length along the viewing axis
        pts = cpos[None, None] + z[..., None] * dirs
        rays = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
        return rgb, pts, z, rays

    def _gather(self):
        """Look around (four views); return the points relative to the body, in a frame aligned
        with the body's true heading, with their colours."""
        mujoco.mj_setState(self.model, self.data, self.walker.env.state[0], STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)
        yaw0 = float(self.walker.env.yaw()[0])
        here = np.array(self.data.xpos[self.pelvis][:2])
        rel, cols = [], []
        self.last_views = []
        for k in range(4):
            rgb, pts, z, _ = self._look(yaw0 + k * np.pi / 2)
            self.last_views.append(rgb)
            dist = np.linalg.norm(pts[..., :2] - here, axis=-1)
            # the robot's own body (within 0.35 m of its pelvis) is not part of the maze
            ok = (z > 0.05) & (dist < self.max_range) & (dist > 0.35)
            q = pts[ok].copy()
            q[:, :2] -= here
            rel.append(q)
            cols.append(rgb[ok].astype(float))
        return np.concatenate(rel), np.concatenate(cols), yaw0

    def _place(self, rel, est_xy, dyaw):
        """World points under a believed pose: body-relative points rotated by the heading error."""
        rot = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
        p = rel.copy()
        p[:, :2] = est_xy + rel[:, :2] @ rot.T
        return p

    def _cells(self, p, est_xy):
        """Continuous points to cells; wall faces on a boundary go to the far cell, ambiguous top
        edges are dropped. Returns (r, c, keep mask)."""
        start, cell = self.world.start, self.walker.cell
        v = start[0] - p[:, 1] / cell
        u = start[1] + p[:, 0] / cell
        v0, u0 = start[0] - est_xy[1] / cell, start[1] + est_xy[0] / cell
        # a surface seen on (or within 8 cm of) the boundary between two cells belongs to the cell
        # farther from the viewer, who stands in free space: a wall's face is not the open cell
        # in front of it
        tall = p[:, 2] > self.wall_z
        top = p[:, 2] > np.quantile(p[tall, 2], 0.98) - 0.03 if tall.any() else tall
        keep = np.ones(len(p), bool)
        for coord, own in ((v, v0), (u, u0)):
            frac = coord - np.floor(coord)
            edge = tall & (np.abs(frac - 0.5) < 0.08 / cell)
            face = edge & ~top
            coord[face] = np.floor(coord[face]) + 0.5 + 0.1 * np.sign(coord[face] - own)
            keep &= ~(edge & top)        # a top surface's edge is ambiguous: drop it
        r, c = np.rint(v).astype(int), np.rint(u).astype(int)
        n = self.world.size
        keep &= (r >= 0) & (r < n) & (c >= 0) & (c < n)
        return r, c, keep

    def _localize(self, rel, est_xy, est_dyaw):
        """Pose as a hidden state: the correction (dx, dy, dyaw) that best explains this view under
        the agent's own map, with a Gaussian odometry prior. Needs ``map_ref`` (the agent's map)."""
        m = self.map_ref
        if m is None or len(m.observed_cells) < 15:
            return np.zeros(2), 0.0, None
        rng = np.random.default_rng(0)
        sub = rel[rng.choice(len(rel), min(len(rel), 3000), replace=False)]
        tall = sub[:, 2] > self.wall_z
        low = sub[:, 2] < self.floor_z
        use = tall | low
        sub, kind = sub[use], np.where(tall[use], 0, 1)      # 0 wall, 1 floor
        logb = np.log(np.clip(m.beliefs, 1e-4, 1.0))
        best = None
        for dx in np.arange(-0.3, 0.31, 0.1):
            for dy in np.arange(-0.3, 0.31, 0.1):
                for dth in (-0.1, -0.05, 0.0, 0.05, 0.1):
                    xy = est_xy + np.array([dx, dy])
                    p = self._place(sub, xy, est_dyaw + dth)
                    r, c, keep = self._cells(p, xy)
                    if not keep.any():
                        continue
                    ll = logb[r[keep], c[keep], kind[keep]].sum()
                    prior = -0.5 * ((dx * dx + dy * dy) / 0.15 ** 2 + (dth / 0.05) ** 2)
                    score = ll + prior
                    if best is None or score > best[0]:
                        best = (score, np.array([dx, dy]), dth)
        if best is None:
            return np.zeros(2), 0.0, None
        return best[1], best[2], float(best[0])

    def _sense(self) -> dict[Cell, str]:
        rel, colours, yaw0 = self._gather()
        # where the robot believes it is (own odometry, Part C step 2): points are seen relative to
        # the body and placed with the believed pose, so drift becomes map error unless corrected
        est_xy, dyaw = self.walker.xy(), self.walker.yaw() - yaw0
        if self.walker.odometry and self.localize:
            dxy, dth, _ = self._localize(rel, est_xy, dyaw)
            if dth or dxy.any():
                self.walker.est_xy = self.walker.est_xy + dxy
                self.walker._yaw_err += dth
                self.walker.est_yaw += dth
                est_xy, dyaw = self.walker.xy(), self.walker.yaw() - yaw0
        p = self._place(rel, est_xy, dyaw)
        r, c, keep = self._cells(p, est_xy)
        p, colours, r, c = p[keep], colours[keep], r[keep], c[keep]
        tall = p[:, 2] > self.wall_z
        low = p[:, 2] < self.floor_z
        # gold by chromaticity, not brightness (the rendered disc is dark gold, e.g. 75/58/11): its
        # blue share is ~8%, the floor's and walls' ~30-37%
        total = colours.sum(axis=1)
        gold = low & (total > 60) & (colours[:, 2] < 0.18 * total) & (colours[:, 0] > colours[:, 1])
        wall_n, floor_n, goal_n = {}, {}, {}
        for mask, store in ((tall, wall_n), (low & ~gold, floor_n), (gold, goal_n)):
            for key in zip(r[mask].tolist(), c[mask].tolist()):
                store[key] = store.get(key, 0) + 1
        visible: dict[Cell, str] = {}
        for key in set(wall_n) | set(floor_n) | set(goal_n):
            w, f, g = wall_n.get(key, 0), floor_n.get(key, 0), goal_n.get(key, 0)
            if max(w, f, g) < self.min_points:
                continue
            visible[key] = "wall" if w >= max(f, g) else ("goal" if g > f / 4 else "floor")
        return visible

    # ------------------------------------------------------------------ maze interface
    def observe(self, collided: bool = False) -> Observation:
        return TouchObservation(self.walker.est_cell(), self._sense(), collided,
                                self.world.position == self.world.goal, None)

    def step(self, direction: Cell) -> Observation:
        start = self.walker.est_cell()                       # the agent's own idea of where it is
        grid_obs = self.walker.step(direction)
        bumped = (start[0] + direction[0], start[1] + direction[1]) if grid_obs.collided else None
        visible = self._sense()
        if bumped is not None:
            visible[bumped] = "wall"                         # touch, not a look
        # standing on the goal disc is felt/seen by the robot itself
        return TouchObservation(self.walker.est_cell(), visible, grid_obs.collided, grid_obs.reached_goal,
                                bumped)
