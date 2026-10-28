"""A small town for bicycle deliveries: a street grid, buildings, a depot, houses and obstacles.

Streets form a grid (``n`` x ``n`` intersections, ``block`` metres apart, 4 m wide with a dashed centre
line and pavements). Every block holds buildings; some carry a numbered delivery house with a coloured
drop-off pad on the pavement in front. Obstacles stand on some street segments:

- **blocking** (a barrier with cones across the whole street): the segment cannot be passed;
- **partial** (a parked car or boxes on one lane): passable by keeping to the other lane.

The rider is given the street map (``Town.graph``: intersections and segments) but **not** the obstacles:
their positions are hidden state it must discover with its own camera. ``Town.truth`` holds them for
scoring only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

ROAD_HALF = 2.0
LANE = 1.0                                   # lane centre offset from the street centre line


@dataclass
class Obstacle:
    edge: tuple[int, int]
    kind: str                                # "blocking" or "partial"
    pos: tuple[float, float]                 # world centre
    lane: int = 0                            # partial: +1 / -1, the lane it occupies (relative to edge a->b)


@dataclass
class Town:
    n: int = 4
    block: float = 25.0
    seed: int = 0
    nodes: dict = field(default_factory=dict)           # id -> (x, y)
    edges: list = field(default_factory=list)           # (a, b) with a < b
    houses: dict = field(default_factory=dict)          # number -> (edge, pad position, side)
    depot: tuple = (0, (0.0, 0.0))                      # node id, position
    obstacles: list = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        for i in range(self.n):
            for j in range(self.n):
                self.nodes[i * self.n + j] = (j * self.block, i * self.block)
        for i in range(self.n):
            for j in range(self.n):
                a = i * self.n + j
                if j + 1 < self.n:
                    self.edges.append((a, a + 1))
                if i + 1 < self.n:
                    self.edges.append((a, a + self.n))
        self.depot = (0, self.nodes[0])
        # numbered houses on random segments (pads on the pavement, one side)
        picks = rng.choice(len(self.edges), size=min(8, len(self.edges)), replace=False)
        for k, e in enumerate(picks, start=1):
            a, b = self.edges[int(e)]
            pa, pb = np.array(self.nodes[a]), np.array(self.nodes[b])
            t = float(rng.uniform(0.35, 0.65))
            d = (pb - pa) / np.linalg.norm(pb - pa)
            side = int(rng.choice([-1, 1]))
            normal = np.array([-d[1], d[0]]) * side
            pad = pa + t * (pb - pa) + normal * (ROAD_HALF - 0.6)
            self.houses[k] = (self.edges[int(e)], tuple(pad), side, t)
        # obstacles: blocking on ~20% of segments, partial on ~25%, none next to the depot
        for e in self.edges:
            if 0 in e:
                continue
            u = rng.random()
            a, b = e
            pa, pb = np.array(self.nodes[a]), np.array(self.nodes[b])
            t = float(rng.uniform(0.3, 0.7))
            pos = tuple(pa + t * (pb - pa))
            if u < 0.2:
                self.obstacles.append(Obstacle(e, "blocking", pos))
            elif u < 0.45:
                self.obstacles.append(Obstacle(e, "partial", pos, int(rng.choice([-1, 1]))))
        self._rng = rng

    @property
    def graph(self) -> dict[int, list[int]]:
        g = {k: [] for k in self.nodes}
        for a, b in self.edges:
            g[a].append(b)
            g[b].append(a)
        return g

    @property
    def truth(self) -> dict:
        """Evaluation only: which segments are blocked."""
        return {"blocked": {o.edge for o in self.obstacles if o.kind == "blocking"},
                "partial": {o.edge for o in self.obstacles if o.kind == "partial"}}

    # ------------------------------------------------------------------ the MuJoCo scene
    def build(self, spec: mujoco.MjSpec, rng=None) -> None:
        rng = self._rng if rng is None else rng
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.45, 0.65, 0.92],
                         rgb2=[0.92, 0.95, 1.0], width=512, height=3072)
        for name, c1, c2, mk in (("grass", [0.27, 0.45, 0.18], [0.2, 0.36, 0.13], [0.36, 0.55, 0.22]),
                                 ("asphalt", [0.22, 0.22, 0.23], [0.2, 0.2, 0.2], [0.33, 0.33, 0.34]),
                                 ("paving", [0.62, 0.6, 0.57], [0.55, 0.53, 0.5], [0.7, 0.68, 0.64])):
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                             rgb1=c1, rgb2=c2, mark=mujoco.mjtMark.mjMARK_RANDOM, random=0.3, markrgb=mk,
                             width=512, height=512)
        for name, rep in (("grass", [80, 80]), ("asphalt", [8, 8]), ("paving", [8, 8])):
            mat = spec.add_material(name=name, texrepeat=rep, reflectance=0.0)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        wb = spec.worldbody
        wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, dir=[-0.4, 0.5, -1.0],
                     diffuse=[0.85, 0.82, 0.75], specular=[0.2, 0.2, 0.2], castshadow=True)
        size = (self.n - 1) * self.block
        wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[size + 60, size + 60, 0.1],
                    pos=[size / 2, size / 2, 0], material="grass", contype=2, conaffinity=1,
                    friction=[1.2, 0.005, 0.001], condim=4)
        vis = dict(contype=0, conaffinity=0)
        # streets: asphalt strips, pavements, dashed centre lines
        for a, b in self.edges:
            pa, pb = np.array(self.nodes[a]), np.array(self.nodes[b])
            mid, d = (pa + pb) / 2, (pb - pa) / np.linalg.norm(pb - pa)
            yaw = float(np.arctan2(d[1], d[0]))
            quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            half = self.block / 2 + ROAD_HALF
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[mid[0], mid[1], 0.001], quat=quat,
                        size=[half, ROAD_HALF, 0.001], material="asphalt", **vis)
            normal = np.array([-d[1], d[0]])
            for side in (-1, 1):
                pv = mid + normal * side * (ROAD_HALF + 0.75)
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[pv[0], pv[1], 0.06], quat=quat,
                            size=[self.block / 2 - ROAD_HALF - 0.75, 0.75, 0.06], material="paving", **vis)
            for t in np.arange(0.15, 0.86, 0.1):
                q = pa + t * (pb - pa)
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[q[0], q[1], 0.0025], quat=quat,
                            size=[0.7, 0.05, 0.001], rgba=[0.95, 0.95, 0.9, 1], **vis)
        # buildings on every block
        palette = [[0.78, 0.72, 0.62, 1], [0.62, 0.66, 0.72, 1], [0.8, 0.6, 0.5, 1], [0.7, 0.72, 0.66, 1],
                   [0.86, 0.82, 0.7, 1]]
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                cx, cy = (j + 0.5) * self.block, (i + 0.5) * self.block
                inner = self.block / 2 - ROAD_HALF - 1.8
                for dx in (-1, 1):
                    for dy in (-1, 1):
                        h = float(rng.uniform(4, 11))
                        w = inner / 2 - 0.4
                        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                    pos=[cx + dx * inner / 2, cy + dy * inner / 2, h / 2], size=[w, w, h / 2],
                                    rgba=palette[int(rng.integers(len(palette)))], **vis)
        # depot (blue pad) and delivery pads (orange)
        dx, dy = self.depot[1]
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[dx, dy, 0.004], size=[1.2, 0.003, 0],
                    rgba=[0.15, 0.35, 0.9, 1], **vis)
        for k, (e, pad, side, t) in self.houses.items():
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[pad[0], pad[1], 0.004], size=[0.5, 0.003, 0],
                        rgba=[0.95, 0.55, 0.1, 1], name=f"house_{k}", **vis)
            a, b = e
            d = np.subtract(self.nodes[b], self.nodes[a]) / self.block
            post = np.array(pad) + np.array([-d[1], d[0]]) * side * 1.3      # on the pavement
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[post[0], post[1], 1.0], size=[0.05, 0.05, 1.0],
                        rgba=[0.95, 0.55, 0.1, 1], **vis)
        # obstacles (visual; the rider's camera sees them, contacts are checked geometrically)
        for k, o in enumerate(self.obstacles):
            a, b = o.edge
            pa, pb = np.array(self.nodes[a]), np.array(self.nodes[b])
            d = (pb - pa) / np.linalg.norm(pb - pa)
            yaw = float(np.arctan2(d[1], d[0]))
            quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            normal = np.array([-d[1], d[0]])
            c = np.array(o.pos)
            if o.kind == "blocking":
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[c[0], c[1], 0.5], quat=quat,
                            size=[0.12, ROAD_HALF, 0.5], rgba=[0.9, 0.9, 0.9, 1], name=f"obstacle_{k}", **vis)
                for s in np.linspace(-1.6, 1.6, 5):
                    q = c + normal * s - d * 0.6
                    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[q[0], q[1], 0.3],
                                size=[0.12, 0.3, 0], rgba=[0.95, 0.4, 0.05, 1], **vis)
            else:
                q = c + normal * o.lane * LANE
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[q[0], q[1], 0.7], quat=quat,
                            size=[2.0, 0.85, 0.7], rgba=[0.2, 0.3, 0.6, 1], name=f"obstacle_{k}", **vis)


def town_scene(town: Town):
    """A scenery callable for ``pai/bike/real_bike.build_spec``."""
    return lambda spec, rng: town.build(spec, rng)
