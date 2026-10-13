"""From pixels to entity tokens (week 3): the drop-in replacement for privileged object state.

    camera frame -> frozen DINOv2 patch features -> object slots -> per-slot token readout
                 -> each known object is read from its own slot (slots trained with a fixed slot per
                    object), or matched to the slot whose predicted kind and colour fit best

Only the objects come from vision. The gripper token (position, opening, wrist force) comes from
proprioception and touch: a body senses itself directly. Object velocities are finite differences
of the perceived positions. With a fixed slot per object, identity is the slot index; with
interchangeable slots, each object is matched by its predicted kind and colour (and a checkpoint
trained for it can start each frame's slot attention from the previous frame's slots).

The output has exactly the layout of `pai.world.entities.encode`, so the world model, goals, planner
and cause inference run unchanged on perceived instead of privileged state. The ablation
"privileged vs perceived" is then a single switch.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pai.perception.features import DinoFeatures
from pai.perception.slots import SlotModel
from pai.world.entities import COLOR, FORCE, GRIP, KIND, KINDS, POS, TOKEN_DIM, VEL, YAW, encode


# ---------------------------------------------------------------------------------------------
# Pixel geometry: where each object is, from its own pixels and the calibrated camera
# ---------------------------------------------------------------------------------------------

def _chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb (..., 3) in [0, 1] -> saturation, value, and the chroma plane (sat cos hue, sat sin hue)."""
    mx, mn = rgb.max(-1), rgb.min(-1)
    c = mx - mn
    sat = c / (mx + 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cc = c + 1e-6
    hue = np.where(mx == r, np.mod((g - b) / cc, 6.0), np.where(mx == g, (b - r) / cc + 2, (r - g) / cc + 4))
    hue = hue * (np.pi / 3)
    return sat, mx, np.stack([sat * np.cos(hue), sat * np.sin(hue)], -1)


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (monotone chain); vertices in a consistent orientation."""
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        out = []
        for q in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


class PixelGeometry:
    """Object positions from the object's own pixels, seen through the calibrated camera.

    The slots find each object at patch resolution (a 4 cm block covers ~5 of 32 x 32 patches), which
    leaves 5-10 mm of error, as much as the 8 mm tolerance of a grasp. The pixels themselves are far
    more precise (a block covers ~1000 pixels at 448 px, ~1.3 mm per pixel). Each object's colour is
    known from the scene description (as for the slot identity), so:

    1. every pixel is labelled with the nearest known colour in the chroma plane (hue x saturation,
       the table's colour included); grey pixels (the robot, the white plate, dark faces) are left
       unlabelled. On development frames: 99.8-100% precision and recall for every block;
    2. an object seen whole: the centroid of its pixels, back-projected through the *believed*
       camera pose to the plane at its resting height, is its position (median error 0.4 mm on
       development frames);
    3. an object seen in part (the hand or the arm hides some of it): its silhouette (the convex hull
       of the projected box or cylinder) is fitted to its pixels. Pixels of its colour inside the
       silhouette count for it, pixels of the table or another object inside count against it,
       hidden (grey) pixels count for neither, so a partial view does not drag the estimate;
    4. an object in the hand: its xy follows the hand (touch), its height is fitted along the
       vertical through that xy, so a block sliding down between the fingers is seen sinking.

    A bumped camera sees the world through the wrong pose, exactly as in
    pai.perception.camera_frame: the displacement is what the agent's cause inference detects.
    """

    GREY_SAT = 0.15      # below this saturation a pixel has no reliable hue (robot, plate, dark sides)
    DESATURATION = 0.85  # rendered colours are ~15% less saturated than the material colours
    WHITE_VALUE = 0.6    # the white plate is bright; hidden robot parts are often darker

    def __init__(self, env, camera: str, size: int):
        import mujoco
        m = env.model
        self.cam = m.camera(camera).id
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(env._nominal["cam_quat"][self.cam], float))
        self.R = rot.reshape(3, 3)
        self.p = np.asarray(env._nominal["cam_pos"][self.cam], float).copy()   # the believed pose
        self.size = int(size)
        self.f = self.size / 2 / np.tan(np.deg2rad(float(m.cam_fovy[self.cam])) / 2)
        self.specs = list(env.objects.values())
        rgba = {b.name: b.rgba for b in env.cfg.blocks}
        colours = [np.asarray(rgba[s.name] if s.name in rgba else env.cfg[s.kind].rgba, float)[:3] for s in self.specs]
        sat, _, ch = _chroma(np.stack(colours))
        self.coloured = sat > 0.3                   # the plate (near white) is found by brightness instead
        _, _, t_ch = _chroma(np.asarray(env.cfg.table_rgba, float)[:3])
        idx = [o for o in range(len(self.specs)) if self.coloured[o]]
        self.ref_label = np.array([-1] + idx, np.int8)
        self.refs = (self.DESATURATION * np.stack([t_ch] + [ch[o] for o in idx])).astype(np.float32)
        # shape of each object in its body frame: 8 box corners, or two rings of a cylinder
        self.shapes, self.rest_z, self.top = [], [], []
        a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        ring = np.stack([np.cos(a), np.sin(a)], -1)
        for s in self.specs:
            r, h = float(s.radius), float(s.height)
            if s.kind == "block":
                c = np.array([[x, y, z] for x in (-r, r) for y in (-r, r) for z in (-h / 2, h / 2)])
                self.rest_z.append(h / 2)
                self.top.append(h / 2)
            elif s.kind == "bowl":   # body origin at the bottom, walls up to the rim
                c = np.concatenate([np.c_[r * ring, np.zeros(24)], np.c_[r * ring, np.full(24, h)]])
                self.rest_z.append(0.0)
                self.top.append(0.012)                  # its base, where something put in it rests
            else:                    # plate: centred cylinder
                c = np.concatenate([np.c_[r * ring, np.full(24, -h / 2)], np.c_[r * ring, np.full(24, h / 2)]])
                self.rest_z.append(h / 2)
                self.top.append(h / 2)
            self.shapes.append(c)
        self.rest_z, self.top = np.array(self.rest_z), np.array(self.top)
        v, u = np.mgrid[0:self.size, 0:self.size]
        self.u, self.v = u.astype(np.float32) + 0.5, v.astype(np.float32) + 0.5    # pixel centres

    # ---- camera ---------------------------------------------------------------------------
    def project(self, x: np.ndarray) -> np.ndarray:
        """world points (..., 3) -> pixels (u right, v down) through the believed camera pose."""
        pc = (np.asarray(x, float) - self.p) @ self.R
        depth = np.maximum(-pc[..., 2], 1e-6)
        return np.stack([self.size / 2 + self.f * pc[..., 0] / depth, self.size / 2 - self.f * pc[..., 1] / depth], -1)

    def ray(self, u: float, v: float) -> np.ndarray:
        d = self.R @ np.array([(u - self.size / 2) / self.f, -(v - self.size / 2) / self.f, -1.0])
        return d / np.linalg.norm(d)

    def on_plane(self, u: float, v: float, z: float) -> np.ndarray:
        d = self.ray(u, v)
        return self.p + (z - self.p[2]) / d[2] * d

    # ---- pixels ---------------------------------------------------------------------------
    def label(self, image: np.ndarray) -> None:
        """Label every pixel: -2 unknown (grey), -1 table, o = object o (coloured objects only)."""
        rgb = image[..., :3].astype(np.float32) / 255.0
        sat, val, ch = _chroma(rgb)
        d = ((ch[..., None, :] - self.refs) ** 2).sum(-1)
        grey = sat < self.GREY_SAT
        self.lab = np.where(grey, np.int8(-2), self.ref_label[d.argmin(-1)]).astype(np.int8)
        self.white = grey & (val > self.WHITE_VALUE)

    def masks(self, o: int) -> tuple[np.ndarray, np.ndarray]:
        """(its pixels, pixels that are certainly something else) of object o."""
        own = self.lab == o if self.coloured[o] else self.white
        return own, (self.lab >= -1) & (self.lab != o)

    def silhouette(self, o: int, pos: np.ndarray, yaw: float = 0.0) -> np.ndarray:
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return _hull(self.project(self.shapes[o] @ rot.T + np.asarray(pos, float)))

    @staticmethod
    def _inside(poly: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        inside = np.ones(u.shape, bool)
        n = len(poly)
        # orientation of the hull: inside is on the same side of every edge as the centroid
        cu, cv = poly.mean(0)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            sgn = np.sign((b[0] - a[0]) * (cv - a[1]) - (b[1] - a[1]) * (cu - a[0]))
            inside &= sgn * ((b[0] - a[0]) * (v - a[1]) - (b[1] - a[1]) * (u - a[0])) >= 0
        return inside

    def _roi(self, polys: list[np.ndarray], margin: float) -> tuple[slice, slice]:
        pts = np.concatenate(polys)
        lo = np.clip(np.floor(pts.min(0) - margin), 0, self.size).astype(int)
        hi = np.clip(np.ceil(pts.max(0) + margin), 0, self.size).astype(int)
        return slice(lo[1], hi[1]), slice(lo[0], hi[0])

    def visible(self, o: int, pos: np.ndarray, yaw: float = 0.0) -> tuple[float, float, int]:
        """(fraction of the silhouette at pos covered by its pixels, fraction hidden, silhouette area)."""
        own, other = self.masks(o)
        poly = self.silhouette(o, pos, yaw)
        rs, cs = self._roi([poly], 1)
        inside = self._inside(poly, self.u[rs, cs], self.v[rs, cs])
        area = max(1, int(inside.sum()))
        seen = float((own[rs, cs] & inside).sum()) / area
        hidden = float((~own[rs, cs] & ~other[rs, cs] & inside).sum()) / area
        return seen, hidden, area

    def blob(self, o: int, near: np.ndarray, radius_px: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Pixel coordinates (u, v) of object o's pixels within radius_px of the pixel `near`."""
        own, _ = self.masks(o)
        r = int(radius_px)
        c = np.round(near).astype(int)
        rs = slice(max(0, c[1] - r), max(0, min(self.size, c[1] + r)))
        cs = slice(max(0, c[0] - r), max(0, min(self.size, c[0] + r)))
        vv, uu = np.nonzero(own[rs, cs])
        if len(uu) == 0:
            return None
        uu = uu + cs.start + 0.5
        vv = vv + rs.start + 0.5
        keep = (uu - near[0]) ** 2 + (vv - near[1]) ** 2 <= radius_px ** 2
        return uu[keep], vv[keep]

    def fit(self, o: int, start: np.ndarray, yaw: float = 0.0, free: str = "xy",
            steps=(0.008, 0.004, 0.002, 0.001, 0.0005)) -> np.ndarray:
        """Silhouette fit of object o (pixels of its colour inside +2, of anything else seen -1,
        hidden 0), coarse to fine, over xy at the start height or over the height at the start xy."""
        own, other = self.masks(o)
        best = np.asarray(start, float).copy()
        span = 3 * steps[0] + 0.01
        ext = ([[-span, -span, 0], [span, span, 0], [-span, span, 0], [span, -span, 0]] if free == "xy"
               else [[0, 0, -3 * span], [0, 0, 3 * span]])
        rs, cs = self._roi([self.silhouette(o, best + np.array(d), yaw) for d in ext], 3)
        w = 2.0 * own[rs, cs].astype(np.float32) - other[rs, cs]
        u, v = self.u[rs, cs], self.v[rs, cs]
        if w.size == 0:
            return best
        cache: dict = {}

        def score(x):
            key = tuple(np.round(x, 5))
            if key not in cache:
                cache[key] = float(w[self._inside(self.silhouette(o, x, yaw), u, v)].sum())
            return cache[key]
        for st in steps:
            offs = ([np.array([dx, dy, 0.0]) for dx in (-st, 0, st) for dy in (-st, 0, st)] if free == "xy"
                    else [np.array([0, 0, dz]) for dz in (-2 * st, -st, 0, st, 2 * st)])
            for _ in range(8):   # hill-climb at this step size
                cands = [best + d for d in offs]
                i = int(np.argmax([score(x) for x in cands]))
                if np.allclose(cands[i], best):
                    break
                best = cands[i]
        return best


class SlotPerception:
    def __init__(self, env, slot_ckpt: str, device: str = "cuda", camera: str | None = None,
                 refine: bool = True, bump_follows: bool = True):
        ckpt = torch.load(slot_ckpt, map_location=device, weights_only=False)
        self.model = SlotModel(**ckpt["model_config"]).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.tok_mean, self.tok_std = ckpt["tok_mean"].to(device), ckpt["tok_std"].to(device)
        self.dino = DinoFeatures(device=device)
        # the camera and resolution the slots were trained on (older checkpoints: front, 224)
        self.camera = camera or ckpt.get("camera", "front")
        self.image_size = int(ckpt.get("image_size", 224))
        # slots trained with a fixed slot per label: object o is always slot object_labels[0] + o
        self.fixed_start = int(ckpt.get("object_labels", [2])[0]) if ckpt.get("assignment") == "fixed" else None
        self.temporal = bool(ckpt.get("temporal_init", False))
        pca = ckpt.get("pca")
        self.pca = None if pca is None else (torch.as_tensor(pca["mean"], device=device),
                                             torch.as_tensor(pca["components"], device=device))
        self.env, self.device = env, device
        # What each known object should look like, from the scene description (kind + colour).
        ref = encode(env, env._observe())  # kind and colour of each object, from the scene description
        self.identity = torch.as_tensor(np.concatenate([ref[1:, KIND], ref[1:, COLOR]], -1), device=device)
        # refine: slots find the objects, their own pixels place them (PixelGeometry). Off: the slot
        # readout alone with the occlusion rules below (results/pixel_v1-v3).
        self.refine = bool(refine)
        self.geom = PixelGeometry(env, self.camera, self.image_size) if self.refine else None
        self.kinds = [s.kind for s in env.objects.values()]
        # The camera_shift disturbance bumps env.cam_id (the camera of obs["image"]), which is not the
        # camera the slots see through: without this, a "camera shift" never reaches pixel perception
        # (results/pixel_v1-v3 measured an invisible disturbance). bump_follows applies the same bump
        # to the perception camera, so pixels face the same disturbance as privileged state.
        self.bump_follows = bool(bump_follows)
        self.reset()

    # Finger opening when the hand has closed on something: an empty hand closes to ~0, a fully
    # open one is at 0.08 m. In between, the fingers are blocked by an object: touch, not vision.
    BLOCKED = (0.005, 0.065)
    HELD_RADIUS = 0.04  # an object seen this close to the hand when the fingers close is the one held
    OCCLUSION_RADIUS = 0.06  # the hand hides objects within this distance below it
    OCCLUSION_SHIFT = 0.03   # the largest apparent shift a partial view causes; beyond it, it moved
    SLIP_CLOSE = 0.015  # fingers closing this much further than at the grasp: the object is sliding out
    # pixel geometry (refine=True)
    SEARCH_PX = 70        # an object's pixels are looked for within this radius of where it was
    WHOLE_SEEN, WHOLE_HIDDEN = 0.9, 0.03   # seen whole: centroid; otherwise the silhouette is fitted
    MIN_SEEN = 0.12       # below this fraction of its silhouette visible, an object is hidden: kept
    HELD_MIN_SEEN = 0.15  # the height of a held object is measured when this much of it is seen
    SINK = 0.006          # a held object is never seen higher than this above its grasp height
    GONE_SEEN = 0.03      # a held object this invisible, while its colour is seen elsewhere, fell out
    YAW_EVERY = 5         # re-estimate a block's yaw every few frames while it is seen whole
    # Recalibrating the camera from pixels. With the slot readout alone (refine=False) this was off:
    # perception noise raised false 'camera moved' alarms in 3-5 of 20 clean episodes
    # (results/pixel_v2). With pixel geometry objects are placed to ~1 mm, and a bumped camera
    # displaces them as it displaces privileged state seen through the camera.
    @property
    def allow_recalibration(self) -> bool:
        return self.refine

    def reset(self) -> None:
        self.prev_slots = None
        self.prev_pos = None
        self.prev_t = None
        self.held = None  # (object index, offset from the hand, finger opening at the grasp)
        # the agent's own recalibration after it concluded "my camera moved" (added to everything
        # seen, as in pai.perception.camera_frame); the slice agent updates it
        self.correction = np.zeros(3, np.float32)
        self.raw = None         # last positions in the believed camera frame (before the correction)
        self.yaw = None         # block yaw estimates (rad)
        self.frame = 0

    def _follow_bump(self) -> None:
        env = self.env
        if not self.bump_follows or not hasattr(env, "_nominal"):
            return
        c = env.model.camera(self.camera).id
        if c == env.cam_id:
            return
        off = env.model.cam_pos[env.cam_id] - env._nominal["cam_pos"][env.cam_id]
        env.model.cam_pos[c] = env._nominal["cam_pos"][c] + off

    def __call__(self, obs: dict, prev_ee: np.ndarray | None = None) -> np.ndarray:
        """The `perceive` interface of the slice agent: observation -> entity tokens."""
        self._follow_bump()
        image = self.env.render(self.camera)  # the perception camera, not necessarily the one in obs
        tokens, self.last_match_cost = self.perceive(image, obs, prev_ee)
        return tokens

    # ---- pixel geometry ----------------------------------------------------------------------
    def _rest_height(self, o: int, xy: np.ndarray, raw: np.ndarray) -> float:
        """Height of object o's centre if it rests at xy: on the table, or on a plate/block/bowl under it."""
        g = self.geom
        z = g.rest_z[o]
        if self.kinds[o] != "block":
            return z
        h = g.specs[o].height / 2
        for s, spec in enumerate(g.specs):
            if s == o:
                continue
            d = np.abs(xy - raw[s, :2])
            under = (d.max() < spec.radius) if spec.kind == "block" else (np.linalg.norm(d) < spec.radius - 0.005)
            if under:
                z = max(z, raw[s, 2] + g.top[s] + h)
        return z

    def _yaw_search(self, o: int, pos: np.ndarray, around: float | None) -> float:
        g = self.geom
        own, other = g.masks(o)
        poly0 = g.silhouette(o, pos, 0.0)
        rs, cs = g._roi([poly0], 12)
        w = 2.0 * own[rs, cs].astype(np.float32) - other[rs, cs]
        u, v = g.u[rs, cs], g.v[rs, cs]
        score = lambda y: float(w[g._inside(g.silhouette(o, pos, y), u, v)].sum())
        cands = (np.deg2rad(np.arange(-45, 45, 7.5)) if around is None
                 else around + np.deg2rad(np.array([-6.0, -3.0, 0.0, 3.0, 6.0])))
        best = float(cands[int(np.argmax([score(y) for y in cands]))])
        for st in np.deg2rad([2.0, 1.0]):
            c = [best - st, best, best + st]
            best = float(c[int(np.argmax([score(y) for y in c]))])
        return (best + np.pi / 4) % (np.pi / 2) - np.pi / 4   # a cube looks the same every 90 degrees

    def _blob_elsewhere(self, o: int, near: np.ndarray):
        b = self.geom.blob(o, self.geom.project(near), 2 * self.geom.size)
        return b if b is not None and len(b[0]) >= 100 else None

    def _locate(self, image: np.ndarray, slot_raw: np.ndarray, ee_raw: np.ndarray) -> np.ndarray:
        """Positions of all objects in the believed camera frame, from their pixels."""
        g = self.geom
        g.label(image)
        n = len(slot_raw)
        first = self.raw is None
        prior = slot_raw.copy() if first else self.raw.copy()
        if self.yaw is None:
            self.yaw = np.zeros(n)
            self.yaw_known = np.zeros(n, bool)
        out = prior.copy()
        held = self.held[0] if self.held is not None else None
        dropped = None
        for o in range(n):
            coloured = bool(g.coloured[o])
            yaw = float(self.yaw[o]) if self.kinds[o] == "block" else 0.0
            if o == held:
                off = self.held[1]
                xy = ee_raw[:2] + off[:2]
                z0 = float(ee_raw[2] + off[2])
                zf = g.fit(o, np.r_[xy, z0], yaw, "z", steps=(0.004, 0.002, 0.001, 0.0005))
                seen, _, _ = g.visible(o, zf, yaw)
                if seen > self.HELD_MIN_SEEN:
                    # seen between the fingers: its height is measured, so a block sliding down
                    # between them is seen sinking (on development frames within 1-2 mm of the truth
                    # while the grasp-offset rule was up to 12 mm too high)
                    out[o] = np.r_[xy, np.clip(zf[2], g.rest_z[o], z0 + self.SINK)]
                    continue
                if seen > self.GONE_SEEN or self._blob_elsewhere(o, prior[o]) is None:
                    out[o] = np.r_[xy, z0]                    # hidden by the hand: it moves with the hand
                    continue
                self.held = None                              # not in the hand any more: it fell out
                held, dropped = None, o
            if not coloured or self.kinds[o] != "block":
                # the plate (white, like parts of the robot) and the bowl (its walls bias a centroid)
                # are fitted, starting from where they were
                start = np.r_[prior[o, :2], g.rest_z[o]]
                steps = (0.016, 0.008, 0.004, 0.002, 0.001, 0.0005) if first else (0.004, 0.002, 0.001, 0.0005)
                p = g.fit(o, start, yaw, "xy", steps=steps)
                seen, _, _ = g.visible(o, p, yaw)
                if seen > self.MIN_SEEN:
                    out[o] = p
                continue
            wide = first or o == dropped
            b = g.blob(o, g.project(prior[o]), 2 * g.size if wide else self.SEARCH_PX)
            if b is None or len(b[0]) < 10:
                continue                                      # hidden: it stays where it was
            z = self._rest_height(o, prior[o, :2], out)
            c = g.on_plane(float(b[0].mean()), float(b[1].mean()), z)
            z = self._rest_height(o, c[:2], out)
            c = g.on_plane(float(b[0].mean()), float(b[1].mean()), z)
            seen, hidden, area = g.visible(o, c, yaw)
            if len(b[0]) < self.MIN_SEEN * area:
                continue                                      # too little of it to place it
            if seen > self.WHOLE_SEEN and hidden < self.WHOLE_HIDDEN:
                p = c
                if not self.yaw_known[o] or self.frame % self.YAW_EVERY == 0:
                    self.yaw[o] = self._yaw_search(o, p, self.yaw[o] if self.yaw_known[o] else None)
                    self.yaw_known[o] = True
            else:
                p = g.fit(o, c, yaw, "xy")
            out[o] = p
        return out

    @torch.no_grad()
    def perceive(self, image: np.ndarray, obs: dict, prev_ee: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """image (S, S, 3) uint8 from `self.camera` at the training resolution; obs for proprioception.
        Returns tokens (N, TOKEN_DIM) and a match cost per object (lower = more confident)."""
        feats = self.dino(image[None]).reshape(1, -1, self.dino.model.embed_dim).float()
        if self.pca is not None:
            feats = (feats - self.pca[0]) @ self.pca[1]
        # Slots start from their learned queries unless the checkpoint was trained to start from the
        # previous frame's slots: untrained, they drift away within a few frames (held-out object
        # IoU 0.65 -> 0.04). With a fixed slot per object, identity needs no temporal link anyway.
        out = self.model(feats, init=self.prev_slots if self.temporal else None)
        self.prev_slots = out["slots"]
        slot_tok = out["tokens"][0] * self.tok_std + self.tok_mean              # (K, TOKEN_DIM), raw units
        looks = torch.cat([slot_tok[:, KIND], slot_tok[:, COLOR]], -1)          # (K, 7)
        cost = torch.cdist(self.identity, looks).cpu().numpy()                  # (n_objects, K)
        if self.fixed_start is not None:
            obj_idx = np.arange(len(self.identity))
            slot_idx = obj_idx + self.fixed_start
        else:
            obj_idx, slot_idx = linear_sum_assignment(cost)

        tokens = encode(self.env, obs, prev_ee)                                 # gripper from the body itself
        perceived = tokens.copy()
        n = len(self.identity)
        slot_raw = np.zeros((n, 3))
        slot_yaw = np.zeros(n)
        for o, k in zip(obj_idx, slot_idx):
            t = slot_tok[k].cpu().numpy()
            slot_raw[o] = t[POS]
            slot_yaw[o] = np.arctan2(t[YAW][0], t[YAW][1])
        ee, grip = tokens[0, POS], float(obs["gripper"])
        blocked = self.BLOCKED[0] < grip < self.BLOCKED[1]
        if self.held is not None and (not blocked or grip < self.held[2] - self.SLIP_CLOSE):
            self.held = None       # opened, closed on nothing, or the fingers close further: it slides out
        if self.refine:
            if self.held is None and blocked and self.raw is not None:
                ee_raw = ee - self.correction
                d = np.linalg.norm(self.raw - ee_raw, axis=1)
                o = int(np.argmin(d))
                if d[o] < self.HELD_RADIUS:
                    self.held = (o, self.raw[o] - ee_raw, grip)
            raw = self._locate(image, slot_raw, ee - self.correction)
            self.raw, self.frame = raw, self.frame + 1
            pos = (raw + self.correction).astype(np.float32)
            yaw = slot_yaw.copy()
            for o in range(n):
                if self.kinds[o] == "block" and self.yaw_known[o]:
                    # the cube's yaw is known modulo 90 degrees; the slot readout picks the quadrant
                    k = np.round((slot_yaw[o] - self.yaw[o]) / (np.pi / 2))
                    yaw[o] = self.yaw[o] + k * np.pi / 2
            perceived[1:, POS] = pos
            perceived[1:, YAW] = np.stack([np.sin(yaw), np.cos(yaw)], -1)
        else:
            pos = (slot_raw + self.correction).astype(np.float32)
            perceived[1:, POS] = pos
            perceived[1:, YAW] = np.stack([np.sin(slot_yaw), np.cos(slot_yaw)], -1)
            self._slot_rules(pos, perceived, ee, blocked, grip)
        # velocities over the time since the last perception (the agent perceives once per planner
        # step, i.e. every action_repeat control steps, not every control step)
        t = self.env.t
        if self.prev_pos is None or self.prev_t is None or t <= self.prev_t:
            perceived[1:, VEL] = 0.0
        else:
            perceived[1:, VEL] = (pos - self.prev_pos) / ((t - self.prev_t) * self.env.control_dt)
        self.prev_pos, self.prev_t = pos, t
        perceived[1:, GRIP] = 0.0
        perceived[1:, FORCE] = 0.0
        assert perceived.shape[-1] == TOKEN_DIM and KINDS[0] == "gripper"
        return perceived, cost[obj_idx, slot_idx]

    def _slot_rules(self, pos: np.ndarray, perceived: np.ndarray, ee: np.ndarray, blocked: bool, grip: float) -> None:
        """The held-object and permanence rules of the slot readout alone (refine=False), in place."""
        # An object in the hand is sensed through the body: the fingers stopped closing on it, and
        # from then on it moves with the hand at the offset it had when grasped. The camera cannot
        # do this well (the hand hides it; lifted, its height is not in the image), and a wrong
        # "it fell" belief makes the agent open its hand and re-grasp forever.
        if self.held is None and blocked:
            ref = self.prev_pos if self.prev_pos is not None else pos           # before this frame's occlusion
            d = np.linalg.norm(ref - ee, axis=1)
            o = int(np.argmin(d))
            if d[o] < self.HELD_RADIUS:
                self.held = (o, ref[o] - ee, grip)
        if self.held is not None:
            o, off, _ = self.held
            pos[o] = ee + off
            perceived[1 + o, POS] = pos[o]
        # Object permanence: an object under the hand is partly hidden by it, which shifts its seen
        # centre by 1-2 cm, exactly where the grasp needs it most. While the hand is over it, the last
        # clear estimate is kept, unless the camera sees it clearly elsewhere (more than the shift a
        # partial view causes): then the hand has pushed it, and the camera is right.
        if self.prev_pos is not None:
            for o in range(len(pos)):
                if self.held is not None and o == self.held[0]:
                    continue
                d_xy = np.linalg.norm(self.prev_pos[o, :2] - ee[:2])
                moved = np.linalg.norm(pos[o, :2] - self.prev_pos[o, :2]) > self.OCCLUSION_SHIFT
                if d_xy < self.OCCLUSION_RADIUS and ee[2] > self.prev_pos[o, 2] and not moved:
                    pos[o] = self.prev_pos[o]
                    perceived[1 + o, POS] = pos[o]
