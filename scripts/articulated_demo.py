"""Scripted opening of every tabletop fixture type: success table, optional GIFs and README media.

    python scripts/articulated_demo.py --seeds 3                       # table: types x hidden causes
    python scripts/articulated_demo.py --types drawer_bar,lid_knob --causes none,locked --gif
    python scripts/articulated_demo.py --media                         # regenerate docs/media/articulated/

Each (type, seed) builds a fresh env with that fixture (the type's parameters are drawn from the
seed), forces the hidden cause of the target part, runs the scripted opener (pai.envs.scripted_open)
and counts an opening when an articulation_opened event was logged. `latched` only applies to lever
types. --gif writes one GIF per case to results/articulated/ (git-ignored); --media writes small
committed PNGs and GIFs to docs/media/articulated/.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
from _cli import parse

from pai.config import REPO_ROOT
from pai.envs.articulated import ArticulatedEnv
from pai.envs.fixtures import FIXTURE_TYPES, add_fixture, fixture_types, make_fixture
from pai.envs.scripted_open import run_open

MEDIA_DIR = REPO_ROOT / "docs" / "media" / "articulated"
GIF_DIR = REPO_ROOT / "results" / "articulated"
MEDIA_SIZE = (240, 320)  # (height, width)
FRAME_EVERY = 5          # control steps per GIF frame: 50 Hz / 5 = 10 fps
GIF_COLOURS = 64         # palette size; keeps a ~10 s GIF under ~2 MB
FAMILY_EXAMPLES = {      # one type per family for the contact sheet
    "drawer": "drawer_bar", "drawer_stack": "stack3_bar", "cabinet_door": "door_lever_down_right",
    "room_door": "room_door_lever", "sliding_door": "sliding_bar_left", "lid": "lid_knob", "flap": "flap_knob",
}
MEDIA_GIFS = [  # (file stem, type, seed, hidden cause)
    ("open_drawer_knob", "drawer_knob", 1, "none"),
    ("open_door_lever_latched", "door_lever_down_right", 1, "latched"),
    ("open_lid_knob", "lid_knob", 1, "none"),
    ("open_flap_bar", "flap_bar", 1, "none"),
    ("fail_drawer_locked", "drawer_bar", 1, "locked"),
    ("fail_door_stuck", "door_bar_right", 1, "stuck"),
]


def extra(ap):
    ap.set_defaults(config="configs/articulated.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--types", default="all", help="comma-separated FIXTURE_TYPES names, or 'all' (tabletop)")
    ap.add_argument("--causes", default="none,locked,stuck,blocked,latched")
    ap.add_argument("--gif", action="store_true", help="save one GIF per case to results/articulated/")
    ap.add_argument("--media", action="store_true", help="regenerate docs/media/articulated/ and exit")


def make_env(cfg_env, type_name: str, seed: int) -> ArticulatedEnv:
    fx = make_fixture(np.random.default_rng(seed), type_name)
    return ArticulatedEnv(cfg_env, fixtures={"fixture": fx})


def media_camera(env: ArticulatedEnv, distance: float = 0.85, azimuth: float = 50.0,
                 elevation: float = -22.0) -> mujoco.MjvCamera:
    """A free camera looking at the fixture's front from the robot's right, so arm and handle both show."""
    cam = mujoco.MjvCamera()
    xy, _ = env.fixture_pose["fixture"]
    cam.lookat[:] = [xy[0] - 0.1, xy[1], 0.13]
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def run_case(env: ArticulatedEnv, seed: int, cause: str, record: bool = False) -> tuple[dict, list]:
    part = env.fixtures[0].target
    env.reset(seed=seed, causes={part: cause})
    frames: list = []
    renderer = cam = None
    if record:
        renderer = mujoco.Renderer(env.model, *MEDIA_SIZE)
        cam = media_camera(env)

    def render():
        renderer.update_scene(env.data, camera=cam)
        return renderer.render()

    r = run_open(env, part, frames=frames if record else None, render=render, frame_every=FRAME_EVERY)
    if renderer is not None:
        renderer.close()
    return r, frames


def success_table(cfg, types: list[str], seeds: int, causes: list[str], gif: bool) -> None:
    rows: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    for name in types:
        for seed in range(seeds):
            env = make_env(cfg.env, name, seed)
            has_lever = any(p.lever_joint for p in env.fixtures[0].parts)
            for cause in causes:
                if cause == "latched" and not has_lever:
                    continue
                r, frames = run_case(env, seed, cause, record=gif)
                rows[name][cause].append(r["opened"])
                if gif:
                    GIF_DIR.mkdir(parents=True, exist_ok=True)
                    save_gif(GIF_DIR / f"{name}_{cause}_s{seed}.gif", frames)
            env.close()
    header = f"{'type':24s} {'family':13s} {'split':5s} " + " ".join(f"{c:>8s}" for c in causes)
    print(header)
    print("-" * len(header))
    fam: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for name in types:
        t = FIXTURE_TYPES[name]
        cells = []
        for c in causes:
            v = rows[name].get(c)
            cells.append(f"{sum(v)}/{len(v)}".rjust(8) if v else "-".rjust(8))
            fam[t.family][c] += v or []
        print(f"{name:24s} {t.family:13s} {t.split:5s} " + " ".join(cells))
    print("\nper family (opened / runs):")
    for f, cs in fam.items():
        print(f"  {f:13s} " + " ".join(f"{c}={sum(v)}/{len(v)}" for c, v in cs.items() if v))
    print(f"\n{time.time() - t0:.0f} s. Expected: `none` and `latched` open (the opener turns levers first);"
          " `locked`, `stuck` and `blocked` do not; the lower drawers of a centred stack are out of reach"
          " (the wrist hits the handle above).")


# ---------------------------------------------------------------------------------------- media
def save_gif(path: Path, frames: list[np.ndarray]) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f).quantize(GIF_COLOURS, method=Image.Quantize.MEDIANCUT) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(FRAME_EVERY * 20), loop=0, optimize=True)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def _snapshot(env: ArticulatedEnv, opening: float) -> np.ndarray:
    """The fixture with its target part set to `opening`, rendered from the media camera."""
    part = env.fixtures[0].target
    p = env.fx.parts[part]
    ids = env.fx.ids[part]
    env.data.qpos[ids["qpos"]] = p.range[0] + opening * (p.range[1] - p.range[0])
    mujoco.mj_forward(env.model, env.data)
    r = mujoco.Renderer(env.model, *MEDIA_SIZE)
    r.update_scene(env.data, camera=media_camera(env))
    img = r.render()
    r.close()
    return img


def _room_door_snapshot(seed: int = 0) -> np.ndarray:
    """The room door lives outside the tabletop scene: render it in a bare room."""
    spec = mujoco.MjSpec()
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], rgba=[0.35, 0.33, 0.3, 1])
    spec.worldbody.add_light(pos=[2.5, -1.5, 3.0], dir=[-0.6, 0.4, -1.0], diffuse=[0.8, 0.8, 0.8])
    fx = add_fixture(spec, spec.worldbody, make_fixture(np.random.default_rng(seed), FAMILY_EXAMPLES["room_door"]), "room")
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[model.joint(fx.parts[0].joint).qposadr[0]] = 0.5
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.3, 0.0, 1.0]
    cam.distance, cam.azimuth, cam.elevation = 3.8, 205.0, -10.0
    r = mujoco.Renderer(model, *MEDIA_SIZE)
    r.update_scene(data, camera=cam)
    img = r.render()
    r.close()
    return img


def make_media(cfg) -> None:
    import imageio

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    tiles = []
    for family, name in FAMILY_EXAMPLES.items():
        if family == "room_door":
            img = _room_door_snapshot()
        else:
            env = make_env(cfg.env, name, 0)
            env.reset(seed=0)
            img = _snapshot(env, 0.6)
            env.close()
        imageio.imwrite(MEDIA_DIR / f"family_{family}.png", img)
        tiles.append(_label(img, f"{family}: {name}"))
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.concatenate([np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)], axis=0)
    imageio.imwrite(MEDIA_DIR / "families.png", grid)
    print("wrote", MEDIA_DIR / "families.png")
    for stem, name, seed, cause in MEDIA_GIFS:
        env = make_env(cfg.env, name, seed)
        r, frames = run_case(env, seed, cause, record=True)
        env.close()
        path = MEDIA_DIR / f"{stem}.gif"
        save_gif(path, [_label(f, f"{name} ({cause})") for f in frames])
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames, opened={r['opened']})")


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    if args.media:
        make_media(cfg)
    else:
        types = fixture_types(tabletop=True) if args.types == "all" else args.types.split(",")
        success_table(cfg, types, args.seeds, args.causes.split(","), args.gif)
