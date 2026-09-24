"""Run the scripted reach -> grasp -> pull with every embodiment on every fixture and print a table.

    python scripts/embodiment_demo.py                          # all bodies with assets, default fixtures
    python scripts/embodiment_demo.py --embodiments shadow g1_hands --fixtures door lib:room_door_bar
    python scripts/embodiment_demo.py --gif                    # results/embodiments/<body>_<fixture>.gif
    python scripts/embodiment_demo.py --media                  # regenerate docs/media/embodiments/

Fixtures: "door" / "drawer" are the local test fixtures of door_scene.py; "lib:<type>" is a type of the
fixture library (pai.envs.fixtures), e.g. lib:door_bar_left, lib:drawer_bar, lib:room_door_bar.

--media writes small committed media: <body>.png (a snapshot of the grasp on the door), grid.png (all
bodies side by side) and <body>_door.gif (reach-grasp-pull, 320 px, 12.5 fps, long clips subsampled to
75 frames so each GIF stays under ~1.5 MB).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.envs.door_scene import DoorSceneEnv  # noqa: E402
from pai.envs.embodiments import SPECS, available_embodiments  # noqa: E402
from pai.envs.scripted_reach import ScriptedReachGraspPull  # noqa: E402

DEFAULT_FIXTURES = ["door", "drawer", "lib:door_bar_left", "lib:drawer_bar", "lib:room_door_bar"]
MEDIA_DIR = ROOT / "docs" / "media" / "embodiments"
GIF_DIR = ROOT / "results" / "embodiments"
FRAME_EVERY = 4        # control steps per frame: 50 Hz / 4 = 12.5 fps, real time
MEDIA_SIZE = 320
MEDIA_MAX_FRAMES = 75  # committed GIFs stay ~1-1.5 MB


def run(env, seed, frames=None, camera="fixture", size=MEDIA_SIZE, snapshot=None, max_steps=800):
    obs = env.reset(seed=seed)
    pol = ScriptedReachGraspPull(env)
    stable = True
    for t in range(max_steps):
        obs = env.step(pol.act(obs))
        if frames is not None and t % FRAME_EVERY == 0:
            frames.append(env.render(camera, size))
        if snapshot is not None and pol.phase == "pull" and pol.k == 25 and not snapshot:
            snapshot.append(env.render(camera, size))   # a moment into the pull: the grasp holds
        if not env.stable():
            stable = False
            break
        if pol.done():
            break
    if frames is not None:   # a short hold at the end of a clip
        frames += [frames[-1]] * 6
    f = obs["fixture"]
    return {**pol.log, "opening": f["opening"], "opening_frac": f["opening_frac"], "opened": bool(f["opened"]),
            "stable": stable, "steps": t + 1}


def label(img, text):
    """Burn a small caption into the top-left corner (no font dependency: Pillow's default font)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def save_gif(path, frames, max_frames=None, colors=128):
    """GIF at 12.5 fps; with max_frames, long clips are subsampled (played faster) to stay small."""
    from PIL import Image
    if max_frames and len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    ims = [Image.fromarray(f).quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 * FRAME_EVERY * 0.02), loop=0,
                optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embodiments", nargs="*", default=None)
    ap.add_argument("--fixtures", nargs="*", default=DEFAULT_FIXTURES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", action="store_true", help="write a GIF per run to results/embodiments/")
    ap.add_argument("--media", action="store_true", help="regenerate docs/media/embodiments/")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="Attach conflict")

    names = args.embodiments or available_embodiments()
    missing = [n for n in names if not SPECS[n].available()]
    if missing:
        print(f"skipping (assets missing, run scripts/fetch_assets.py): {missing}")
        names = [n for n in names if n not in missing]
    if args.gif:
        GIF_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in names:
        for fx in args.fixtures:
            t0 = time.time()
            env = DoorSceneEnv(n, fx, seed=args.seed)
            frames = [] if args.gif else None
            r = run(env, args.seed, frames)
            if frames:
                save_gif(GIF_DIR / f"{n}_{fx.replace(':', '_')}.gif", frames)
            env.close()
            rows.append((n, fx, r, time.time() - t0))
            print(f"  {n:13s} {fx:20s} done in {rows[-1][3]:.1f}s", flush=True)

    print()
    print(f"{'embodiment':13s} {'fingers':>7s} {'fixture':20s} {'reached':>8s} {'grasped':>8s} {'contacts':>8s} "
          f"{'opening':>8s} {'frac':>6s} {'opened':>7s} {'stable':>7s}")
    for n, fx, r, _ in rows:
        print(f"{n:13s} {SPECS[n].n_fingers:7d} {fx:20s} {str(r['reached']):>8s} {str(r['grasped']):>8s} "
              f"{r['contact_fingers']:8d} {r['opening']:8.3f} {r['opening_frac']:6.2f} {str(r['opened']):>7s} "
              f"{str(r['stable']):>7s}")
    ok = sum(r["opened"] for _, _, r, _ in rows)
    print(f"\nopened {ok}/{len(rows)} (door/drawer 'opened' = past success_q of the fixture)")

    if args.media:
        write_media(names, args.seed)


def write_media(names, seed):
    import imageio.v2 as imageio
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    tiles = []
    for n in names:
        env = DoorSceneEnv(n, "door", seed=seed)
        frames, snap = [], []
        run(env, seed, frames, snapshot=snap)
        env.close()
        img = label(snap[0] if snap else frames[len(frames) // 2], f"{n} ({SPECS[n].n_fingers} fingers)")
        imageio.imwrite(MEDIA_DIR / f"{n}.png", img)
        save_gif(MEDIA_DIR / f"{n}_door.gif", [label(f, n) for f in frames], max_frames=MEDIA_MAX_FRAMES)
        tiles.append(img)
        print(f"  media: {n}.png, {n}_door.gif")
    cols = 3
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    imageio.imwrite(MEDIA_DIR / "grid.png", grid)
    for p in sorted(MEDIA_DIR.iterdir()):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
