"""Home doors: oracle success table per type x embodiment x lock state, and the README media.

    python scripts/home_door_demo.py                                   # panda_2f, robotiq_2f85, shadow; all types
    python scripts/home_door_demo.py --embodiments shadow g1_hands --types hd_knob_pull_left --places drawer
    python scripts/home_door_demo.py --media                           # regenerate docs/media/home_doors/

The oracle (pai.envs.home_oracle) reads the ground truth and runs a fixed plan, only to show that every
variant is physically solvable with each body; it is never used by the learning agent. A case counts as
solved when a door_opened event was logged. "other_room" keys make key-locked doors unsolvable; the
table reports them separately (the oracle must fail there).
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.envs.door_scene import DoorSceneEnv  # noqa: E402
from pai.envs.embodiments import SPECS  # noqa: E402
from pai.envs.home_doors import HOME_DOOR_TYPES, home_door_fixture, home_door_types  # noqa: E402
from pai.envs.home_oracle import run_oracle  # noqa: E402

MEDIA_DIR = ROOT / "docs" / "media" / "home_doors"
FRAME_EVERY = 5          # 50 Hz / 5 = 10 fps
MEDIA_SIZE = 320
MAX_FRAMES = 110


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def save_gif(path, frames, colors=96):
    from PIL import Image
    if len(frames) > MAX_FRAMES:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, MAX_FRAMES).astype(int)]
    ims = [Image.fromarray(f).quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 * FRAME_EVERY * 0.02), loop=0,
                optimize=True)


def table(embodiments, types, states, places, seeds):
    rows = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    for emb in embodiments:
        for t in types:
            env = DoorSceneEnv(emb, home_door_fixture(t), seed=0)
            for st in states:
                if st not in env.runtime.available_states():
                    continue
                for pl in (places if st in ("key", "both") else ["table"]):
                    for s in range(seeds):
                        r = run_oracle(env, seed=s, lock_state=st, key_place=pl)
                        key = f"{st}/{pl}" if st in ("key", "both") else st
                        rows[(emb, t)][key].append((r["opened"], r["why"]))
            env.close()
            print(f"  {emb:13s} {t:22s} done ({time.time() - t0:.0f} s)", flush=True)
    cols = sorted({c for v in rows.values() for c in v}, key=lambda c: (c.split("/")[0] != "unlocked", c))
    print(f"\n{'embodiment':13s} {'type':22s} {'split':5s} " + " ".join(f"{c:>18s}" for c in cols))
    tot = defaultdict(lambda: [0, 0])
    for (emb, t), v in rows.items():
        cells = []
        for c in cols:
            x = v.get(c)
            if not x:
                cells.append(f"{'-':>18s}")
                continue
            ok = sum(o for o, _ in x)
            cells.append(f"{ok}/{len(x)}".rjust(18))
            solvable = not c.endswith("other_room")
            tot[(emb, solvable)][0] += ok
            tot[(emb, solvable)][1] += len(x)
        print(f"{emb:13s} {t:22s} {HOME_DOOR_TYPES[t].split:5s} " + " ".join(cells))
    print()
    for (emb, solvable), (ok, n) in sorted(tot.items()):
        print(f"{emb:13s} {'solvable' if solvable else 'UNSOLVABLE (key in other room)':32s} opened {ok}/{n}")
    fails = [(emb, t, c, why) for (emb, t), v in rows.items() for c, x in v.items() for o, why in x
             if not o and not c.endswith("other_room")]
    if fails:
        print("\nfailures:")
        for f in fails:
            print("  ", *f)


def media():
    """Snapshot grid of the variants and GIFs of the mechanisms (robotiq: fast and clear)."""
    import imageio.v2 as imageio
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    tiles = []
    for t in HOME_DOOR_TYPES:
        env = DoorSceneEnv("robotiq_2f85", home_door_fixture(t), seed=0)
        env.reset(seed=0, lock_state="unlocked")
        tiles.append(label(env.render("overview", MEDIA_SIZE), t))
        env.close()
    cols = 4
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    imageio.imwrite(MEDIA_DIR / "variants.png", grid)
    clips = [  # (file, embodiment, type, lock state, key place, caption)
        ("knob_turn_open", "robotiq_2f85", "hd_knob_pull_left", "unlocked", "table", "turn knob, pull"),
        ("deadbolt_thumb_unlock", "robotiq_2f85", "hd_lever_pull_right", "deadbolt", "table", "thumb-turn, lever"),
        ("key_in_drawer", "robotiq_2f85", "hd_knob_push_right", "key", "drawer", "drawer -> key -> door"),
        ("pushbar_push", "robotiq_2f85", "hd_pushbar_push_left", "unlocked", "table", "push bar"),
        ("panda_thumb_unlock", "panda_2f", "hd_knob_pull_left", "deadbolt", "table", "panda: thumb-turn, knob"),
    ]
    for stem, emb, t, st, pl, cap in clips:
        env = DoorSceneEnv(emb, home_door_fixture(t), seed=0)
        frames = []
        r = run_oracle(env, seed=0, lock_state=st, key_place=pl, frames=frames, camera="overview",
                       frame_every=FRAME_EVERY, frame_size=MEDIA_SIZE)
        save_gif(MEDIA_DIR / f"{stem}.gif", [label(f, f"{cap} ({emb})") for f in frames + [frames[-1]] * 8])
        print(f"  {stem}.gif opened={r['opened']} {r['why']}")
        env.close()
    wrong_key_clip()
    for p in sorted(MEDIA_DIR.iterdir()):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.2f} MB")


def wrong_key_clip():
    """A wrong key goes in but does not turn: the oracle is handed the decoy on purpose."""
    from pai.envs import home_oracle as HO
    env = DoorSceneEnv("robotiq_2f85", home_door_fixture("hd_knob_push_right", n_keys=2), seed=0)
    obs = env.reset(seed=0, lock_state="key", key_place="table")
    info = env.info
    wrong = next(n for n, k in info["keys"].items() if not k["matches"])
    oracle = HO.HomeDoorOracle(env, max_steps=1400)
    oracle.force_key = wrong
    frames = []
    for t in range(1400):
        obs = env.step(oracle.act(obs))
        if t % FRAME_EVERY == 0:
            frames.append(label(env.render("overview", MEDIA_SIZE), "wrong key: inserts, does not turn"))
        if oracle.done():
            break
    save_gif(MEDIA_DIR / "wrong_key_fails.gif", frames + [frames[-1]] * 8)
    print("  wrong_key_fails.gif", oracle.stage, oracle.result, [e["type"] for e in env.runtime.events][1:])
    env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embodiments", nargs="*", default=["panda_2f", "robotiq_2f85", "shadow"])
    ap.add_argument("--types", nargs="*", default=list(HOME_DOOR_TYPES))
    ap.add_argument("--split", default=None, help="only types of this split")
    ap.add_argument("--states", nargs="*", default=["unlocked", "deadbolt", "key", "both"])
    ap.add_argument("--places", nargs="*", default=["table", "shelf", "drawer", "other_room"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--media", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="Attach conflict")
    if args.media:
        media()
        return
    types = [t for t in args.types if args.split is None or t in home_door_types(args.split)]
    embs = [e for e in args.embodiments if SPECS[e].available()]
    table(embs, types, args.states, args.places, args.seeds)


if __name__ == "__main__":
    main()
