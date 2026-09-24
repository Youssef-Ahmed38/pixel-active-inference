"""Run the generic skills (pai.skills) with every embodiment and print a table skill x embodiment.

    python scripts/skills_demo.py                               # all bodies with assets
    python scripts/skills_demo.py --embodiments panda_2f shadow -v   # plus every SkillResult
    python scripts/skills_demo.py --media                       # regenerate docs/media/skills/

Episodes (the skills get only a Target, never the fixture type):
- door    reach(handle) -> grasp -> turn(+0.4 rad about the bar) -> pull 0.12 m -> retreat
- drawer  reach(handle) -> grasp -> turn -> pull 0.15 m -> release -> retreat
- locked  the same drawer held shut by an equality constraint: reach -> grasp -> pull (should stall)
- peg     a peg taken from a holder -> insert into a slot; again -> insert into a solid block (should stall)

Cells: "ok", "STALL" (commanded motion, no progress), or the failure reason. The "locked pull" and
"insert block" columns are expected to read STALL: that is the signal a discovery agent uses.

--media writes docs/media/skills/<body>_door.gif (reach-grasp-turn-pull, 320 px, captioned with the
running skill) and grid.gif (the bodies side by side, 3 x 240 px), each kept under 2 MB.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.envs.door_scene import DoorSceneEnv, add_test_drawer  # noqa: E402
from pai.envs.embodiments import SPECS, available_embodiments  # noqa: E402
from pai.skills import SkillContext, handle_target, run_skill  # noqa: E402
from pai.skills.scenes import BLOCK, PEG, SLOT, hand_peg, let_go_holder, locked, with_peg  # noqa: E402

MEDIA_DIR = ROOT / "docs" / "media" / "skills"
MEDIA_BODIES = ["panda_2f", "allegro", "shadow"]
FRAME_EVERY = 4          # 50 Hz control / 4 = 12.5 fps
MAX_BYTES = 2_000_000

EPISODES = {
    "door": [("reach", {}), ("grasp", {}), ("turn", {"angle": 0.4}),
             ("push_pull", {"direction": "pull", "distance": 0.12}), ("retreat", {})],
    "drawer": [("reach", {}), ("grasp", {}), ("turn", {"angle": 0.4}),
               ("push_pull", {"direction": "pull", "distance": 0.15}), ("release", {}), ("retreat", {})],
    "locked": [("reach", {}), ("grasp", {}), ("push_pull", {"direction": "pull", "distance": 0.15})],
}
COLUMNS = [("door", "reach"), ("door", "grasp"), ("door", "turn"), ("door", "push_pull"), ("door", "retreat"),
           ("drawer", "reach"), ("drawer", "grasp"), ("drawer", "turn"), ("drawer", "push_pull"),
           ("drawer", "release"), ("drawer", "retreat"), ("locked", "push_pull"), ("peg", "grasp"),
           ("peg", "insert"), ("block", "insert")]
HEADERS = ["door reach", "grasp", "turn", "pull", "retreat", "drawer reach", "grasp", "turn", "pull", "release",
           "retreat", "locked pull", "peg grasp", "insert slot", "insert block"]


def cell(r) -> str:
    if r is None:
        return "-"
    if r.stalled:
        return "STALL"
    return "ok" if r.success else r.reason


def run_handle_episode(env, steps, seed=0, on_step=None, verbose=False, tag=""):
    ctx = SkillContext(env, env.reset(seed=seed), on_step=on_step)
    h = handle_target(env.fixture)
    out = {}
    for name, kw in steps:
        r = run_skill(ctx, name, h if name in ("reach", "turn", "push_pull") else None, **kw)
        out[name] = r
        if verbose:
            print(f"    {tag:7s} {r.row()}")
    f = env.fixture_state()
    return out, f


def run_peg(env, target, seed=0, verbose=False, tag=""):
    env.reset(seed=seed)
    hand_peg(env)
    ctx = SkillContext(env)
    g = run_skill(ctx, "grasp", PEG)
    let_go_holder(env)
    r = run_skill(ctx, "insert", target) if g.success else None
    if verbose:
        print(f"    {tag:7s} {g.row()}")
        if r is not None:
            print(f"    {tag:7s} {r.row()}")
    return g, r


def run_body(n, seed=0, verbose=False) -> dict:
    res: dict = {}
    for fx in ("door", "drawer", "locked"):
        env = DoorSceneEnv(n, locked(add_test_drawer) if fx == "locked" else fx, seed=seed)
        out, f = run_handle_episode(env, EPISODES[fx], seed, verbose=verbose, tag=fx)
        for k, r in out.items():
            res[(fx, k)] = r
        res[(fx, "opening")] = f["opening"]
        env.close()
    env = DoorSceneEnv(n, with_peg(add_test_drawer), seed=seed)
    g, r = run_peg(env, SLOT, seed, verbose, "slot")
    res[("peg", "grasp")], res[("peg", "insert")] = g, r
    _, r = run_peg(env, BLOCK, seed, verbose, "block")
    res[("block", "insert")] = r
    env.close()
    return res


# ---------------------------------------------------------------------------------------------- media
def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 6 * len(text), 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(im)


def save_gif(path, frames, colors=96):
    from PIL import Image
    ims = [Image.fromarray(f).quantize(colors=colors, method=Image.Quantize.MEDIANCUT) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 * FRAME_EVERY * 0.02), loop=0,
                optimize=True)


def record_door(n, seed=0, size=320):
    env = DoorSceneEnv(n, "door", seed=seed)
    frames, state = [], {"k": 0, "skill": ""}

    def on_step(e, obs):
        if state["k"] % FRAME_EVERY == 0:
            frames.append(label(e.render("fixture", size), f"{n}: {state['skill']}"))
        state["k"] += 1

    ctx = SkillContext(env, env.reset(seed=seed), on_step=on_step)
    h = handle_target(env.fixture)
    for name, kw in EPISODES["door"][:4]:
        state["skill"] = name + ("(+0.4 rad)" if name == "turn" else "(pull)" if name == "push_pull" else "")
        run_skill(ctx, name, h, **kw)
    env.close()
    return frames + [frames[-1]] * 8


def write_media(names, seed=0):
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    clips = {}
    for n in names:
        frames = record_door(n, seed)
        save_gif(MEDIA_DIR / f"{n}_door.gif", frames)
        clips[n] = frames
    from PIL import Image
    L = max(len(c) for c in clips.values())
    small = {n: [np.asarray(Image.fromarray(f).resize((240, 240), Image.Resampling.LANCZOS)) for f in c]
             for n, c in clips.items()}
    grid = [np.concatenate([small[n][min(i, len(small[n]) - 1)] for n in names], 1) for i in range(0, L, 2)]
    save_gif(MEDIA_DIR / "grid.gif", grid, colors=64)
    for p in sorted(MEDIA_DIR.glob("*.gif")):
        size = p.stat().st_size
        print(f"  {p.relative_to(ROOT)}  {size / 1e6:.2f} MB{'  (over 2 MB!)' if size > MAX_BYTES else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embodiments", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true", help="print every SkillResult")
    ap.add_argument("--media", action="store_true", help="regenerate docs/media/skills/")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="Attach conflict")

    names = args.embodiments or available_embodiments()
    missing = [n for n in names if not SPECS[n].available()]
    if missing:
        print(f"skipping (assets missing, run scripts/fetch_assets.py): {missing}")
        names = [n for n in names if n not in missing]

    table = {}
    for n in names:
        t0 = time.time()
        if args.verbose:
            print(n)
        table[n] = run_body(n, args.seed, args.verbose)
        print(f"  {n:13s} done in {time.time() - t0:.1f}s  (door opening {table[n][('door', 'opening')]:.2f} rad, "
              f"drawer {table[n][('drawer', 'opening')]:.3f} m, locked {table[n][('locked', 'opening')]:.4f} m)",
              flush=True)

    w = [max(len(h), 5) for h in HEADERS]
    print("\n" + f"{'embodiment':13s} " + " ".join(f"{h:>{k}s}" for h, k in zip(HEADERS, w)))
    for n in names:
        print(f"{n:13s} " + " ".join(f"{cell(table[n].get(c)):>{k}s}" for c, k in zip(COLUMNS, w)))
    expected_stall = [("locked", "push_pull"), ("block", "insert")]
    ok = sum(bool(table[n].get(c) and table[n][c].success) for n in names for c in COLUMNS if c not in expected_stall)
    st = sum(bool(table[n].get(c) and table[n][c].stalled) for n in names for c in expected_stall)
    print(f"\nsucceeded {ok}/{len(names) * (len(COLUMNS) - 2)} skill runs; "
          f"stalled where expected {st}/{len(names) * 2}")

    if args.media:
        write_media([n for n in MEDIA_BODIES if n in names] or names[:3], args.seed)


if __name__ == "__main__":
    main()
