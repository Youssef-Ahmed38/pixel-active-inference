"""README media from an evaluation run: 480 px GIFs (every 2nd or 3rd frame) and one snapshot each.

    python scripts/readme_media.py results/slice_v7
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "slice_v7"
DST = ROOT / "docs" / "media" / "slice"
DST.mkdir(parents=True, exist_ok=True)
for cond in ("none", "push", "heavier_object", "slippery_object", "camera_shift"):
    src = SRC / f"slice_{cond}.gif"
    if not src.exists():
        continue
    frames = imageio.mimread(src, memtest=False)
    step = 2 if len(frames) < 200 else 3
    out = []
    for f in frames[::step]:
        im = Image.fromarray(np.asarray(f)[..., :3])
        w, h = im.size
        out.append(np.asarray(im.resize((480, int(h * 480 / w)), Image.LANCZOS)))
    path = DST / f"{cond}.gif"
    imageio.mimsave(path, out, duration=0.08 * step, loop=0)
    imageio.imwrite(DST / f"{cond}.png", out[len(out) * 2 // 3])
    print(path.name, len(out), f"{path.stat().st_size / 1e6:.2f} MB")
