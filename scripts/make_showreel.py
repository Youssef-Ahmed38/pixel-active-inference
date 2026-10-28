"""One short video of the whole project: a title card per stage, then that stage's milestone clip.

    python scripts/make_showreel.py

Writes docs/media/showreel.mp4 (960x540, < 3 MB) and docs/media/showreel.png (a contact sheet).
Every clip is an existing milestone video; numbers on the cards come from the reports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 960, 540, 25
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_AR = "C:/Windows/Fonts/tahoma.ttf"

SEGMENTS = [
    ("Stage 1 - the arm names hidden causes",
     ["A push, a heavier or slippery block, a moved camera: it infers which from its own surprise,",
      "estimates it (extra mass, camera offset, onset) and adapts. Cause correct 97/100 (learned)."],
     "results/slice_v7/slice_heavier_object.gif", 8),
    ("Stage 2 - a humanoid learns to walk",
     ["Curious motor babbling, then a practice curriculum: stand, walk, turn, run;",
      "every gate 20/20 on fresh seeds, push recovery 97% at 1.0 m/s."],
     "docs/media/locomotion/g1_locomotion_v3.mp4", 12),
    ("Stage 2 - after practising with walls",
     ["Walking into a wall and pushing on: falls 2/40 (v4) vs 14/40 (v3).",
      "It names why it fell (gait, push, slip, wall) in 92.6% of 544 fresh falls."],
     "docs/media/locomotion/g1_wall_contact.mp4", 10),
    ("Stage 3 - exploring a maze through its own eyes",
     ["Head camera and its own odometry, no ground-truth map or position;",
      "curiosity (expected free energy): goal in 26/30 new mazes vs 17/30 nearest frontier."],
     "docs/media/maze/walker_maze_branchy_camera.mp4", 24),
    ("Stage 4 - discovering how a locked door opens",
     ["Nothing tells it what a key is: it experiments and finds the key in the drawer",
      "(earlier build shown). The newer agent learns the chain as a recipe: 38/40 unseen scenes."],
     "docs/media/discovery/key_in_drawer.gif", 10),
    ("Stage 7 - riding a bicycle",
     ["Learned by watching an expert's motion, then improved by practice."],
     "docs/media/bike/expert.mp4", 8),
]

EXPLANATION = {
    "title": "Stage 6 - it explains why, in English and Arabic",
    "en": ["My evidence suggests that the object weighed more than expected.",
           "I updated my expected weight. My final observation indicates that the goal was reached.",
           "I estimated about 0.438 kg of extra mass.            (true: 0.435 kg)"],
    "ar": ["تشير أدلتي إلى أن الجسم كان أثقل مما توقعت. حدثت توقعاتي للوزن.",
           "تشير ملاحظتي الأخيرة إلى تحقق الهدف. قدرت الكتلة الإضافية بحوالي 0.438 كغ."],
}


def card(title: str, lines: list[str], arabic: list[str] | None = None) -> np.ndarray:
    img = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(img)
    d.text((48, 150), title, font=ImageFont.truetype(FONT, 34), fill=(245, 200, 80))
    small = ImageFont.truetype(FONT, 22)
    y = 230
    for line in lines:
        d.text((48, y), line, font=small, fill=(230, 230, 230))
        y += 36
    if arabic:
        ar = ImageFont.truetype(FONT_AR, 24)
        y += 14
        for line in arabic:
            # PIL's complex-text layout (raqm) shapes and orders Arabic itself: pass the raw text
            w = d.textlength(line, font=ar, direction="rtl", language="ar")
            d.text((W - 48 - w, y), line, font=ar, fill=(150, 210, 255), direction="rtl", language="ar")
            y += 40
    d.text((48, H - 50), "Pixel Active Inference - one agent, seven worlds (simulation)",
           font=ImageFont.truetype(FONT, 16), fill=(120, 130, 145))
    return np.asarray(img)


def fit(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)[..., :3]
    h, w = frame.shape[:2]
    s = min(W / w, H / h)
    small = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 3), np.uint8)
    y, x = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    return out


def clip(path: Path, seconds: float) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    src_fps = meta.get("fps") or (1000.0 / meta["duration"] if meta.get("duration") else FPS)
    frames = [f for f in reader]
    reader.close()
    total = len(frames) / src_fps
    # speed the clip up to fit ``seconds`` when it is longer; never slow it down
    speed = max(1.0, total / seconds)
    n = int(min(total, seconds * speed) / speed * FPS)
    idx = [min(len(frames) - 1, int(i * speed * src_fps / FPS)) for i in range(n)]
    return [fit(frames[i]) for i in idx]


def main():
    frames: list[np.ndarray] = []
    hold = lambda img, s: frames.extend([img] * int(s * FPS))
    hold(card("One agent, seven worlds",
              ["Active inference: learned world models, surprise, hidden-cause inference,",
               "curiosity, memory and recipes - the same principles in every world.",
               "Every number below is on seeds never used in development."]), 4)
    for i, (title, lines, src, seconds) in enumerate(SEGMENTS):
        hold(card(title, lines), 3.5)
        frames.extend(clip(ROOT / src, seconds))
        if i == 0:
            hold(card(EXPLANATION["title"], EXPLANATION["en"], EXPLANATION["ar"]), 7)
    out = ROOT / "docs/media/showreel.mp4"
    imageio.mimsave(out, frames, fps=FPS, quality=5, macro_block_size=8)
    sheet = np.vstack([np.hstack([cv2.resize(frames[int(k * len(frames) / 8)], (W // 2, H // 2))
                                  for k in range(j, j + 2)]) for j in range(0, 8, 2)])
    cv2.imwrite(str(out.with_suffix(".png")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(out.relative_to(ROOT), f"{out.stat().st_size / 1e6:.2f} MB", f"{len(frames) / FPS:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
