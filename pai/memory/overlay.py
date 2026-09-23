"""Side panel for demo videos: what the agent is doing, how surprised it is, and why it thinks so.

Every value drawn comes from the agent's own state at that moment (subgoal, surprise, cause
belief), never from the simulator's ground truth, so the video shows the agent's reasoning, not ours.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pai.causes.inference import SURPRISE_THRESHOLD
from pai.memory.report import CAUSE_TEXT

PANEL_W = 300
BG, FG, DIM, WARN, OK = (18, 20, 26), (235, 235, 235), (140, 145, 155), (235, 120, 70), (110, 200, 120)


def annotate(frame: np.ndarray, goal: str, subgoal: str | None, surprise: float,
             posterior: dict[str, float] | None, t: float) -> np.ndarray:
    h, w = frame.shape[:2]
    canvas = Image.new("RGB", (w + PANEL_W, h), BG)
    canvas.paste(Image.fromarray(frame), (0, 0))
    d = ImageDraw.Draw(canvas)
    big, small = ImageFont.load_default(size=16), ImageFont.load_default(size=13)
    x, y = w + 14, 12
    d.text((x, y), f"goal: {goal}", fill=FG, font=big)
    y += 26
    d.text((x, y), f"now: {subgoal or 'done'}", fill=FG, font=small)
    y += 20
    d.text((x, y), f"t = {t:4.1f} s", fill=DIM, font=small)
    y += 30

    d.text((x, y), "surprise", fill=DIM, font=small)
    y += 18
    level = min(1.0, surprise / (3 * SURPRISE_THRESHOLD))
    colour = WARN if surprise > SURPRISE_THRESHOLD else OK
    d.rectangle([x, y, x + PANEL_W - 30, y + 12], outline=DIM)
    d.rectangle([x, y, x + int((PANEL_W - 30) * level), y + 12], fill=colour)
    thr = x + int((PANEL_W - 30) / 3)
    d.line([thr, y - 3, thr, y + 15], fill=FG)
    y += 30

    d.text((x, y), "my explanation", fill=DIM, font=small)
    y += 18
    if posterior:
        for cause, p in sorted(posterior.items(), key=lambda kv: -kv[1]):
            d.rectangle([x, y + 2, x + int(120 * p), y + 12], fill=WARN if cause != "none" and p > 0.5 else DIM)
            d.text((x + 128, y), f"{p:.2f} {cause}", fill=FG, font=small)
            y += 18
        best = max(posterior, key=posterior.get)
        y += 6
        d.text((x, y), f'"{CAUSE_TEXT[best]}"', fill=FG, font=small)
    else:
        d.text((x, y), "(nothing to explain)", fill=DIM, font=small)
    return np.asarray(canvas)
