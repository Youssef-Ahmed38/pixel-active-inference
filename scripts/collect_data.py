"""Collect (joint angles, image) pairs for the PixelAI decoder.

    python scripts/collect_data.py data.n_samples=60000 data.workers=8
"""

from _cli import parse

from pai.data.collect import collect

if __name__ == "__main__":
    _, cfg = parse(__doc__)
    collect(cfg)
