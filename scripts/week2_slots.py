"""Week 2 in one go (for a Kaggle session): collect frames -> DINOv2 features -> train object slots.

    python scripts/week2_slots.py --config configs/tabletop.yaml --episodes 300 --workers 4 \
        slots.frames_dir=/tmp/frames slots.cache_dir=/tmp/features slots.out_dir=runs/slots

Each step is skipped if its output already exists, so a restarted session continues where it
stopped. Keep the bulky frames and features on local disk (/tmp) and only the checkpoint and logs
in the saved output (runs/).
"""

from pathlib import Path

from _cli import parse

from pai.perception.collect import collect_frames
from pai.perception.train_slots import extract_features, train_slots


def extra(ap):
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    sc = cfg.slots
    collect_frames(cfg.env, sc.frames_dir, args.episodes, workers=args.workers, seed=int(cfg.seed) + 101)
    if not (Path(sc.cache_dir) / "episode.npy").exists():
        extract_features(sc.frames_dir, sc.cache_dir, n_labels=2 + 6)
    print(train_slots(cfg))
