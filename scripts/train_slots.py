"""Week 2: extract DINOv2 features from collected frames (once), then train object slots.

    python scripts/collect_frames.py --config configs/tabletop.yaml --episodes 300
    python scripts/train_slots.py --config configs/tabletop.yaml --extract
    python scripts/train_slots.py --config configs/tabletop.yaml
"""

from _cli import parse

from pai.perception.train_slots import extract_features, train_slots


def extra(ap):
    ap.add_argument("--extract", action="store_true", help="only extract and cache DINOv2 features")


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    if args.extract:
        extract_features(cfg.slots.frames_dir, cfg.slots.cache_dir, n_labels=2 + 6)
    else:
        print(train_slots(cfg))
