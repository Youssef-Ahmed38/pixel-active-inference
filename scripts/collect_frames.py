"""Collect camera frames + segmentation masks + tokens for object-slot training.

    python scripts/collect_frames.py --config configs/tabletop.yaml --episodes 300
"""

from _cli import parse

from pai.perception.collect import collect_frames


def extra(ap):
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default="data/slice_frames")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--camera", default=None, help="default: slots.camera from the config")
    ap.add_argument("--image-size", type=int, default=None, help="default: slots.image_size from the config")


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    collect_frames(cfg.env, args.out, args.episodes, workers=args.workers, camera=args.camera or cfg.slots.camera,
                   seed=int(cfg.seed) + 101, image_size=args.image_size or int(cfg.slots.image_size))
