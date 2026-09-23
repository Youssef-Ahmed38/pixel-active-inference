"""Collect planner-rate transitions for the entity world model (vertical slice).

    python scripts/slice_collect.py --config configs/tabletop.yaml --episodes 1200
"""

from _cli import parse

from pai.world.data import collect_transitions


def extra(ap):
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--out", default="data/slice_transitions")
    ap.add_argument("--workers", type=int, default=8)


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    collect_transitions(cfg.env, args.out, args.episodes, workers=args.workers, seed=int(cfg.seed))
