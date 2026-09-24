"""Week 2 in one go (for a Kaggle session): collect frames -> DINOv2 features -> train object slots.

    python scripts/week2_slots.py --config configs/tabletop.yaml --episodes 300 --workers 4 \
        slots.frames_dir=/tmp/frames slots.cache_dir=/tmp/features slots.out_dir=runs/slots

With two GPUs, launch it with `torchrun --nproc_per_node=2` (the Kaggle notebook does this):
frame collection runs once on the CPUs, feature extraction splits the episodes across the GPUs,
and slot training is data-parallel.

Each step is skipped if its output already exists, so a restarted session continues where it
stopped. Keep the bulky frames and features on local disk (/tmp) and only the checkpoint and logs
in the saved output (runs/).
"""

from pathlib import Path

from _cli import parse

from pai.perception.collect import collect_frames
from pai.perception.train_slots import extract_features, train_slots
from pai.train.common import barrier, setup_runtime


def extra(ap):
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    sc = cfg.slots
    rt = setup_runtime()
    if rt.is_main:  # rendering is CPU-bound; the other GPU process waits
        collect_frames(cfg.env, sc.frames_dir, args.episodes, workers=args.workers, seed=int(cfg.seed) + 101,
                       camera=sc.camera, image_size=int(sc.image_size))
    barrier(rt)
    if not (Path(sc.cache_dir) / "episode.npy").exists():
        extract_features(sc.frames_dir, sc.cache_dir, n_labels=2 + 6, feat_dim=int(sc.get("feat_dim", 384)))
    print(train_slots(cfg))
