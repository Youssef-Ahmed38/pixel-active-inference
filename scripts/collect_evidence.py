"""Collect many real agent episodes as evidence for the thinker, on every GPU (Kaggle job C).

    python scripts/collect_evidence.py --config configs/tabletop.yaml --episodes 200          # 1 process
    torchrun --nproc_per_node=2 scripts/collect_evidence.py --config configs/tabletop.yaml \
        --episodes 200 slice.world_model=/kaggle/input/pai-models/world_model.pt              # 2x T4
    python scripts/collect_evidence.py --merge runs/evidence                                  # shards -> one file

The agent is the one of scripts/slice_eval.py (same run_episode, same disturbances, same teacher);
only the bookkeeping differs, so a Kaggle session can produce thousands of episodes:

- Calibration: rank 0 calibrates surprise on clean episodes (seeds 5000+, as slice_eval), saves it
  to <out>/calibration.pkl, the other ranks wait and load it. Every rank then uses the same one.
- Episodes: `--episodes` per condition, interleaved (global id g = i * n_conditions + c, so every
  prefix is balanced). Env seed 10000 + g (`--episode-seed`), disjoint from slice_eval's evaluation
  seeds (1000+) and calibration seeds (5000+); the disturbance parameters come from an rng seeded
  by (seed, env seed), so an episode is the same whichever rank or session runs it.
- Shards: the global list is cut into chunks of `--chunk` episodes; chunk k goes to rank
  k % world_size and is written atomically to <out>/shard_<k>.npz. Resuming = skipping chunks whose
  shard exists, so a restarted (or re-sized) job only runs what is missing.
- Format: every shard and the merged file follow the evidence contract of pai/thinker/real.py
  (keys of slice_eval's <tag>_evidence.npz). `episode` indexes the file's own `labels` (0..E-1, as
  the loader requires); the globally unique id and env seed are kept in `episode_id` and `seed`.
  Extra per-episode keys: `success`, `inferred`.

At the end rank 0 merges every shard into <out>/<tag>_evidence.npz, the input of
scripts/train_thinker_real.py.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/: _cli and slice_eval

from _cli import parse  # noqa: E402
from slice_eval import CONDITIONS, disturbance_for  # noqa: E402

from pai.agents.slice_agent import run_episode  # noqa: E402
from pai.causes.inference import CAUSES, CHANNELS, calibrate  # noqa: E402
from pai.envs import TabletopEnv  # noqa: E402
from pai.train.common import Progress, barrier, cleanup_runtime, setup_runtime  # noqa: E402
from pai.world.model import load_world_model  # noqa: E402

CALIBRATION_SEED = 5000   # as slice_eval
EPISODE_SEED = 10000      # default first env seed; slice_eval evaluates on 1000 + i


def episode_plan(conditions: list[str], per_condition: int) -> list[tuple[int, str]]:
    """(global id, condition), interleaved so that any prefix has every condition."""
    return [(i * len(conditions) + c, cond) for i in range(per_condition) for c, cond in enumerate(conditions)]


def chunks_for_rank(n_episodes: int, chunk: int, rank: int, world: int) -> list[int]:
    return [k for k in range((n_episodes + chunk - 1) // chunk) if k % world == rank]


def shard_path(out: Path, k: int) -> Path:
    return out / f"shard_{k:05d}.npz"


def pack(episodes: list[dict]) -> dict[str, np.ndarray]:
    """Per-episode dicts -> evidence arrays (episode renumbered 0..E-1 in the given order)."""
    return {"z": np.concatenate([e["z"] for e in episodes]).astype(np.float32),
            "holding": np.concatenate([e["holding"] for e in episodes]).astype(bool),
            "phase": np.concatenate([e["phase"] for e in episodes]).astype(np.int64),
            "episode": np.concatenate([np.full(len(e["z"]), i, np.int64) for i, e in enumerate(episodes)]),
            "labels": np.array([e["label"] for e in episodes]),
            "teacher": np.array([e["teacher"] for e in episodes], np.float32),
            "causes": np.array(CAUSES), "channels": np.array(CHANNELS),
            "onset": np.array([e["onset"] for e in episodes], np.int64),
            "episode_id": np.array([e["episode_id"] for e in episodes], np.int64),
            "seed": np.array([e["seed"] for e in episodes], np.int64),
            "success": np.array([e["success"] for e in episodes], bool),
            "inferred": np.array([e["inferred"] for e in episodes])}


def unpack(d: dict[str, np.ndarray]) -> list[dict]:
    """Inverse of `pack` (for merging)."""
    out = []
    for i in range(len(d["labels"])):
        sel = d["episode"] == i
        out.append({"z": d["z"][sel], "holding": d["holding"][sel], "phase": d["phase"][sel],
                    "label": str(d["labels"][i]), "teacher": d["teacher"][i], "onset": int(d["onset"][i]),
                    "episode_id": int(d["episode_id"][i]), "seed": int(d["seed"][i]),
                    "success": bool(d["success"][i]), "inferred": str(d["inferred"][i])})
    return out


def save_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez appends .npz unless it is there
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def merge(shard_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Concatenate every shard in shard_dir into one evidence file, ordered by global episode id."""
    shard_dir = Path(shard_dir)
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz in {shard_dir}")
    episodes = []
    for p in shards:
        with np.load(p, allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        if tuple(d["causes"].tolist()) != CAUSES or tuple(d["channels"].tolist()) != CHANNELS:
            raise ValueError(f"{p}: causes/channels differ from pai.causes.inference")
        episodes += unpack(d)
    episodes.sort(key=lambda e: e["episode_id"])
    ids = [e["episode_id"] for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate episode ids across shards in {shard_dir}")
    out_path = Path(out_path) if out_path else shard_dir / "collected_evidence.npz"
    save_atomic(out_path, pack(episodes))
    return out_path


def run_chunk(env, wm, cfg, calibration, plan: list[tuple[int, str]], device, progress: Progress,
              episode_seed: int = EPISODE_SEED) -> list[dict]:
    episodes = []
    for g, cond in plan:
        seed = episode_seed + g
        rng = np.random.default_rng((int(cfg.seed), seed))  # per episode: independent of rank and order
        torch.manual_seed(seed)
        evidence = []
        rec = run_episode(env, wm, cfg, g, seed=seed, disturbances=disturbance_for(cond, rng), device=device,
                          calibration=calibration, evidence=evidence)
        progress.update(extra=f"ep {g} {cond} success={rec.success} inferred={rec.inferred_cause}")
        if not evidence:  # the goal held before the first step: nothing for the thinker to see
            print(f"  ep {g} ({cond}): no planner steps, skipped", flush=True)
            continue
        post = rec.cause_posterior or {"none": 1.0}
        episodes.append({"z": np.stack([e[0] for e in evidence]), "holding": np.array([e[1] for e in evidence]),
                         "phase": np.array([e[2] for e in evidence]), "label": cond,
                         "teacher": [post.get(c, 0.0) for c in CAUSES],
                         "onset": rec.true_disturbances[0]["t"] // int(cfg.slice.action_repeat)
                         if rec.true_disturbances else -1,
                         "episode_id": g, "seed": seed, "success": bool(rec.success),
                         "inferred": str(rec.inferred_cause)})
    return episodes


def get_calibration(rt, env, wm, cfg, out: Path, n: int, device):
    """Rank 0 calibrates (or reuses a saved calibration); the other ranks load its file."""
    path = out / "calibration.pkl"
    if rt.is_main and not path.exists():
        raw = []
        for i in range(n):
            torch.manual_seed(CALIBRATION_SEED + i)
            run_episode(env, wm, cfg, -1, seed=CALIBRATION_SEED + i, device=device, raw=raw)
        cal = calibrate(np.stack([r[0] for r in raw]), np.stack([r[1] for r in raw]), [r[2] for r in raw])
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(cal))
        os.replace(tmp, path)
        print(f"calibration: threshold {cal.threshold:.1f} from {len(raw)} clean steps ({n} episodes)", flush=True)
    barrier(rt)
    return pickle.loads(path.read_bytes())


def extra(ap):
    ap.add_argument("--conditions", type=lambda s: s.split(","), default=list(CONDITIONS),
                    help="comma-separated, e.g. none,push")
    ap.add_argument("--episodes", type=int, default=200, help="episodes per condition (all ranks together)")
    ap.add_argument("--calibration-episodes", type=int, default=8)
    ap.add_argument("--episode-seed", type=int, default=EPISODE_SEED,
                    help="env seed of episode 0; give a second account another range (e.g. 20000)")
    ap.add_argument("--chunk", type=int, default=50, help="episodes per shard (the unit of resuming)")
    ap.add_argument("--out", default="runs/evidence", help="shard folder")
    ap.add_argument("--tag", default="collected", help="merged file: <out>/<tag>_evidence.npz")
    ap.add_argument("--merge", default=None, metavar="DIR", help="only merge the shards in DIR, then exit")


def main() -> None:
    args, cfg = parse(__doc__, extra)
    if args.merge:
        print(f"wrote {merge(args.merge, Path(args.merge) / f'{args.tag}_evidence.npz')}")
        return
    rt = setup_runtime()
    device = str(rt.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = episode_plan(args.conditions, args.episodes)
    meta = {"conditions": args.conditions, "episodes_per_condition": args.episodes, "chunk": args.chunk,
            "seed": int(cfg.seed), "episode_seed": args.episode_seed, "calibration_episodes": args.calibration_episodes,
            "world_model": str(cfg.slice.world_model)}
    meta_path = out / "meta.json"
    if meta_path.exists():  # resuming into a folder made with other settings would mix episodes
        old = json.loads(meta_path.read_text())
        diff = {k: (old.get(k), v) for k, v in meta.items() if k != "world_model" and old.get(k) != v}
        if diff:  # every rank checks, so none is left waiting at a barrier
            raise SystemExit(f"{out} was collected with other settings {diff}; use another --out")
    barrier(rt)
    if rt.is_main:
        meta_path.write_text(json.dumps(meta, indent=1))

    wm = load_world_model(cfg.slice.world_model, device)
    env = TabletopEnv(type(cfg.env)({**cfg.env, "render_images": False}), disturbances=[])
    calibration = get_calibration(rt, env, wm, cfg, out, args.calibration_episodes, device)

    mine = chunks_for_rank(len(plan), args.chunk, rt.rank, rt.world_size)
    todo = [k for k in mine if not shard_path(out, k).exists()]
    n_todo = sum(len(plan[k * args.chunk:(k + 1) * args.chunk]) for k in todo)
    print(f"rank {rt.rank}/{rt.world_size} on {device}: {len(mine)} shards, {len(mine) - len(todo)} done, "
          f"{n_todo} episodes to run", flush=True)
    progress = Progress(n_todo, f"rank {rt.rank} episodes", every=1, unit=" ep")
    t0 = time.time()
    for k in todo:
        eps = run_chunk(env, wm, cfg, calibration, plan[k * args.chunk:(k + 1) * args.chunk], device, progress,
                        args.episode_seed)
        if eps:
            save_atomic(shard_path(out, k), pack(eps))
        print(f"  rank {rt.rank}: wrote {shard_path(out, k).name} ({len(eps)} episodes)", flush=True)
    if n_todo:
        dt = time.time() - t0
        print(f"rank {rt.rank}: {n_todo} episodes in {dt / 60:.1f} min = {3600 * n_todo / dt:.0f} episodes/hour",
              flush=True)

    barrier(rt)
    if rt.is_main:
        merged = merge(out, out / f"{args.tag}_evidence.npz")
        with np.load(merged) as f:
            labels, success = f["labels"], f["success"]
        print(f"merged {len(labels)} episodes into {merged}; success {success.mean():.0%}; "
              f"labels {dict(zip(*np.unique(labels, return_counts=True)))}", flush=True)
    cleanup_runtime(rt)


if __name__ == "__main__":
    main()
