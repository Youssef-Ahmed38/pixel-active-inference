"""Camera frames + per-object segmentation masks + entity tokens, for learning object slots (week 2).

Frames are rendered at the DINOv2-friendly size (224 = 16 x 14-pixel patches) from the front
camera at every planner step of the same exploration/demonstration episodes the world model
uses. Masks label each pixel 0 (background/table), 1 (robot) or 2 + object index, and supervise
the slots at first (rule 2: privileged information first, then remove it and report the gap).

    <dir>/ep_00012.npz: images (T, 224, 224, 3) uint8, masks (T, 224, 224) uint8,
                        tokens (T, N, D) float32, actions (T-1, A) float32
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

IMAGE_SIZE = 224


def _worker(args):
    env_cfg, out_dir, episode_ids, camera, seed = args
    from pai.envs import TabletopEnv
    from pai.world.data import _episode

    cfg = type(env_cfg)({**env_cfg, "image_size": IMAGE_SIZE, "render_images": False})
    env = TabletopEnv(cfg, disturbances=[])
    for eid in episode_ids:
        path = out_dir / f"ep_{eid:05d}.npz"
        if path.exists():
            continue
        images, masks = [], []

        def record(env, obs):
            images.append(env.render(camera))
            masks.append(env.render_segmentation(camera).astype(np.uint8))

        rng = np.random.default_rng(seed * 1_000_003 + eid)
        toks, acts = _episode(env, rng, action_repeat=5, max_steps=600, noise=0.03, on_step=record)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, images=np.stack(images), masks=np.stack(masks), tokens=toks, actions=acts)
        tmp.replace(path)
        done = len(list(out_dir.glob("ep_*.npz")))
        if done % 10 == 0:  # progress, so a long silent collection does not look stuck
            print(f"  frames: {done} episodes written", flush=True)
    env.close()
    return len(episode_ids)


def collect_frames(env_cfg, out_dir: str | Path, n_episodes: int, workers: int = 8, camera: str = "front",
                   seed: int = 0) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(range(n_episodes))
    chunks = [ids[i::workers] for i in range(workers)]
    t0 = time.time()
    print(f"collecting camera frames: {n_episodes} episodes with {workers} workers (CPU) -> {out_dir}", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        pool.map(_worker, [(env_cfg, out_dir, c, camera, seed) for c in chunks])
    print(f"{n_episodes} episodes of frames in {time.time() - t0:.0f}s -> {out_dir}")
