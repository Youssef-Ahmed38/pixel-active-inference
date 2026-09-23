"""Extract frozen DINOv2 features once, then train the slot model on the cached features (week 2).

Cache layout in <cache_dir> (memory-mapped, so it never has to fit in RAM):
    feats.npy   (N, 256, 384) float16   DINOv2 ViT-S/14 patch features
    masks.npy   (N, 256, L)   float16   ground-truth label fractions per patch
    tokens.npy  (N, E, TOKEN_DIM) float32 privileged entity tokens (gripper + objects)
    episode.npy (N,)          int32     episode index of each frame (for held-out splits)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from pai.perception.features import DinoFeatures, masks_to_patches
from pai.perception.slots import SlotModel, match_slots, slot_losses
from pai.world.entities import TOKEN_DIM
from torch.nn.parallel import DistributedDataParallel as DDP

from pai.train.common import (JsonlLogger, Progress, amp_dtype, autocast, barrier, cleanup_runtime, git_commit,
                              save_checkpoint, setup_runtime)


def extract_features(frames_dir: str | Path, cache_dir: str | Path, n_labels: int, every: int = 2,
                     batch: int = 64) -> None:
    """Under torchrun each GPU encodes every world_size-th episode into the shared memmaps."""
    rt = setup_runtime()
    frames_dir, cache_dir = Path(frames_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in frames_dir.glob("ep_*.npz") if not f.name.endswith(".tmp.npz"))
    sizes = [len(range(0, np.load(f)["images"].shape[0], every)) for f in files]
    N = sum(sizes)
    first = np.load(files[0])
    E = first["tokens"].shape[1]
    shapes = {"feats": (np.float16, (N, 256, 384)), "masks": (np.float16, (N, 256, n_labels)),
              "tokens": (np.float32, (N, E, TOKEN_DIM))}
    if rt.is_main:  # one process creates the files, then every process opens them for writing
        for name, (dt, shape) in shapes.items():
            np.lib.format.open_memmap(cache_dir / f"{name}.npy", "w+", dt, shape).flush()
    barrier(rt)
    feats, masks, tokens = (np.load(cache_dir / f"{n}.npy", mmap_mode="r+") for n in ("feats", "masks", "tokens"))
    episode = np.zeros(N, np.int32)
    dino = DinoFeatures(device=rt.device)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    t0 = time.time()
    mine = sum(1 for e in range(len(files)) if e % rt.world_size == rt.rank)
    progress = Progress(mine, f"features (GPU {rt.rank})", every=10, unit=" ep")
    for e, f in enumerate(files):
        i = int(offsets[e])
        episode[i : i + sizes[e]] = e
        if e % rt.world_size != rt.rank:
            continue
        d = np.load(f)
        idx = np.arange(0, d["images"].shape[0], every)
        imgs = d["images"][idx]
        for s in range(0, len(idx), batch):
            fb = dino(imgs[s : s + batch]).reshape(-1, 256, 384)
            feats[i + s : i + s + len(fb)] = fb.cpu().numpy().astype(np.float16)
        masks[i : i + len(idx)] = masks_to_patches(d["masks"][idx], n_labels).reshape(len(idx), 256, n_labels)
        tokens[i : i + len(idx)] = d["tokens"][idx]
        progress.update()
    feats.flush(), masks.flush(), tokens.flush()
    barrier(rt)
    if rt.is_main:  # written last: its presence marks a complete cache
        np.save(cache_dir / "episode.npy", episode)
        print(f"{N} frames from {len(files)} episodes on {rt.world_size} GPU(s) in {time.time() - t0:.0f}s -> {cache_dir}")
    barrier(rt)


def train_slots(cfg, device: str | None = None) -> Path:
    """Under torchrun: data-parallel, each GPU draws its own batch and gradients are averaged,
    so every step sees world_size x batch_size frames."""
    sc = cfg.slots
    rt = setup_runtime()
    device = torch.device(device) if device else rt.device
    torch.manual_seed(cfg.seed + rt.rank)
    cache = Path(sc.cache_dir)
    feats = np.load(cache / "feats.npy", mmap_mode="r")
    masks = np.load(cache / "masks.npy", mmap_mode="r")
    tokens = np.load(cache / "tokens.npy", mmap_mode="r")
    episode = np.load(cache / "episode.npy")
    n_ep = int(episode.max()) + 1
    split_rng = np.random.default_rng(cfg.seed)  # same held-out episodes on every GPU
    val_eps = split_rng.choice(n_ep, max(1, n_ep // 20), replace=False)  # held-out episodes, not frames
    rng = np.random.default_rng(cfg.seed + 1 + rt.rank)  # different batches on every GPU
    val_idx, train_idx = np.flatnonzero(np.isin(episode, val_eps)), np.flatnonzero(~np.isin(episode, val_eps))
    n_labels = masks.shape[-1]

    model = SlotModel(n_slots=sc.n_slots, dim=sc.dim, iters=sc.iters).to(device)
    gpu_ids = [rt.local_rank] if device.type == "cuda" else None  # None: CPU processes (gloo tests)
    net = DDP(model, device_ids=gpu_ids) if rt.distributed else model
    opt = torch.optim.AdamW(model.parameters(), lr=sc.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=sc.lr, total_steps=sc.steps, pct_start=0.05)
    dtype = amp_dtype("auto", device)
    tok_mean = torch.as_tensor(np.asarray(tokens[train_idx[:20000]]).reshape(-1, TOKEN_DIM).mean(0), device=device)
    tok_std = torch.as_tensor(np.asarray(tokens[train_idx[:20000]]).reshape(-1, TOKEN_DIM).std(0) + 1e-3, device=device)
    out_dir = Path(sc.out_dir)
    logger = JsonlLogger(out_dir / "log.jsonl", enabled=rt.is_main)
    logger.log(event="start", frames=len(train_idx), val_frames=len(val_idx), gpus=rt.world_size,
               commit=git_commit(), params=sum(p.numel() for p in model.parameters()))

    def batch(idx):
        # sorted reads are faster on memmaps; tiny validation sets are sampled with replacement
        b = np.sort(rng.choice(idx, sc.batch_size, replace=len(idx) < sc.batch_size))
        f = torch.as_tensor(np.asarray(feats[b]), device=device).float()
        m = torch.as_tensor(np.asarray(masks[b]), device=device).float()
        t = (torch.as_tensor(np.asarray(tokens[b]), device=device) - tok_mean) / tok_std
        return f, m, t

    labels = slice(2, n_labels)
    progress = Progress(sc.steps, f"slots ({rt.world_size} GPU)", every=100, unit=" steps", enabled=rt.is_main)
    for step in range(1, sc.steps + 1):
        f, m, t = batch(train_idx)
        with autocast(device, dtype):
            out = net(f)
        out = {k: v.float() for k, v in out.items()}
        losses = slot_losses(out, f, m, t, labels, mask_weight=sc.mask_weight, token_weight=sc.token_weight)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        progress.update(extra=f"loss {float(losses['total'].detach()):.3f}")
        if rt.is_main and (step % sc.log_every == 0 or step == sc.steps):
            logger.log(step=step, **{k: round(float(v.detach()), 4) for k, v in losses.items()},
                       **evaluate_slots(model, batch, val_idx, labels, tok_std))
    path = out_dir / "slots.pt"
    if rt.is_main:
        save_checkpoint(path, {"model": model.state_dict(), "model_config": model.config(), "tok_mean": tok_mean.cpu(),
                               "tok_std": tok_std.cpu(), "cfg": cfg.to_dict(), "commit": git_commit()})
    barrier(rt)
    cleanup_runtime(rt)
    return path


@torch.no_grad()
def evaluate_slots(model, batch_fn, val_idx, labels: slice, tok_std, n_batches: int = 4) -> dict:
    """Object mIoU of matched slots, and position error (mm) of the slot token readout."""
    model.eval()
    ious, pos_err = [], []
    for _ in range(n_batches):
        f, m, t = batch_fn(val_idx)
        out = model(f)
        pairs = match_slots(out["alpha"], m)
        a = out["alpha"]
        for b, (ks, ls) in enumerate(pairs):
            for k, l in zip(ks, ls):
                if labels.start <= l < labels.stop:
                    hard_slot = a[b].argmax(0) == k
                    hard_gt = m[b].argmax(-1) == l
                    union = (hard_slot | hard_gt).sum()
                    if union > 0:
                        ious.append(float((hard_slot & hard_gt).sum() / union))
                    e = l - labels.start + 1
                    pos_err.append(float(((out["tokens"][b, k, :3] - t[b, e, :3]) * tok_std[:3]).norm()))
    model.train()
    return {"val_object_miou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "val_token_pos_err_mm": round(1000 * float(np.mean(pos_err)), 2) if pos_err else -1.0}
