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

from pai.perception.features import PATCH, DinoFeatures, masks_to_patches
from pai.perception.slots import SlotModel, assign_slots, slot_losses
from pai.world.entities import TOKEN_DIM
from torch.nn.parallel import DistributedDataParallel as DDP

from pai.train.common import (JsonlLogger, Progress, amp_dtype, autocast, barrier, cleanup_runtime, git_commit,
                              save_checkpoint, setup_runtime)


def fit_pca(dino, files: list[Path], dim: int, n_frames: int = 400, per_frame: int = 128, seed: int = 0) -> dict:
    """PCA of DINOv2 patch features on the first episodes: keeps the cache small enough for GPU
    memory at 1024 patches per frame. Returns {"mean": (C,), "components": (C, dim)}."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in files:
        imgs = np.load(f)["images"][::4]
        for s in range(0, len(imgs), 32):
            fb = dino(imgs[s : s + 32]).float().reshape(-1, dino.model.embed_dim).cpu().numpy()
            rows.append(fb[rng.choice(len(fb), min(len(fb), per_frame * 32), replace=False)])
        if sum(len(r) for r in rows) >= n_frames * per_frame:
            break
    x = np.concatenate(rows).astype(np.float64)
    mean = x.mean(0)
    _, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
    kept = float((sv[:dim] ** 2).sum() / (sv**2).sum())
    return {"mean": mean.astype(np.float32), "components": vt[:dim].T.astype(np.float32), "variance_kept": kept}


def extract_features(frames_dir: str | Path, cache_dir: str | Path, n_labels: int, every: int = 2,
                     batch: int = 32, feat_dim: int = 384, pca_from: str | Path | None = None) -> None:
    """Under torchrun each GPU encodes every world_size-th episode into the shared memmaps.
    feat_dim < 384: features are compressed by a PCA fitted on the first episodes (saved as pca.npz,
    and used again at run time by pai.perception.slot_tokens). pca_from: reuse that pca.npz instead,
    so that a new cache can be trained on together with an older one."""
    rt = setup_runtime()
    frames_dir, cache_dir = Path(frames_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in frames_dir.glob("ep_*.npz") if not f.name.endswith(".tmp.npz"))
    sizes = [len(range(0, np.load(f)["images"].shape[0], every)) for f in files]
    N = sum(sizes)
    first = np.load(files[0])
    E = first["tokens"].shape[1]
    P = (first["images"].shape[1] // PATCH) * (first["images"].shape[2] // PATCH)
    dino = DinoFeatures(device=rt.device)
    C = dino.model.embed_dim if feat_dim >= dino.model.embed_dim else feat_dim
    if C < dino.model.embed_dim and rt.is_main:
        pca = dict(np.load(pca_from)) if pca_from else fit_pca(dino, files, C)
        np.savez(cache_dir / "pca.npz", **pca)
        print(f"PCA {dino.model.embed_dim} -> {C} dims keeps {100 * pca['variance_kept']:.1f}% of the variance", flush=True)
    shapes = {"feats": (np.float16, (N, P, C)), "masks": (np.float16, (N, P, n_labels)),
              "tokens": (np.float32, (N, E, TOKEN_DIM))}
    if rt.is_main:  # one process creates the files, then every process opens them for writing
        for name, (dt, shape) in shapes.items():
            np.lib.format.open_memmap(cache_dir / f"{name}.npy", "w+", dt, shape).flush()
    barrier(rt)
    project = None
    if C < dino.model.embed_dim:
        pca = np.load(cache_dir / "pca.npz")
        mean_t = torch.as_tensor(pca["mean"], device=rt.device)
        comp_t = torch.as_tensor(pca["components"], device=rt.device)
        project = lambda f: (f.float() - mean_t) @ comp_t  # noqa: E731
    feats, masks, tokens = (np.load(cache_dir / f"{n}.npy", mmap_mode="r+") for n in ("feats", "masks", "tokens"))
    episode = np.zeros(N, np.int32)
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
            fb = dino(imgs[s : s + batch]).reshape(-1, P, dino.model.embed_dim)
            if project is not None:
                fb = project(fb)
            feats[i + s : i + s + len(fb)] = fb.cpu().numpy().astype(np.float16)
        masks[i : i + len(idx)] = masks_to_patches(d["masks"][idx], n_labels).reshape(len(idx), P, n_labels)
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
    n_val = max(1, int(round(n_ep * float(sc.get("val_fraction", 0.05)))))
    val_eps = split_rng.choice(n_ep, n_val, replace=False)  # held-out episodes, not frames
    if sc.get("val_episodes"):  # an explicit held-out set (e.g. to compare runs on different caches)
        val_eps = np.asarray(list(sc.val_episodes), int)
    rng = np.random.default_rng(cfg.seed + 1 + rt.rank)  # different batches on every GPU
    val_idx, train_idx = np.flatnonzero(np.isin(episode, val_eps)), np.flatnonzero(~np.isin(episode, val_eps))
    n_labels = masks.shape[-1]

    model = SlotModel(feat_dim=feats.shape[-1], n_patches=feats.shape[1], n_slots=sc.n_slots, dim=sc.dim,
                      iters=sc.iters).to(device)
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

    # The slot model is small, so reading random frames from disk on the CPU (Kaggle has 2 cores)
    # is slower than the GPU's work and the GPU waits. When the whole cache fits, it is copied to
    # GPU memory once (features in fp16: ~3.4 GB for 17k frames) and batches never touch the CPU.
    on_gpu = _fits_on_gpu(sc.get("gpu_cache", "auto"), device, feats, masks, tokens)
    if on_gpu:
        feats, masks, tokens = _to_gpu(feats, device), _to_gpu(masks, device), _to_gpu(tokens, device, half=False)
        logger.log(event="gpu_cache", gb=round(sum(a.numel() * a.element_size() for a in (feats, masks, tokens)) / 1e9, 2))
        if rt.is_main:
            print("feature cache copied to GPU memory", flush=True)

    def batch(idx, b=None):
        # sorted reads are faster on memmaps; tiny validation sets are sampled with replacement
        if b is None:
            b = np.sort(rng.choice(idx, sc.batch_size, replace=len(idx) < sc.batch_size))
        if on_gpu:
            bi = torch.as_tensor(b, device=device)
            return feats[bi].float(), masks[bi].float(), (tokens[bi].float() - tok_mean) / tok_std
        f = torch.as_tensor(np.asarray(feats[b]), device=device).float()
        m = torch.as_tensor(np.asarray(masks[b]), device=device).float()
        t = (torch.as_tensor(np.asarray(tokens[b]), device=device) - tok_mean) / tok_std
        return f, m, t

    labels = slice(2, n_labels)
    assignment = str(sc.get("assignment", "hungarian"))
    progress = Progress(sc.steps, f"slots ({rt.world_size} GPU)", every=100, unit=" steps", enabled=rt.is_main)
    for step in range(1, sc.steps + 1):
        f, m, t = batch(train_idx)
        with autocast(device, dtype):
            out = net(f)
        out = {k: v.float() for k, v in out.items()}
        losses = slot_losses(out, f, m, t, labels, mask_weight=sc.mask_weight, token_weight=sc.token_weight,
                             recon_weight=float(sc.get("recon_weight", 1.0)), attn_weight=float(sc.get("attn_weight", 1.0)),
                             pos_weight=float(sc.get("pos_weight", 1.0)), assignment=assignment)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        progress.update(extra=f"loss {float(losses['total'].detach()):.3f}")
        if rt.is_main and (step % sc.log_every == 0 or step == sc.steps):
            final = step == sc.steps  # the last evaluation covers every held-out frame
            logger.log(step=step, **{k: round(float(v.detach()), 4) for k, v in losses.items()},
                       **evaluate_slots(model, batch, val_idx, labels, tok_std, n_frames=None if final else 256,
                                        batch_size=sc.batch_size, per_label=final, assignment=assignment))
    path = out_dir / "slots.pt"
    if rt.is_main:
        pca = dict(np.load(cache / "pca.npz")) if (cache / "pca.npz").exists() else None
        save_checkpoint(path, {"model": model.state_dict(), "model_config": model.config(), "tok_mean": tok_mean.cpu(),
                               "tok_std": tok_std.cpu(), "cfg": cfg.to_dict(), "commit": git_commit(),
                               "camera": sc.get("camera", "front"), "image_size": int(sc.get("image_size", 224)),
                               "assignment": assignment, "object_labels": [labels.start, labels.stop],
                               "pca": pca})
    barrier(rt)
    cleanup_runtime(rt)
    return path


def _fits_on_gpu(mode, device, *arrays) -> bool:
    """mode: True, False or "auto" (use it when the cache takes under half of the free GPU memory)."""
    if device.type != "cuda" or mode is False or str(mode).lower() == "false":
        return False
    if mode is True or str(mode).lower() == "true":
        return True
    need = sum(a.nbytes for a in arrays)  # upper bound: features and masks are stored as fp16
    free, _ = torch.cuda.mem_get_info(device)
    return need < 0.5 * free


def _to_gpu(a: np.ndarray, device, half: bool = True, chunk: int = 1024) -> torch.Tensor:
    """Copy a memmap to the GPU in chunks, never holding it all in CPU memory. half: floats as fp16
    (features and masks; tokens keep float32, their positions need the precision)."""
    dtype = torch.from_numpy(np.asarray(a[:1])).dtype
    if half and np.issubdtype(a.dtype, np.floating):
        dtype = torch.float16
    out = torch.empty(a.shape, dtype=dtype, device=device)
    for s in range(0, len(a), chunk):
        out[s : s + chunk] = torch.from_numpy(np.asarray(a[s : s + chunk])).to(device, dtype)
    return out


@torch.no_grad()
def evaluate_slots(model, batch_fn, val_idx, labels: slice, tok_std, n_frames: int | None = 256,
                   batch_size: int = 32, per_label: bool = False, assignment: str = "hungarian") -> dict:
    """Object IoU of matched slots, soft (fractional masks: objects are often smaller than a patch,
    so a majority vote per patch would miss them entirely), and position error (mm) of the readout,
    on visible objects of held-out episodes. n_frames: an evenly spaced subset (None: all).
    per_label: also each object label's IoU and error (keys ..._l2, ..._l3, ...)."""
    model.eval()
    idx = np.asarray(val_idx)
    if n_frames is not None and len(idx) > n_frames:
        idx = idx[np.linspace(0, len(idx) - 1, n_frames).astype(int)]
    ious, pos_err, lab = [], [], []
    for s in range(0, len(idx), batch_size):
        f, m, t = batch_fn(val_idx, idx[s : s + batch_size])
        out = model(f)
        a = out["alpha"]
        perm = assign_slots(a, m, assignment).cpu().numpy()  # the same assignment as in training
        for b in range(len(perm)):
            for l, k in enumerate(perm[b]):
                if labels.start <= l < labels.stop and m[b, :, l].sum() > 0.5:  # visible objects only
                    inter = (a[b, k] * m[b, :, l]).sum()
                    ious.append(float(inter / (a[b, k].sum() + m[b, :, l].sum() - inter + 1e-8)))
                    e = l - labels.start + 1
                    pos_err.append(float(((out["tokens"][b, k, :3] - t[b, e, :3]) * tok_std[:3]).norm()))
                    lab.append(l)
    model.train()
    ious, pos_err, lab = np.asarray(ious), np.asarray(pos_err), np.asarray(lab)
    res = {"val_object_miou": round(float(ious.mean()), 4) if len(ious) else 0.0,
           "val_token_pos_err_mm": round(1000 * float(pos_err.mean()), 2) if len(pos_err) else -1.0,
           "val_token_pos_median_mm": round(1000 * float(np.median(pos_err)), 2) if len(pos_err) else -1.0}
    if per_label:
        for l in range(labels.start, labels.stop):
            sel = lab == l
            if sel.any():
                res[f"val_miou_l{l}"] = round(float(ious[sel].mean()), 4)
                res[f"val_pos_err_mm_l{l}"] = round(1000 * float(pos_err[sel].mean()), 2)
    return res
