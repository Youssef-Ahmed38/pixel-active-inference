"""Train the PixelAI decoder g(q) -> image with MSE, resumable, single- or multi-GPU."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from pai.data import JointImageDataset
from pai.models import build_decoder
from pai.train.common import (
    JsonlLogger, amp_dtype, autocast, cleanup_runtime, git_commit, load_checkpoint, save_checkpoint, setup_runtime,
)


def _images(batch_uint8: torch.Tensor, size: int, device) -> torch.Tensor:
    x = batch_uint8.to(device, non_blocking=True).float() / 255.0
    return x if x.shape[-1] == size else F.interpolate(x, size=(size, size), mode="area")


def _lr(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


@torch.no_grad()
def evaluate(model, loader, size, device, dtype, max_batches: int = 50) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (q, img) in enumerate(loader):
        if i >= max_batches:
            break
        target = _images(img, size, device)
        with autocast(device, dtype):
            pred = model(q.to(device))
        total += F.mse_loss(pred.float(), target, reduction="sum").item()
        n += target.numel()
    model.train()
    return total / max(1, n)


def train(cfg, resume_from: str | None = None) -> Path:
    rt = setup_runtime()
    tc = cfg.train
    out_dir = Path(tc.out_dir)
    torch.manual_seed(cfg.seed + rt.rank)

    full = JointImageDataset(cfg.data.dir)
    train_ds, val_ds = full.split(tc.val_fraction, seed=cfg.seed)
    meta = full.meta
    size = int(cfg.decoder.image_size)
    if meta["image_size"] < size:
        raise ValueError(f"dataset images are {meta['image_size']}px but decoder wants {size}px")

    model = build_decoder(meta["joint_low"], meta["joint_high"], cfg.decoder).to(rt.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    dtype = amp_dtype(tc.amp, rt.device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    step = 0
    ckpt = load_checkpoint(out_dir, resume_from)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        if rt.is_main:
            print(f"resumed from step {step}")

    net = DDP(model, device_ids=[rt.local_rank]) if rt.distributed else model
    sampler = DistributedSampler(train_ds, rt.world_size, rt.rank, seed=cfg.seed) if rt.distributed else None
    loader_kw = dict(num_workers=tc.num_workers, pin_memory=rt.device.type == "cuda", persistent_workers=tc.num_workers > 0)
    train_loader = DataLoader(train_ds, tc.batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, tc.batch_size, shuffle=False, num_workers=0)

    logger = JsonlLogger(out_dir / "log.jsonl", rt.is_main)
    if rt.is_main:
        n_params = sum(p.numel() for p in model.parameters())
        logger.log(event="start", step=step, params=n_params, world_size=rt.world_size,
                   amp=str(dtype), commit=git_commit(), n_train=len(train_ds))

    def state():
        return {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                "decoder_config": model.config(), "cfg": cfg.to_dict(), "commit": git_commit()}

    net.train()
    epoch, t_last, seen = step // max(1, len(train_loader)), time.time(), 0
    while step < tc.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for q, img in train_loader:
            if step >= tc.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = _lr(step, tc)
            target = _images(img, size, rt.device)
            with autocast(rt.device, dtype):
                pred = net(q.to(rt.device, non_blocking=True))
            loss = F.mse_loss(pred.float(), target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            seen += q.shape[0] * rt.world_size

            if rt.is_main and step % tc.log_every == 0:
                dt = time.time() - t_last
                logger.log(step=step, loss=loss.item(), lr=_lr(step, tc), img_per_s=seen / dt)
                t_last, seen = time.time(), 0
            if rt.is_main and step % tc.ckpt_every == 0:  # checkpoint before eval, so an eval crash loses nothing
                save_checkpoint(out_dir / "ckpt_last.pt", state())
            if rt.is_main and step % tc.eval_every == 0:
                logger.log(step=step, val_mse=evaluate(model, val_loader, size, rt.device, dtype))
        epoch += 1

    if rt.is_main:
        logger.log(step=step, val_mse=evaluate(model, val_loader, size, rt.device, dtype, max_batches=10**9))
        save_checkpoint(out_dir / "ckpt_last.pt", state())
        save_checkpoint(out_dir / "decoder.pt", {"model": model.state_dict(), "decoder_config": model.config(),
                                                 "step": step, "commit": git_commit()})
        _save_samples(model, val_ds, size, rt.device, out_dir / "samples.png")
    cleanup_runtime(rt)
    return out_dir / "decoder.pt"


@torch.no_grad()
def _save_samples(model, ds, size, device, path: Path, n: int = 8) -> None:
    import imageio

    model.eval()
    q, img = zip(*[ds[i] for i in range(min(n, len(ds)))])
    pred = model(torch.stack(q).to(device)).float()
    real = _images(torch.stack(img), size, device)
    grid = torch.cat([torch.cat(list(real), -1), torch.cat(list(pred), -1)], -2)
    imageio.imwrite(path, (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
