"""Shared training plumbing: device/DDP setup, mixed precision, atomic checkpoints, logging.

Works unchanged on a laptop (1 GPU or CPU), Colab (1x T4) and Kaggle (2x T4 via
`torchrun --nproc_per_node=2`). Every run resumes from <out_dir>/ckpt_last.pt if present,
so a Kaggle/Colab session that dies can simply be restarted.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class Runtime:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def setup_runtime() -> Runtime:
    """Single process, or one process per GPU under torchrun. Safe to call more than once."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        # nccl on GPUs; PAI_DIST_BACKEND=gloo runs the same multi-process code on CPUs (for tests)
        backend = os.environ.get("PAI_DIST_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
        if backend == "nccl":
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend)
        device = torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
        return Runtime(device, dist.get_rank(), dist.get_world_size(), local_rank)
    return Runtime(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def cleanup_runtime(rt: Runtime) -> None:
    if rt.distributed and dist.is_initialized():
        dist.destroy_process_group()


def barrier(rt: Runtime) -> None:
    if rt.distributed:
        dist.barrier()


def amp_dtype(setting: str, device: torch.device) -> torch.dtype | None:
    """'auto': bf16 where supported (Ampere+), fp16 otherwise (T4 = Turing), None on CPU."""
    if device.type != "cuda" or setting in (False, "off", "none"):
        return None
    if setting == "fp16":
        return torch.float16
    if setting == "bf16":
        return torch.bfloat16
    try:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # older torch without the kwarg
        bf16 = torch.cuda.get_device_capability(device)[0] >= 8
    return torch.bfloat16 if bf16 else torch.float16


def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device.type, dtype=dtype) if dtype is not None else nullcontext()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(out_dir: Path, resume_from: str | None = None) -> dict | None:
    """Latest checkpoint in out_dir, else in resume_from (e.g. a previous Kaggle version's output)."""
    for d in [out_dir, Path(resume_from) if resume_from else None]:
        if d is not None and (d / "ckpt_last.pt").exists():
            return torch.load(d / "ckpt_last.pt", map_location="cpu", weights_only=False)
    return None


class JsonlLogger:
    def __init__(self, path: Path, enabled: bool = True):
        self.path, self.enabled = path, enabled
        self.t0 = time.time()
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **row) -> None:
        if not self.enabled:
            return
        row = {"time": round(time.time() - self.t0, 2), **row}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
