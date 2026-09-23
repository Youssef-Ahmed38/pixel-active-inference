"""Measure the throughput numbers the compute plan depends on, on whatever machine this runs on.

    python scripts/benchmark.py                 # everything
    python scripts/benchmark.py --skip-collect  # no parallel rendering test

Writes results/benchmark_<host>.json.
"""

import json
import platform
import shutil
import socket
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from _cli import parse

from pai.config import Config
from pai.data.collect import collect
from pai.envs import PandaEnv
from pai.models import JointDecoder
from pai.train.common import amp_dtype, autocast


def timeit(fn, n: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n


def bench_sim(cfg) -> dict:
    env = PandaEnv(cfg.env, disturbances=[])
    env.reset()
    m, d = env.model, env.data
    physics = 1.0 / timeit(lambda: mujoco.mj_step(m, d), 2000)
    render = 1.0 / timeit(env.render, 100)
    rng = np.random.default_rng(0)
    step = 1.0 / timeit(lambda: env.step(rng.uniform(-0.5, 0.5, env.n_active)), 100)
    gl = _gl_renderer()  # needs the renderer's GL context, so query before closing it
    env.close()
    return {"physics_steps_per_s": physics, "render_fps_256": render, "env_steps_per_s_with_render": step,
            "gl_renderer": gl}


def _gl_renderer() -> str:
    try:
        from OpenGL import GL

        return GL.glGetString(GL.GL_RENDERER).decode()
    except Exception as e:  # pragma: no cover - diagnostic only
        return f"unknown ({type(e).__name__})"


def bench_collect(cfg, workers_list) -> dict:
    out = {}
    for w in workers_list:
        tmp = Path(tempfile.mkdtemp(prefix="pai_bench_"))
        c = Config(cfg.to_dict())
        c.data.dir, c.data.n_samples, c.data.shard_size = str(tmp), 2000 * w, 1000
        t = time.time()
        collect(c, workers=w)
        out[f"workers_{w}"] = 2000 * w / (time.time() - t)  # includes process start-up
        shutil.rmtree(tmp, ignore_errors=True)
    return {"collect_samples_per_s": out}


def bench_decoder(cfg, device) -> dict:
    res = {}
    dtype = amp_dtype("auto", device)
    n_j = len(cfg.env.active_joints)
    for size in (128, 256):
        dec = JointDecoder([-3.0] * n_j, [3.0] * n_j, size, cfg.decoder.base_channels, cfg.decoder.max_channels).to(device)
        opt = torch.optim.AdamW(dec.parameters(), 1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
        bs = int(cfg.train.batch_size)
        q = torch.randn(bs, n_j, device=device)
        tgt = torch.rand(bs, 3, size, size, device=device)

        def train_step():
            with autocast(device, dtype):
                pred = dec(q)
            loss = F.mse_loss(pred.float(), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        n = 20 if device.type == "cuda" else 2
        res[f"train_img_per_s_{size}"] = bs / timeit(train_step, n, warmup=2)

        # One PixelAI inference step: forward + gradient w.r.t. the joint belief.
        s = torch.rand(1, 3, size, size, device=device)
        dec.eval()

        def aif_step():
            mu = torch.zeros(1, n_j, device=device, requires_grad=True)
            ((s - dec(mu)) ** 2).mean().backward()

        res[f"aif_step_ms_{size}"] = 1000 * timeit(aif_step, 50 if device.type == "cuda" else 5)
        res[f"params_{size}"] = sum(p.numel() for p in dec.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            train_step()
            res[f"train_peak_mem_gb_{size}"] = torch.cuda.max_memory_allocated() / 1e9
        del dec, opt
    res["amp"] = str(dtype)
    return res


def main():
    args, cfg = parse(__doc__, lambda ap: ap.add_argument("--skip-collect", action="store_true"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mujoco": mujoco.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    print("simulation...")
    report.update(bench_sim(cfg))
    if not args.skip_collect:
        print("parallel collection...")
        report.update(bench_collect(cfg, [1, 4, 8]))
    print("decoder...")
    report.update(bench_decoder(cfg, device))
    out = Path("results") / f"benchmark_{report['host']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
