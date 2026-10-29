"""Train the G1 to walk on the GPU with MuJoCo Playground (MJX physics + Brax PPO), in WSL2.

    /opt/pai-venv/bin/python scripts/gpu/g1_playground_train.py --steps 20000000 --envs 4096 --out runs/gpu/g1_flat_s0

A standard, published baseline pipeline (the MuJoCo team's G1 joystick environment and PPO settings),
run on the laptop GPU; used as the walking baseline and to measure GPU training speed. Any policy
trained here is evaluated on the exact CPU physics before a number is reported.
"""

import argparse
import functools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

# Brax still calls jax.device_put_replicated, removed in recent JAX; this restores its behaviour (copy a
# pytree onto each device along a new leading axis) using jax.device_put.
if not callable(getattr(jax, "_pai_shim", None)):
    def _device_put_replicated(x, devices):
        return jax.tree_util.tree_map(
            lambda a: jax.device_put(jnp.broadcast_to(jnp.asarray(a), (len(devices),) + jnp.shape(a)), devices[0]), x)
    jax.device_put_replicated = _device_put_replicated
    jax._pai_shim = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="G1JoystickFlatTerrain")
    ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--restore", type=Path, help="resume from a Brax checkpoint folder")
    a = ap.parse_args()
    env = registry.load(a.env)
    env_cfg = registry.get_default_config(a.env)
    p = locomotion_params.brax_ppo_config(a.env)
    p.num_timesteps = a.steps
    p.num_envs = a.envs
    p.num_evals = max(2, min(p.num_evals, 10))
    a.out.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()

    def progress(step, metrics):
        row = {"step": int(step), "time_s": round(time.time() - t0, 1),
               "reward": float(metrics.get("eval/episode_reward", float("nan")))}
        log.append(row)
        print(row, flush=True)
        (a.out / "log.json").write_text(json.dumps(log, indent=1))

    net = functools.partial(ppo_networks.make_ppo_networks, **p.network_factory) if "network_factory" in p else ppo_networks.make_ppo_networks
    kw = {k: v for k, v in p.items() if k != "network_factory"}
    kw["save_checkpoint_path"] = str((a.out / "checkpoints").resolve())     # Brax saves at every evaluation
    if a.restore is not None:
        kw["restore_checkpoint_path"] = str(a.restore.resolve())
    make_inference, params, metrics = ppo.train(environment=env, eval_env=registry.load(a.env, config=env_cfg),
                                                wrap_env_fn=wrapper.wrap_for_brax_training, network_factory=net,
                                                progress_fn=progress, seed=a.seed, **kw)
    total = time.time() - t0
    (a.out / "summary.json").write_text(json.dumps({"env": a.env, "steps": a.steps, "envs": a.envs, "seed": a.seed,
                                                    "wall_s": round(total, 1), "steps_per_s": round(a.steps / total),
                                                    "devices": [str(d) for d in jax.devices()]}, indent=1))
    import pickle
    with open(a.out / "params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("done", round(total), "s", round(a.steps / total), "steps/s")


if __name__ == "__main__":
    main()
