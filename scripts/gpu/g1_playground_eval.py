"""Evaluate the GPU-trained G1 walking baseline with tests like the agent's own walking gates.

    /opt/pai-venv/bin/python scripts/gpu/g1_playground_eval.py --run runs/gpu/g1_flat_full_s0 --output results/gpu_g1_baseline_eval.json

Policy: MuJoCo Playground's G1 joystick task trained with its published PPO recipe
(`scripts/gpu/g1_playground_train.py`). Every test runs twice: with the training physics (3 solver
iterations, as used for speed) and with **full-accuracy physics** (MuJoCo's default 100 solver and 50
line-search iterations), which solve the same equations as CPU MuJoCo; a policy that only works with
the approximate physics would fail there. 20 starts per test, from the environment's own randomised
start states; new evaluation seeds. The tests mirror `scripts/locomotion_eval.py`:

- stand 30 s (command zero);
- walk 10 m at 0.8 m/s within 20 s;
- headings: turn to a random target heading and hold it within 15 degrees at 0.5 m/s (8 s), steering
  by a yaw-rate command (the policy takes velocity commands, not headings);
- pushes while walking at 0.5 m/s: a sudden sideways velocity change of 0.25-2.0 m/s at 3 s; recovered
  if still up 4 s later.

Differences from the agent's tests, stated in the results: the baseline is tested on its own G1 model
and nominal dynamics (the agent's tests randomise friction, masses, payload and servo strength); its
commands are limited to 1 m/s, so there is no 2 m/s run test.
"""

import argparse
import functools
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

if not callable(getattr(jax, "_pai_shim", None)):          # see g1_playground_train.py
    def _device_put_replicated(x, devices):
        return jax.tree_util.tree_map(
            lambda a: jax.device_put(jnp.broadcast_to(jnp.asarray(a), (len(devices),) + jnp.shape(a)), devices[0]), x)
    jax.device_put_replicated = _device_put_replicated
    jax._pai_shim = _device_put_replicated

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

ENV = "G1JoystickFlatTerrain"
N = 20


def load_policy(run: Path, env):
    p = locomotion_params.brax_ppo_config(ENV)
    nf = dict(p.network_factory) if "network_factory" in p else {}
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **nf)
    params = pickle.loads((run / "params.pkl").read_bytes())
    make = ppo_networks.make_inference_fn(nets)
    return jax.jit(make(params, deterministic=True))


def set_physics(env, full: bool):
    m = env._mj_model
    if full:
        m.opt.iterations, m.opt.ls_iterations = 100, 50
    env._mjx_model = mjx.put_model(m, impl=env._mjx_model.impl.value)


def rollout(env, policy, seed, seconds, command_fn, push=None):
    """Run N starts; command_fn(t, state) -> (N, 3) command. Returns per-start arrays."""
    keys = jax.random.split(jax.random.PRNGKey(seed), N)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    act = jax.vmap(policy)
    s = reset(keys)
    s.info["push_interval_steps"] = jnp.full((N,), 10 ** 9)          # no random pushes of the env
    steps = int(round(seconds / env.dt))
    alive = jnp.ones(N, bool)
    xy0 = s.data.qpos[:, :2]
    trace = []
    k_act = jax.random.split(jax.random.PRNGKey(seed + 1), N)
    for k in range(steps):
        t = k * env.dt
        cmd = command_fn(t, s)
        s.info["command"] = cmd
        s.info["step"] = jnp.zeros((N,), s.info["step"].dtype)        # keep the command fixed
        if push is not None and abs(t - push[0]) < env.dt / 2:
            qv = s.data.qvel.at[:, :2].add(push[1])
            s = s.replace(data=s.data.replace(qvel=qv))
        a, _ = act(s.obs, k_act)
        s = step(s, a)
        alive = alive & (s.done < 0.5)
        trace.append((np.asarray(s.data.qpos[:, :3]), np.asarray(s.data.qpos[:, 3:7]), np.asarray(alive)))
    return trace, np.asarray(xy0)


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_tests(env, policy, seed0):
    out = {}
    zero = lambda t, s: jnp.zeros((N, 3))
    tr, _ = rollout(env, policy, seed0, 30.0, zero)
    out["stand_30s"] = int(tr[-1][2].sum())
    # walk 10 m at 0.8 m/s within 20 s
    fwd = lambda t, s: jnp.tile(jnp.array([0.8, 0.0, 0.0]), (N, 1))
    tr, xy0 = rollout(env, policy, seed0 + 1, 20.0, fwd)
    reached = np.zeros(N, bool)
    for pos, _, alive in tr:
        reached |= alive & (np.linalg.norm(pos[:, :2] - xy0, axis=1) >= 10.0)
    out["walk_10m"] = int(reached.sum())
    # headings: random target heading, steer with a yaw-rate command at 0.5 m/s
    rng = np.random.default_rng(seed0 + 2)
    offsets = rng.uniform(-np.pi, np.pi, N)
    target = {}

    def head(t, s):
        yaw = yaw_of(np.asarray(s.data.qpos[:, 3:7]))
        if "y" not in target:
            target["y"] = yaw + offsets
        err = (target["y"] - yaw + np.pi) % (2 * np.pi) - np.pi
        return jnp.asarray(np.stack([np.full(N, 0.5), np.zeros(N), np.clip(2.0 * err, -1.0, 1.0)], 1), jnp.float32)

    tr, _ = rollout(env, policy, seed0 + 2, 8.0, head)
    pos, quat, alive = tr[-1]
    err = np.degrees(np.abs((target["y"] - yaw_of(quat) + np.pi) % (2 * np.pi) - np.pi))
    out["headings"] = int((alive & (err < 15.0)).sum())
    out["headings_median_error_deg"] = round(float(np.median(err[alive])), 2) if alive.any() else None
    # pushes while walking at 0.5 m/s
    walk = lambda t, s: jnp.tile(jnp.array([0.5, 0.0, 0.0]), (N, 1))
    pushes = {}
    for k, dv in enumerate((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
        ang = np.random.default_rng(seed0 + 10 + k).uniform(0, 2 * np.pi, N)
        vec = jnp.asarray(np.stack([np.cos(ang), np.sin(ang)], 1) * dv, jnp.float32)
        tr, _ = rollout(env, policy, seed0 + 10 + k, 7.0, walk, push=(3.0, vec))
        pushes[str(dv)] = int(tr[-1][2].sum())
    out["pushes_recovered_of_20"] = pushes
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=61000)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    report = {"run": a.run.as_posix(), "starts_per_test": N, "eval_seed": a.seed, "env": ENV}
    for name, full in (("training_physics", False), ("full_accuracy_physics", True)):
        env = registry.load(ENV)
        set_physics(env, full)
        env.reset = env.reset                      # (methods read env._mjx_model at call time)
        policy = load_policy(a.run, env)
        report[name] = run_tests(env, policy, a.seed)
        print(name, json.dumps(report[name]), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
