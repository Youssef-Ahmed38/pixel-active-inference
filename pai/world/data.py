"""Transition data for the entity world model: (tokens_t, action_t, tokens_t+1) at planner rate.

The planner acts every `action_repeat` control steps (default 5 x 0.02 s = 0.1 s) and holds the
action in between; transitions are recorded at that rate. Data is undisturbed on purpose: the
model learns *nominal* physics, so pushes and unexpected masses later show up as surprise.

Policies mixed per episode:
- noisy scripted pick-and-place (onto the plate, into the bowl, or to a random spot)
- exploration: wander between random points near objects, toggling the gripper at random
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

ACTION_DIM = 4  # dx, dy, dz (m/s), grip (-1 close .. 1 open); yaw is held fixed in the slice


def _episode(env, rng, action_repeat: int, max_steps: int, noise: float, on_step=None):
    """One exploration/demonstration episode. on_step(env, obs) is called after reset and after
    every planner step, e.g. to record camera frames for perception training."""
    from pai.envs.scripted import PickPlace
    from pai.world.entities import encode

    obs = env.reset(seed=int(rng.integers(1 << 31)))
    blocks = [b.name for b in env.cfg.blocks]
    mode = rng.choice(["pick_place", "explore"], p=[0.7, 0.3])
    obj = str(rng.choice(blocks))
    target = str(rng.choice(["plate", "bowl"]))
    script = PickPlace(env, obj, target) if mode == "pick_place" else None
    waypoint, grip = None, 1.0
    prev_ee = obs["ee_pos"].copy()
    tok = encode(env, obs, prev_ee)
    toks, acts = [tok], []
    if on_step is not None:
        on_step(env, obs)
    for _ in range(max_steps // action_repeat):
        if script is not None and not script.done():
            a = script.act(obs)
            a = np.r_[a[:3], a[4]]
        else:
            if waypoint is None or np.linalg.norm(waypoint - obs["ee_pos"]) < 0.02 or rng.random() < 0.05:
                name = str(rng.choice(list(env.objects)))
                waypoint = obs["state"][name]["pos"] + rng.normal(0, 0.05, 3) + [0, 0, rng.uniform(0.0, 0.15)]
                if rng.random() < 0.3:
                    grip = -grip
            a = np.r_[np.clip(4.0 * (waypoint - env.ee_target_pos), -0.25, 0.25), grip]
        a = a + np.r_[rng.normal(0, noise, 3), 0.0]
        if rng.random() < 0.02:  # occasional wrong gripper command, so the model sees drops
            a[3] = -a[3]
        a = np.clip(a, [-0.25, -0.25, -0.25, -1], [0.25, 0.25, 0.25, 1]).astype(np.float32)
        for _ in range(action_repeat):
            prev_ee = obs["ee_pos"].copy()
            obs = env.step(np.r_[a[:3], 0.0, a[3]])
        toks.append(encode(env, obs, prev_ee))
        acts.append(a)
        if on_step is not None:
            on_step(env, obs)
    return np.stack(toks), np.stack(acts)


def _worker(args):
    env_cfg, out_dir, episode_ids, action_repeat, max_steps, noise, seed = args
    from pai.envs import TabletopEnv

    env = TabletopEnv(env_cfg, disturbances=[])
    for eid in episode_ids:
        path = out_dir / f"ep_{eid:05d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng(seed * 1_000_003 + eid)
        toks, acts = _episode(env, rng, action_repeat, max_steps, noise)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, tokens=toks, actions=acts)
        tmp.replace(path)
    env.close()
    return len(episode_ids)


def collect_transitions(env_cfg, out_dir: str | Path, n_episodes: int, workers: int = 8,
                        action_repeat: int = 5, max_steps: int = 600, noise: float = 0.03, seed: int = 0) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = type(env_cfg)({**env_cfg, "render_images": False})
    ids = list(range(n_episodes))
    chunks = [ids[i::workers] for i in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        pool.map(_worker, [(env_cfg, out_dir, c, action_repeat, max_steps, noise, seed) for c in chunks])
    print(f"{n_episodes} episodes in {time.time() - t0:.0f}s -> {out_dir}")


def episode_files(data_dir: str | Path) -> list[Path]:
    return [f for f in sorted(Path(data_dir).glob("ep_*.npz")) if not f.name.endswith(".tmp.npz")]


def load_transitions(data_dir: str | Path, files: list[Path] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (tokens_t, actions_t, tokens_t+1, episode index) stacked over episodes."""
    x, a, y, e = [], [], [], []
    for i, f in enumerate(files if files is not None else episode_files(data_dir)):
        d = np.load(f)
        x.append(d["tokens"][:-1])
        a.append(d["actions"])
        y.append(d["tokens"][1:])
        e.append(np.full(len(d["actions"]), i))
    return np.concatenate(x), np.concatenate(a), np.concatenate(y), np.concatenate(e)
