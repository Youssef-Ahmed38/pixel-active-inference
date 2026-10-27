"""Motor babbling: the body explores its own motor space to learn a body model.

Two explorers with the same budget of steps:

* ``random``  - smooth random servo targets (an Ornstein-Uhlenbeck process) around the default
                posture;
* ``curious`` - at every step it proposes several random variations and executes the one the
                current ensemble body model disagrees about most (the epistemic value of the
                expected free energy), retraining the model as it goes.

Falls are kept: they are informative transitions (the body ends on the ground and resets).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pai.locomotion.env import ACT_DIM, PROPRIO, EnvConfig, LocomotionEnv
from pai.locomotion.world_model import EnsembleBodyModel, ReplayBuffer, train_steps


@dataclass
class BabbleConfig:
    num_envs: int = 512
    steps: int = 600
    theta: float = 0.15            # OU mean reversion per step
    sigma: float = 0.35            # OU noise per step (stationary std ~ sigma / sqrt(2 theta))
    candidates: int = 8
    warmup_frac: float = 0.2
    retrain_every: int = 25
    train_steps: int = 40
    batch: int = 2048


def babble(mode: str, cfg: BabbleConfig, env_cfg: EnvConfig, seed: int, device,
           model: EnsembleBodyModel | None = None, log=print):
    """Collect (s, a, s') transitions; returns arrays plus fall count and the model (curious)."""
    if mode not in ("random", "curious"):
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    env = LocomotionEnv(cfg.num_envs, env_cfg, seed=seed, stage=0)
    env.pushes_enabled = False
    obs = env.observe()
    a = np.zeros((cfg.num_envs, ACT_DIM))
    S, A, S2 = [], [], []
    falls = 0
    buf = ReplayBuffer(cfg.num_envs * cfg.steps)
    if mode == "curious":
        model = model or EnsembleBodyModel().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    warm = int(cfg.warmup_frac * cfg.steps)
    for k in range(cfg.steps):
        if mode == "curious" and k >= warm and model.fitted:
            eps = rng.normal(0, cfg.sigma, (cfg.candidates, cfg.num_envs, ACT_DIM))
            cand = a[None] * (1 - cfg.theta) + eps
            s_t = torch.as_tensor(obs[:, PROPRIO], device=device).repeat(cfg.candidates, 1)
            c_t = torch.as_tensor(np.clip(cand, -env_cfg.action_clip, env_cfg.action_clip)
                                  .reshape(-1, ACT_DIM), device=device, dtype=torch.float32)
            _, dis = model.surprise_and_disagreement(s_t, c_t)
            best = dis.reshape(cfg.candidates, cfg.num_envs).argmax(0).cpu().numpy()
            a = cand[best, np.arange(cfg.num_envs)]
        else:
            a = a * (1 - cfg.theta) + rng.normal(0, cfg.sigma, a.shape)
        a = np.clip(a, -env_cfg.action_clip, env_cfg.action_clip)
        nobs, _, done, info = env.step(a)
        nxt = nobs.copy()
        valid = np.ones(cfg.num_envs, bool)
        if "done_ids" in info:
            ids = info["done_ids"]
            nxt[ids] = info["final_obs"]
            falls += int(info["fell"].sum())     # the fall transition itself is kept
        s, s2, a_rec = obs[valid][:, PROPRIO], nxt[valid][:, PROPRIO], a[valid].astype(np.float32)
        S.append(s); A.append(a_rec); S2.append(s2)
        if "done_ids" in info:
            a[info["done_ids"]] = 0.0               # a fresh body starts from the default posture
        if mode == "curious":
            buf.add(s, a_rec, s2)
            if k + 1 == warm or (k >= warm and (k + 1) % cfg.retrain_every == 0):
                if not model.fitted:
                    t = [torch.as_tensor(x[:buf.n], device=device) for x in (buf.s, buf.a, buf.s2)]
                    model.fit_normalizers(*t)
                train_steps(model, opt, buf, cfg.train_steps * (4 if k + 1 == warm else 1),
                            cfg.batch, rng, device)
        obs = nobs
    return (np.concatenate(S), np.concatenate(A), np.concatenate(S2)), falls, model


def fit_model(data, device, steps: int, seed: int, batch: int = 4096, members: int = 5,
              log=print) -> EnsembleBodyModel:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    s, a, s2 = data
    m = EnsembleBodyModel(members).to(device)
    m.fit_normalizers(*(torch.as_tensor(x, device=device) for x in (s, a, s2)))
    buf = ReplayBuffer(len(s))
    buf.add(s, a, s2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for i in range(0, steps, 500):
        loss = train_steps(m, opt, buf, min(500, steps - i), batch, rng, device)
        for _ in range(min(500, steps - i)):
            sched.step()
        log(f"  body model step {i + 500}: loss {loss:.4f}")
    return m
