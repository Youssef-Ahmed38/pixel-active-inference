"""PPO practice loop with a learned body model, curiosity and an automatic curriculum.

Objective per step (a reward-style form of negative expected free energy):
    pragmatic  = preference scores for the goal (velocity/heading command, upright, rhythm)
    epistemic  = beta_t * normalised ensemble disagreement of the body model (curiosity),
                 beta_t annealed to zero over ``curiosity_anneal`` of training
Falls end the episode (a large negative preference). The body model keeps learning from all
practice transitions and its prediction error (surprise) is logged as a learning signal.

Curriculum: stand -> balance (pushes) -> walk -> turn -> run. The next stage starts when the
current one meets its advancement rule on the practice episodes (see ``ready_to_advance``).
Earlier commands stay in the mix (15% stand-still commands and the ranges overlap).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from pai.locomotion.env import ACT_DIM, PROPRIO, EnvConfig, LocomotionEnv
from pai.locomotion.policy import ActorCritic
from pai.locomotion.world_model import EnsembleBodyModel, ReplayBuffer, train_steps


@dataclass
class TrainConfig:
    num_envs: int = 2048
    horizon: int = 24
    iterations: int = 3000
    max_hours: float = 4.0
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    lr: float = 1e-3
    kl_target: float = 0.01
    lr_min: float = 1e-4
    lr_max: float = 1e-3
    entropy: float = 0.005
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_std: float = 0.8
    hidden: tuple = (512, 256, 128)
    curiosity: float = 0.02
    curiosity_anneal: float = 0.4
    wm_members: int = 5
    wm_steps: int = 8
    wm_batch: int = 2048
    wm_lr: float = 1e-3
    wm_buffer: int = 500_000
    advance_window: int = 2000
    advance_survival: float = 0.85
    advance_lin: float = 0.6
    advance_yaw: float = 0.55
    min_iters_per_stage: int = 60
    start_stage: int = 0
    ckpt_every: int = 50
    seed: int = 0
    torch_threads: int = 2


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class StageStats:
    """Episode outcomes of the current stage (for the advancement rule and logs)."""

    def __init__(self, window: int):
        self.window = window
        self.rows: list[dict] = []

    def add(self, row):
        self.rows.append(row)
        self.rows = self.rows[-self.window:]

    def mean(self, key):
        v = [r[key] for r in self.rows if key in r and np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")


def ready_to_advance(stage_name: str, st: StageStats, cfg: TrainConfig, iters_in_stage: int) -> bool:
    if iters_in_stage < cfg.min_iters_per_stage or len(st.rows) < st.window:
        return False
    surv = st.mean("survival")
    if stage_name in ("stand", "balance"):
        return surv >= cfg.advance_survival
    if stage_name == "walk":
        return surv >= cfg.advance_survival and st.mean("lin_track") >= cfg.advance_lin
    if stage_name == "turn":
        return (surv >= cfg.advance_survival and st.mean("lin_track") >= cfg.advance_lin
                and st.mean("yaw_track") >= cfg.advance_yaw)
    return False


class Trainer:
    def __init__(self, out_dir: Path, tcfg: TrainConfig, ecfg: EnvConfig, device: str = "cuda",
                 body_model_path: Path | None = None, log=print):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t, self.e = tcfg, ecfg
        self.log = log
        torch.set_num_threads(tcfg.torch_threads)
        torch.manual_seed(tcfg.seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(tcfg.seed + 7)
        self.env = LocomotionEnv(tcfg.num_envs, ecfg, seed=tcfg.seed, stage=tcfg.start_stage)
        self.ac = ActorCritic(hidden=tcfg.hidden, init_std=tcfg.init_std).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=tcfg.lr)
        self.lr = tcfg.lr
        self.wm = EnsembleBodyModel(tcfg.wm_members).to(self.device)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=tcfg.wm_lr)
        self.buf = ReplayBuffer(tcfg.wm_buffer)
        self.body_model_source = None
        if body_model_path and Path(body_model_path).is_file():
            ck = torch.load(body_model_path, map_location=self.device, weights_only=False)
            self.wm.load_state_dict(ck["model"])
            self.wm.fitted = True
            self.body_model_source = str(body_model_path)
        self.it = 0
        self.stage_iter = 0
        self.total_steps = 0
        self.stage_log: list[dict] = []
        self.stats = StageStats(tcfg.advance_window)
        self.dis_mean, self.dis_std = 0.0, 1.0
        self.elapsed_before = 0.0
        self.ep_ret = np.zeros(tcfg.num_envs)
        self.ep_lin = np.zeros(tcfg.num_envs)
        self.ep_yaw = np.zeros(tcfg.num_envs)
        self.ep_len = np.zeros(tcfg.num_envs)

    # ------------------------------------------------------------ checkpoints
    def state(self) -> dict:
        return {"policy": self.ac.state_dict(), "opt": self.opt.state_dict(), "lr": self.lr,
                "hidden": self.ac.hidden, "model": self.wm.state_dict(), "wm_opt": self.wm_opt.state_dict(),
                "it": self.it, "stage": self.env.stage, "stage_iter": self.stage_iter,
                "total_steps": self.total_steps, "stage_log": self.stage_log,
                "dis": (self.dis_mean, self.dis_std), "elapsed": self.elapsed(),
                "train_cfg": asdict(self.t), "env_cfg": json.loads(json.dumps(asdict(self.e), default=list)),
                "commit": git_commit(), "body_model_source": self.body_model_source}

    def save(self, name="ckpt_last.pt"):
        tmp = self.out / (name + ".tmp")
        torch.save(self.state(), tmp)
        os.replace(tmp, self.out / name)
        # the frozen controller file other stages load
        torch.save({"state_dict": self.ac.state_dict(), "hidden": self.ac.hidden, "it": self.it,
                    "stage": self.env.stage, "total_steps": self.total_steps, "commit": git_commit()},
                   self.out / "policy.pt")

    def restore(self, path: Path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.ac.load_state_dict(ck["policy"])
        self.opt.load_state_dict(ck["opt"])
        self.lr = float(np.clip(ck["lr"], self.t.lr_min, self.t.lr_max))
        for g in self.opt.param_groups:
            g["lr"] = self.lr
        self.wm.load_state_dict(ck["model"])
        self.wm.fitted = True
        self.wm_opt.load_state_dict(ck["wm_opt"])
        self.it, self.stage_iter, self.total_steps = ck["it"], ck["stage_iter"], ck["total_steps"]
        self.env.stage = ck["stage"]
        self.env.reset(np.arange(self.env.n))
        self.stage_log = ck["stage_log"]
        self.dis_mean, self.dis_std = ck["dis"]
        self.elapsed_before = ck.get("elapsed", 0.0)
        self.body_model_source = ck.get("body_model_source")
        self.log(f"resumed from {path} at iteration {self.it}, stage {self.env.stage}")

    def elapsed(self):
        return self.elapsed_before + (time.time() - self._t0 if hasattr(self, "_t0") else 0.0)

    # ------------------------------------------------------------ main loop
    def run(self):
        t, env, dev = self.t, self.env, self.device
        self._t0 = time.time()
        obs = torch.as_tensor(env.observe(), device=dev)
        N, T = t.num_envs, t.horizon
        logf = open(self.out / "log.jsonl", "a")
        while self.it < t.iterations and self.elapsed() < t.max_hours * 3600:
            it_t0 = time.time()
            B_obs = torch.zeros(T, N, obs.shape[1], device=dev)
            B_act = torch.zeros(T, N, ACT_DIM, device=dev)
            B_logp = torch.zeros(T, N, device=dev)
            B_val = torch.zeros(T, N, device=dev)
            B_rew = torch.zeros(T, N, device=dev)
            B_done = torch.zeros(T, N, device=dev)
            B_next = torch.zeros(T, N, obs.shape[1], device=dev)
            B_valid = torch.zeros(T, N, device=dev)       # next obs belongs to the same episode
            ended = {"fell": 0, "trunc": 0}
            sim_time = 0.0
            for k in range(T):
                a, logp, v = self.ac.act(obs)
                ts = time.time()
                nobs, r, done, info = env.step(a.cpu().numpy())
                sim_time += time.time() - ts
                nobs_t = torch.as_tensor(nobs, device=dev)
                r_t = torch.as_tensor(r, device=dev)
                next_for_model = nobs_t.clone()
                valid = torch.ones(N, device=dev)
                terms = info["terms"]
                self.ep_ret += r
                self.ep_lin += terms["lin"]
                self.ep_yaw += terms["yaw"]
                self.ep_len += 1
                if "done_ids" in info:
                    ids = info["done_ids"]
                    fin = torch.as_tensor(info["final_obs"], device=dev)
                    ids_t = torch.as_tensor(ids, device=dev)
                    next_for_model[ids_t] = fin
                    fell = info["fell"][ids]
                    valid[ids_t[torch.as_tensor(fell, device=dev)]] = 0.0   # fall: no successor
                    tr = ids[~fell]
                    if len(tr):                                        # time limit: bootstrap
                        with torch.no_grad():
                            vb = self.ac.value(self.ac.obs_norm(torch.as_tensor(info["final_obs"][~fell], device=dev)))
                        r_t[torch.as_tensor(tr, device=dev)] += t.gamma * vb
                    for i in ids:
                        L = self.ep_len[i]
                        self.stats.add({"survival": float(not info["fell"][i]),
                                        "lin_track": self.ep_lin[i] / L, "yaw_track": self.ep_yaw[i] / L,
                                        "return": self.ep_ret[i], "length_s": L * env.cfg.ctrl_dt})
                    ended["fell"] += int(fell.sum())
                    ended["trunc"] += int((~fell).sum())
                    self.ep_ret[ids] = self.ep_lin[ids] = self.ep_yaw[ids] = self.ep_len[ids] = 0
                B_obs[k], B_act[k], B_logp[k], B_val[k] = obs, a, logp, v
                B_rew[k], B_done[k] = r_t, torch.as_tensor(done, device=dev, dtype=torch.float32)
                B_next[k], B_valid[k] = next_for_model, valid
                obs = nobs_t
            self.total_steps += N * T
            # ---- body model: surprise, curiosity, learning
            s = B_obs[..., PROPRIO].reshape(-1, len(PROPRIO))
            s2 = B_next[..., PROPRIO].reshape(-1, len(PROPRIO))
            act = B_act.reshape(-1, ACT_DIM).clamp(-env.cfg.action_clip, env.cfg.action_clip)
            vmask = B_valid.reshape(-1) > 0
            if not self.wm.fitted:
                self.wm.fit_normalizers(s[vmask], act[vmask], s2[vmask])
            sur, dis = self.wm.surprise_and_disagreement(s, act, s2)
            beta = t.curiosity * max(0.0, 1.0 - self.it / max(1.0, t.curiosity_anneal * t.iterations))
            self.dis_mean = 0.99 * self.dis_mean + 0.01 * dis.mean().item() if self.it else dis.mean().item()
            self.dis_std = 0.99 * self.dis_std + 0.01 * dis.std().item() if self.it else max(dis.std().item(), 1e-6)
            if beta > 0:
                bonus = ((dis - self.dis_mean) / (self.dis_std + 1e-8)).clamp(-3, 3)
                B_rew += beta * bonus.reshape(T, N)
            pick = vmask.nonzero().squeeze(-1)
            pick = pick[torch.randperm(len(pick), device=dev)[:16384]]
            self.buf.add(s[pick].cpu().numpy(), act[pick].cpu().numpy(), s2[pick].cpu().numpy())
            wm_loss = train_steps(self.wm, self.wm_opt, self.buf, t.wm_steps, t.wm_batch, self.rng, dev)
            # surprise right before a fall versus ordinary steps
            fell_mask = (B_valid == 0).reshape(-1)
            sur_fall = float(sur[fell_mask].mean()) if fell_mask.any() else float("nan")
            sur_ok = float(sur[vmask].mean())
            # ---- GAE and PPO update
            with torch.no_grad():
                last_v = self.ac.value(self.ac.obs_norm(obs))
            adv = torch.zeros_like(B_rew)
            gae = torch.zeros(N, device=dev)
            for k in reversed(range(T)):
                nv = last_v if k == T - 1 else B_val[k + 1]
                nonterm = 1.0 - B_done[k]
                delta = B_rew[k] + t.gamma * nv * nonterm - B_val[k]
                gae = delta + t.gamma * t.lam * nonterm * gae
                adv[k] = gae
            ret = adv + B_val
            stats = self._update(B_obs.reshape(-1, obs.shape[1]), B_act.reshape(-1, ACT_DIM),
                                 B_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1), B_val.reshape(-1))
            # the observation normaliser changes only between iterations, so the stored
            # log-probabilities always match the policy that is being updated
            self.ac.obs_norm.update(B_obs.reshape(-1, obs.shape[1]))
            self.it += 1
            self.stage_iter += 1
            # ---- curriculum
            stage_name = env.cfg.stages[env.stage].name
            advanced = False
            if env.stage < len(env.cfg.stages) - 1 and ready_to_advance(stage_name, self.stats, t, self.stage_iter):
                self.stage_log.append({"stage": stage_name, "done_at_iter": self.it, "steps": self.total_steps,
                                       "survival": self.stats.mean("survival"),
                                       "lin_track": self.stats.mean("lin_track"),
                                       "yaw_track": self.stats.mean("yaw_track")})
                env.stage += 1
                self.stage_iter = 0
                self.stats = StageStats(t.advance_window)
                advanced = True
                self.save(f"stage_{stage_name}.pt")
            row = {"it": self.it, "steps": self.total_steps, "stage": env.cfg.stages[env.stage].name,
                   "time_s": round(self.elapsed(), 1), "sps": round(N * T / (time.time() - it_t0)),
                   "sim_frac": round(sim_time / (time.time() - it_t0), 2),
                   "falls": ended["fell"], "timeouts": ended["trunc"],
                   "fall_rate": ended["fell"] / max(1, ended["fell"] + ended["trunc"]),
                   "survival": self.stats.mean("survival"), "lin_track": self.stats.mean("lin_track"),
                   "yaw_track": self.stats.mean("yaw_track"), "ep_return": self.stats.mean("return"),
                   "ep_length_s": self.stats.mean("length_s"),
                   "reward": float(B_rew.mean()), "beta": beta, "disagreement": float(dis.mean()),
                   "surprise": sur_ok, "surprise_before_fall": sur_fall, "wm_loss": wm_loss,
                   "lr": self.lr, "std": float(self.ac.log_std.detach().exp().mean()), **stats,
                   "advanced": advanced}
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            if self.it % 10 == 0 or advanced:
                self.log(f"it {self.it} {row['stage']} steps {self.total_steps/1e6:.1f}M sps {row['sps']} "
                         f"surv {row['survival']:.2f} lin {row['lin_track']:.2f} yaw {row['yaw_track']:.2f} "
                         f"len {row['ep_length_s']:.1f}s falls {row['falls']} sur {sur_ok:.3f}/{sur_fall:.3f} "
                         f"kl {stats['kl']:.4f} lr {self.lr:.1e} t {row['time_s']/60:.0f}min")
            if self.it % t.ckpt_every == 0:
                self.save()
        self.save()
        logf.close()

    def _update(self, obs, act, logp_old, adv, ret, val_old):
        t = self.t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        mb = n // t.minibatches
        kls, pls, vls, ents = [], [], [], []
        for _ in range(t.epochs):
            perm = torch.randperm(n, device=obs.device)
            for i in range(t.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                on = self.ac.obs_norm(obs[idx])
                d = self.ac.dist(on)
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = torch.exp(logp - logp_old[idx])
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - t.clip, 1 + t.clip) * adv[idx]
                pl = -torch.min(s1, s2).mean()
                v = self.ac.value(on)
                v_clip = val_old[idx] + (v - val_old[idx]).clamp(-t.clip, t.clip)
                vl = torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + t.value_coef * vl - t.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), t.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    lr_ = logp - logp_old[idx]
                    kl = ((lr_.exp() - 1) - lr_).mean().item()
                kls.append(kl); pls.append(pl.item()); vls.append(vl.item()); ents.append(ent.item())
            # adaptive learning rate on the epoch's mean KL, kept in [lr_min, lr_max]
            # (per-minibatch adaptation drove it to the floor: Adam moves every weight by ~lr, so
            # even tiny steps give a KL near the target with these layer sizes)
            kl_ep = float(np.mean(kls[-t.minibatches:]))
            if kl_ep > 2 * t.kl_target:
                self.lr = max(t.lr_min, self.lr / 1.5)
            elif kl_ep < t.kl_target / 2:
                self.lr = min(t.lr_max, self.lr * 1.5)
            for g in self.opt.param_groups:
                g["lr"] = self.lr
        return {"kl": float(np.mean(kls)), "policy_loss": float(np.mean(pls)),
                "value_loss": float(np.mean(vls)), "entropy": float(np.mean(ents))}
