"""Train the G1's riding policy with PPO (CycleRL-style) on town routes with corners and U-turns.

    python scripts/bike_g1_ppo_train.py --out data/bike_ppo/run1 --hours 3

Riders on routes from the street maps of training towns 5000-5007 (never used in evaluation); 30% of
routes contain a U-turn. Simulation on CPU threads (``mujoco.rollout``), the network on the GPU.
Checkpoints every 25 iterations; checkpoints are chosen afterwards on development towns
(``scripts/bike_g1_ppo_eval.py``), never on final towns.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.bike.ppo_rider import ACT_DIM, OBS_DIM, FastBikeVecEnv as BikeVecEnv
from pai.locomotion.policy import ActorCritic, save_policy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--max-steps", type=float, default=40e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--init", type=Path, help="start from this checkpoint's weights")
    ap.add_argument("--car-frac", type=float, default=0.0, help="fraction of routes with lane changes around parked cars")
    ap.add_argument("--vary-geometry", action="store_true", help="random corner (3.0-3.8 m) and U-turn (2.2-2.6 m) radii")
    ap.add_argument("--residual", type=Path,
                    help="residual mode: correct the practised rule with the gains in this result file")
    args = ap.parse_args()
    gamma, lam, clip, epochs, minibatches = 0.99, 0.95, 0.2, 5, 4
    lr, lr_min, lr_max, kl_target, ent_coef = 3e-4, 1e-5, 1e-3, 0.01, 0.002
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    gains = None if args.residual is None else np.array(json.loads(args.residual.read_text())["practised"])
    env = BikeVecEnv(args.envs, towns=range(5000, 5008), seed=args.seed, nthread=args.threads, residual_gains=gains,
                     vary_geometry=args.vary_geometry, car_frac=args.car_frac)
    ac = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=(256, 256),
                     init_std=0.4 if gains is None else 0.3).to(dev)
    if args.init is not None:
        ac.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["state_dict"])
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    N, T = args.envs, args.horizon
    obs = torch.as_tensor(env.observe(), device=dev)
    ac.obs_norm.update(obs)
    t0, it, total = time.time(), 0, 0
    ep_ret, ep_len = np.zeros(N), np.zeros(N)
    recent = []
    log = open(args.out / "log.jsonl", "a")
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))
    while time.time() - t0 < args.hours * 3600 and total < args.max_steps:
        it_t0 = time.time()
        B = {k: torch.zeros((T, N) + s, device=dev) for k, s in
             (("obs", (OBS_DIM,)), ("act", (ACT_DIM,)), ("logp", ()), ("val", ()), ("rew", ()), ("done", ()))}
        ends = {"fell": 0, "finished": 0, "trunc": 0}
        for k in range(T):
            a, logp, v = ac.act(obs)
            nobs, r, done, info = env.step(a.cpu().numpy())
            r_t = torch.as_tensor(r, device=dev)
            ep_ret += r
            ep_len += 1
            if "done_ids" in info:
                ids = info["done_ids"]
                tr = info["trunc"][ids]
                if tr.any():                     # time limit: bootstrap from the final observation
                    with torch.no_grad():
                        vb = ac.value(ac.obs_norm(torch.as_tensor(info["final_obs"][tr], device=dev)))
                    r_t[torch.as_tensor(ids[tr], device=dev)] += gamma * vb
                for i in ids:
                    recent.append({"ret": ep_ret[i], "len_s": ep_len[i] * env.dt,
                                   "fell": bool(info["fell"][i]), "finished": bool(info["finished"][i])})
                for key in ends:
                    ends[key] += int(info[key][ids].sum())
                ep_ret[ids] = 0
                ep_len[ids] = 0
            B["obs"][k], B["act"][k], B["logp"][k], B["val"][k] = obs, a, logp, v
            B["rew"][k] = r_t
            B["done"][k] = torch.as_tensor(done, device=dev, dtype=torch.float32)
            obs = torch.as_tensor(nobs, device=dev)
        total += N * T
        with torch.no_grad():
            last_v = ac.value(ac.obs_norm(obs))
        adv = torch.zeros_like(B["rew"])
        gae = torch.zeros(N, device=dev)
        for k in reversed(range(T)):
            nv = last_v if k == T - 1 else B["val"][k + 1]
            nonterm = 1.0 - B["done"][k]
            delta = B["rew"][k] + gamma * nv * nonterm - B["val"][k]
            gae = delta + gamma * lam * nonterm * gae
            adv[k] = gae
        ret = (adv + B["val"]).reshape(-1)
        o_, a_, lp_, v_ = (B["obs"].reshape(-1, OBS_DIM), B["act"].reshape(-1, ACT_DIM),
                           B["logp"].reshape(-1), B["val"].reshape(-1))
        adv = adv.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n, mb = o_.shape[0], o_.shape[0] // minibatches
        kls = []
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for j in range(minibatches):
                idx = perm[j * mb:(j + 1) * mb]
                on = ac.obs_norm(o_[idx])
                d = ac.dist(on)
                logp = d.log_prob(a_[idx]).sum(-1)
                ratio = torch.exp(logp - lp_[idx])
                pl = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                v = ac.value(on)
                vc = v_[idx] + (v - v_[idx]).clamp(-clip, clip)
                vl = torch.max((v - ret[idx]) ** 2, (vc - ret[idx]) ** 2).mean()
                loss = pl + vl - ent_coef * d.entropy().sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    lr_ = logp - lp_[idx]
                    kls.append(((lr_.exp() - 1) - lr_).mean().item())
            kl = float(np.mean(kls[-minibatches:]))
            lr = max(lr_min, lr / 1.5) if kl > 2 * kl_target else min(lr_max, lr * 1.5) if kl < kl_target / 2 else lr
            for g in opt.param_groups:
                g["lr"] = lr
        ac.obs_norm.update(o_)
        it += 1
        recent = recent[-2000:]
        row = {"it": it, "steps": total, "time_min": round((time.time() - t0) / 60, 1),
               "sps": round(N * T / (time.time() - it_t0)), **ends,
               "ep_return": float(np.mean([e["ret"] for e in recent])) if recent else None,
               "ep_len_s": float(np.mean([e["len_s"] for e in recent])) if recent else None,
               "finish_rate": float(np.mean([e["finished"] for e in recent])) if recent else None,
               "lr": lr, "kl": float(np.mean(kls)), "std": float(ac.log_std.detach().exp().mean())}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if it % 10 == 0 and recent:
            print(f"it {it} {total/1e6:.1f}M sps {row['sps']} ret {row['ep_return']:.1f} len {row['ep_len_s']:.1f}s "
                  f"finish {row['finish_rate']:.2f} fell {ends['fell']} lr {lr:.1e} std {row["std"]:.2f} "
                  f"t {row['time_min']:.0f}min", flush=True)
        if it % 25 == 0:
            save_policy(args.out / f"ckpt_{it:05d}.pt", ac, {"it": it, "steps": total, "obs_dim": OBS_DIM,
                                                             "act_dim": ACT_DIM, "residual": gains})
    save_policy(args.out / "ckpt_last.pt", ac, {"it": it, "steps": total, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
                                                 "residual": gains})
    print("done", it, total)


if __name__ == "__main__":
    main()
