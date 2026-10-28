"""Evaluate PPO riding policies against the practised lean-target rule (v3) on the same routes.

    python scripts/bike_g1_ppo_eval.py --ckpts data/bike_ppo/run2/ckpt_*.pt --towns 4100 4105 --output results/bike_g1_ppo_dev.json
    python scripts/bike_g1_ppo_eval.py --ckpts <chosen> --towns 4230 4240 --with-rule --output results/bike_g1_ppo_final.json

Routes as in ``scripts/bike_g1_turn_practice.py``: per town two routes with corners and two with a
U-turn. A ride ends with a fall, or when the rider is more than 3 m from its route. Measures as in
CycleRL: completion, falls, mean heading error, mean and maximum distance from the lane.
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

_ENVS, _POL = {}, {}


def load(path):
    from pai.bike.ppo_rider import ACT_DIM, OBS_DIM
    from pai.locomotion.policy import ActorCritic
    if path not in _POL:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        ac = ActorCritic(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=tuple(ck["hidden"]))
        ac.load_state_dict(ck["state_dict"])
        ac.eval()
        ac.residual_gains = ck.get("residual")
        _POL[path] = ac
    return _POL[path]


def ride(job):
    seed, route, who = job[:3]
    vary = job[3] if len(job) > 3 else False
    torch.set_num_threads(1)
    from pai.bike.g1_rider import G1BikeEnv
    from pai.bike.ppo_rider import ARM_KP, STEER_KP, PPOFollower
    from pai.bike.riding import CORNER_R, LeanFollower, lane_path
    from pai.bike.town import Town
    if seed not in _ENVS:
        _ENVS[seed] = (Town(n=4, seed=seed), G1BikeEnv(scenery=False, max_seconds=1e9, arm_kp=ARM_KP, steer_kp=STEER_KP))
    town, env = _ENVS[seed]
    if vary:      # randomized geometry, the same for every rider on this route
        import pai.bike.riding as riding
        g = np.random.default_rng(abs(hash((seed, tuple(route)))) % 2**32)
        keep, riding.U_TURN_R = riding.U_TURN_R, float(g.uniform(2.2, 2.6))
        path = lane_path(town.nodes, route, radius=float(g.uniform(3.0, 3.8)))
        riding.U_TURN_R = keep
    else:
        path = lane_path(town.nodes, route, radius=CORNER_R)
    d = path[min(5, len(path) - 1)] - path[0]
    s = env.reset(seed=seed, speed=2.5, roll=0.0, start_xy=path[0], yaw=float(np.arctan2(d[1], d[0])))
    if who.startswith("rule:"):
        f = LeanFollower(path, np.array(json.loads(Path(who[5:]).read_text())["practised"]))
    else:
        pol = load(who)
        f = PPOFollower(path, pol, env, residual_gains=pol.residual_gains)
    head, cross, off = [], [], False
    info = {"progress": 0.0}
    while env.data.time < 120 and not env.fallen and not f.finished:
        a, info = f.action(s, env.data.qpos[:2])
        s = env.step(a)
        head.append(abs(info["heading_error"]))
        cross.append(info["cross_track"])
        if info["cross_track"] > 3.0:
            off = True
            break
    done = f.finished and not env.fallen and not off
    return {"who": who, "seed": seed, "route": route, "u_turn": route[1] == route[3] if len(route) == 5 else False,
            "completed": bool(done), "fell": bool(env.fallen), "off_road": off,
            "progress": 1.0 if done else float(info["progress"]), "time_s": round(env.data.time, 2),
            "mean_heading_error_deg": round(float(np.degrees(np.mean(head))), 2) if head else None,
            "mean_cross_track_m": round(float(np.mean(cross)), 3) if cross else None,
            "max_cross_track_m": round(float(np.max(cross)), 3) if cross else None}


def summarise(rows):
    ok = [r for r in rows if r["completed"]]
    u = [r for r in rows if r["u_turn"]]
    return {"completed": len(ok), "routes": len(rows), "u_turns_completed": sum(r["completed"] for r in u),
            "u_turns": len(u), "falls": sum(r["fell"] for r in rows), "off_road": sum(r["off_road"] for r in rows),
            "mean_progress": round(float(np.mean([r["progress"] for r in rows])), 3),
            "mean_heading_error_deg_completed": round(float(np.mean([r["mean_heading_error_deg"] for r in ok])), 2) if ok else None,
            "mean_cross_track_m_completed": round(float(np.mean([r["mean_cross_track_m"] for r in ok])), 3) if ok else None,
            "max_cross_track_m_completed": round(float(np.max([r["max_cross_track_m"] for r in ok])), 3) if ok else None}


def main():
    import bike_g1_turn_practice as tp
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--towns", type=int, nargs=2, required=True)
    ap.add_argument("--with-rule", action="store_true", help="also ride the practised rule (v3) on the same routes")
    ap.add_argument("--rule", default=str(ROOT / "results/bike_g1_turns_v3.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--vary-geometry", action="store_true", help="random corner (3.0-3.8 m) and U-turn (2.2-2.6 m) radii per route")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ckpts = sorted({p for c in args.ckpts for p in glob(c)})
    courses = tp.courses(range(*args.towns), u_turns=2)
    who = ckpts + (["rule:" + args.rule] if args.with_rule else [])
    with Pool(args.workers) as pool:
        rows = pool.map(ride, [(s, r, w, args.vary_geometry) for w in who for s, r in courses], chunksize=4)
    summary = {Path(w).name if not w.startswith("rule:") else "practised_rule_v3": summarise([r for r in rows if r["who"] == w])
               for w in who}
    for r in rows:
        r["who"] = Path(r["who"]).name if not r["who"].startswith("rule:") else "practised_rule_v3"
    out = {"towns": args.towns, "vary_geometry": args.vary_geometry, "summary": summary, "rides": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    for k, v in summary.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
