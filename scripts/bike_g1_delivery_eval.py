"""Parcel deliveries by bicycle in new towns: looking and replanning against riding blind.

    python scripts/bike_g1_delivery_eval.py --output results/bike_g1_delivery.json

Each final town (4500-4519, never used before) gets two deliveries from the depot to a numbered house.
The steering gains are the ones learned by practice (``results/bike_g1_turns.json``, frozen); the
blocked-street prior is counted in the practice towns 4000-4003. Two riders, same towns, same houses,
same steering:

- **looking**: looks down every street it plans to take (head camera), replans around barriers,
  changes lane around parked cars;
- **blind** (baseline): the shortest route on the street map, no looking (the obstacles are unknown to
  both until seen).

Measures: delivered (Wilson 95%), collisions, falls, leaving the road, time and distance of the
delivered rides, heading and cross-track error (as CycleRL reports them), the camera's street readings
against the hidden truth, and a paired sign test over the deliveries.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb, sqrt
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.bike.delivery import Delivery, blocked_prior

ARM_KP, STEER_KP = 5.0, 300.0
_CACHE = {}


def jobs_for(towns, per_town=2):
    from pai.bike.town import Town
    out = []
    for s in towns:
        rng = np.random.default_rng(s + 991)
        houses = rng.choice(sorted(Town(n=4, seed=s).houses), size=per_town, replace=False)
        out += [(s, int(h)) for h in houses]
    return out


def deliver(args):
    seed, house, blind, gains, prior = args[:5]
    ppo = args[5] if len(args) > 5 else None
    arm_steer = args[6] if len(args) > 6 else False
    if seed not in _CACHE or _CACHE.get("_ppo") != ppo or _CACHE.get("_arm") != arm_steer:
        from pai.bike.g1_rider import G1BikeEnv
        from pai.bike.navigation import HeadCamera
        from pai.bike.town import Town, town_scene
        town = Town(n=4, seed=seed)
        env = G1BikeEnv(scenery=town_scene(town), max_seconds=1e9, arm_kp=ARM_KP, steer_kp=STEER_KP,
                        arm_steer=arm_steer)
        follower = None
        if ppo is not None:          # the learned PPO riding policy instead of the practised rule
            sys.path.insert(0, str(ROOT / "scripts"))
            from bike_g1_ppo_eval import load
            from pai.bike.ppo_rider import PPOFollower
            policy = load(ppo)
            follower = lambda path, env=env: PPOFollower(path, policy, env, residual_gains=policy.residual_gains)
        _CACHE.clear()
        _CACHE["_ppo"] = ppo
        _CACHE["_arm"] = arm_steer
        _CACHE[seed] = Delivery(town, env, gains, prior, HeadCamera(env.model), follower=follower)
    r = _CACHE[seed].run(house, seed=seed, blind=blind)
    r["town"] = seed
    return r


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def sign_test(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarise(rows):
    ok = [r for r in rows if r["outcome"] == "delivered"]
    looks = sum(r["looks"] for r in rows)
    out = {"delivered": len(ok), "deliveries": len(rows), "wilson95": wilson(len(ok), len(rows))}
    for o in ("collision", "fell", "off_road", "timeout", "no_route"):
        out[o] = sum(r["outcome"] == o for r in rows)
    out["collisions_by_kind"] = {k: sum(r["hit"] == k for r in rows) for k in ("blocking", "partial")}
    if ok:
        out["mean_time_s_delivered"] = round(float(np.mean([r["time_s"] for r in ok])), 1)
        out["mean_distance_m_delivered"] = round(float(np.mean([r["distance_m"] for r in ok])), 1)
        out["mean_heading_error_deg_delivered"] = round(float(np.mean([r["mean_heading_error_deg"] for r in ok])), 2)
        out["mean_cross_track_m_delivered"] = round(float(np.mean([r["mean_cross_track_m"] for r in ok])), 3)
    out["replans"] = sum(r["replans"] for r in rows)
    out["street_looks"] = looks
    out["street_looks_wrong"] = sum(r["wrong_looks"] for r in rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gains-from", type=Path, default=ROOT / "results/bike_g1_turns_v3.json")
    ap.add_argument("--towns", type=int, nargs=2, default=[4540, 4560])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ppo", help="steer with this PPO checkpoint instead of the practised rule")
    ap.add_argument("--arm-steer", action="store_true", help="the G1's arms turn the handlebar (no fork servo)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    gains = np.array(json.loads(args.gains_from.read_text())["practised"])
    prior = blocked_prior(range(4000, 4004))
    jobs = jobs_for(range(*args.towns))
    with Pool(args.workers) as pool:
        # one town per worker task keeps a worker's cached scene: order jobs by town
        res = pool.map(deliver, [(s, h, blind, gains, prior, args.ppo, args.arm_steer) for s, h in jobs for blind in (False, True)],
                       chunksize=2)
    look = [r for r in res if not r["blind"]]
    blind = [r for r in res if r["blind"]]
    a = sum(x["outcome"] == "delivered" and y["outcome"] != "delivered" for x, y in zip(look, blind))
    b = sum(y["outcome"] == "delivered" and x["outcome"] != "delivered" for x, y in zip(look, blind))
    report = {"towns": args.towns, "steering": args.ppo and Path(args.ppo).name or "practised rule", "arm_steer": args.arm_steer, "gains": gains.round(4).tolist(), "blocked_prior": round(prior, 3),
              "arm_kp": ARM_KP, "steer_kp": STEER_KP,
              "summary": {"looking": summarise(look), "blind": summarise(blind)},
              "paired": {"looking_only": a, "blind_only": b, "sign_test_p": round(sign_test(a, b), 4)},
              "deliveries": res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    for k, v in report["summary"].items():
        print(k, v)
    print("paired", report["paired"])


if __name__ == "__main__":
    main()
