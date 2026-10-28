"""The G1 learns to balance the realistic bicycle by practice, then rides fresh starts.

    python scripts/bike_g1_practice.py --output results/bike_g1_practice.json

The rider's steering policy is the Stage 7 linear balance law (steer against roll, roll rate and
heading; drive torque toward a target speed) whose gains it improves by practice: a cross-entropy
search over the gains on practice starts (seeds 3000-3011, tilt up to +-0.1 rad, 2.5-4 m/s, 30 m),
keeping only what improves the practice score. Development starts 3100-3119 pick nothing; they only
check. The final ride is once, on new starts 3200-3239, for the practised gains, the untuned
Stage 7 gains and no steering, on the same starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.bike.g1_rider import G1BikeEnv

UNTUNED = np.array([2.0, 0.8, 0.15, 0.0, 3.0])   # roll, roll rate, yaw, steer damping, speed gain


def policy(gains, s, target):
    k_roll, k_rate, k_yaw, k_steer, k_speed = gains
    roll, rate, speed, steer, _, yaw = s
    torque = np.clip(k_speed * (target - speed), -12, 12)
    steering = np.clip(-k_roll * roll - k_rate * rate - k_yaw * yaw - k_steer * steer, -0.6, 0.6)
    return np.array([torque, steering, 0.0])


def start(seed):
    rng = np.random.default_rng(seed)
    return float(rng.uniform(-0.1, 0.1)), float(rng.uniform(2.5, 4.0))


def ride(env, gains, seed, steer=True):
    roll, speed = start(seed)
    s = env.reset(seed=seed, speed=speed, roll=roll)
    while not env.done:
        a = policy(gains, s, speed)
        if not steer:
            a[1] = 0.0
        s = env.step(a)
    return {"seed": seed, "start_roll": roll, "start_speed": speed, "distance": round(env.travelled, 2),
            "fell": env.fallen, "success": (not env.fallen) and env.travelled >= env.distance}


def score(env, gains, seeds):
    return float(np.mean([min(ride(env, gains, s)["distance"], env.distance) for s in seeds]))


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=15)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    env = G1BikeEnv(distance=30.0, max_seconds=20.0)
    practice = list(range(3000, 3012))
    rng = np.random.default_rng(0)
    mean, sd = UNTUNED.copy(), np.array([1.5, 0.6, 0.2, 0.5, 1.5])
    best, best_score = UNTUNED.copy(), score(env, UNTUNED, practice)
    history = [{"iteration": 0, "best_practice_m": best_score, "gains": best.round(3).tolist()}]
    for it in range(1, args.iterations + 1):
        cand = np.clip(mean + sd * rng.standard_normal((args.population, 5)), 0.0, None)
        scores = np.array([score(env, g, practice) for g in cand])
        elite = cand[np.argsort(scores)[-6:]]
        mean, sd = elite.mean(0), elite.std(0) + 0.02
        if scores.max() > best_score:
            best, best_score = cand[int(scores.argmax())].copy(), float(scores.max())
        history.append({"iteration": it, "best_practice_m": best_score, "gains": best.round(3).tolist()})
        print(f"iteration {it}: best practice mean distance {best_score:.2f} m", flush=True)
    dev = [ride(env, best, s) for s in range(3100, 3120)]
    final = {name: [ride(env, g, s, steer) for s in range(3200, 3240)]
             for name, g, steer in (("practised", best, True), ("untuned", UNTUNED, True),
                                    ("no_steering", UNTUNED, False))}
    summary = {name: {"success": sum(r["success"] for r in rows), "starts": len(rows),
                      "wilson95": wilson(sum(r["success"] for r in rows), len(rows)),
                      "mean_distance_m": round(float(np.mean([r["distance"] for r in rows])), 2)}
               for name, rows in final.items()}
    report = {"practice_seeds": [practice[0], practice[-1]], "dev_seeds": [3100, 3119],
              "final_seeds": [3200, 3239], "distance_m": 30.0, "practised_gains": best.round(4).tolist(),
              "untuned_gains": UNTUNED.tolist(), "history": history,
              "dev": {"success": sum(r["success"] for r in dev), "starts": len(dev)},
              "summary": summary, "final": final}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print("dev", report["dev"])
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
