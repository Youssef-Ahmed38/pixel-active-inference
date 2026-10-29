"""The G1 learns to steer its bicycle through town corners by practice.

    python scripts/bike_g1_turn_practice.py --output results/bike_g1_turns.json

Routes: random walks of 3-4 street segments with at least one 90-degree turn, in the right-hand lane
with corners rounded to 3.4 m, starting at 2.5 m/s. Practice towns 4000-4003 (two routes each); a
cross-entropy search over the eight parameters of ``pai/bike/riding.LeanFollower`` keeps only
improvements of the practice score (mean route progress; a fall, or leaving the route by more than
3 m, ends the route). Development towns 4100-4104 only check. Final once on new towns
4200-4209 (two routes each; ``--final-towns``) for the practised and the default parameters, with CycleRL-style measures:
completion, falls, mean absolute heading error, mean and maximum cross-track error.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.bike.riding import CORNER_R, LEAN_DEFAULT as DEFAULT, LeanFollower, lane_path

_ENVS = {}
ARM_KP = 5.0
ARM_STEER = False         # True: the G1's arms (and waist yaw) turn the handlebar; the fork servo is off
LANE_CHANGES = False      # True: half of the routes change lane around 1-2 parked cars (as in deliveries)
STEER_KP = 300.0
OFF_ROAD_M = 3.0          # further than this from the route ends the ride (left the road)
LO = np.array([0, 0, 0, 0, 1, 1.0, 1.5, 1.0])
HI = np.array([12, 4, 1.5, 2.5, 12, 8.0, 4.5, 3.5])


def routes_for(town, rng, k=2):
    g = town.graph
    out = []
    while len(out) < k:
        r = [int(rng.choice(list(town.nodes)))]
        for _ in range(int(rng.integers(3, 5))):
            nxt = [n for n in g[r[-1]] if len(r) < 2 or n != r[-2]]
            r.append(int(rng.choice(nxt)))
        pts = [np.array(town.nodes[n]) for n in r]
        dirs = [tuple(np.sign(b - a)) for a, b in zip(pts[:-1], pts[1:])]
        if any(d1 != d2 for d1, d2 in zip(dirs[:-1], dirs[1:])):
            out.append(r)
    return out


def u_turn_routes(town, rng, k=2):
    """Routes with a U-turn at an intersection: a -> b -> c, back to b, then on to another street."""
    g = town.graph
    out = []
    while len(out) < k:
        a = int(rng.choice(list(town.nodes)))
        b = int(rng.choice(g[a]))
        c = int(rng.choice([n for n in g[b] if n != a] or g[b]))
        d = [n for n in g[b] if n != c]
        out.append([a, b, c, b, int(rng.choice(d))])
    return out


def _init_worker(arm_steer, lane_changes=False):
    global ARM_STEER, LANE_CHANGES
    ARM_STEER = arm_steer
    LANE_CHANGES = lane_changes


def with_cars(path, seed, route):
    """For half of the routes (chosen by a hash of town and route, the same in every run): 1-2 parked
    cars in the rider's lane at straight places, the path changing lane around them as the delivery
    planner does (``pai/bike/navigation.avoid_cars``), often just after a corner."""
    from pai.bike.navigation import avoid_cars
    from pai.bike.riding import curvature
    g = np.random.default_rng(seed * 1_000_003 + sum((i + 1) * 97 * n for i, n in enumerate(route)) + 7)
    if g.random() < 0.5:
        return path
    kap = np.abs(curvature(path))
    ok = [k for k in range(16, len(path) - 16) if kap[max(0, k - 2):k + 3].max() < 0.02]
    if not ok:
        return path
    cars = [tuple(path[int(g.choice(ok))]) for _ in range(int(g.integers(1, 3)))]
    return avoid_cars(path, cars)


def env_for(seed):
    if seed not in _ENVS:
        from pai.bike.g1_rider import G1BikeEnv
        from pai.bike.town import Town, town_scene
        town = Town(n=4, seed=seed)
        _ENVS[seed] = (town, G1BikeEnv(scenery=town_scene(town), max_seconds=1e9, arm_kp=ARM_KP, steer_kp=STEER_KP,
                                       arm_steer=ARM_STEER))
    return _ENVS[seed]


def ride(args):
    seed, route, gains, max_t = args
    town, env = env_for(seed)
    path = lane_path(town.nodes, route, radius=CORNER_R)
    if LANE_CHANGES:
        path = with_cars(path, seed, route)
    f = LeanFollower(path, gains)
    d = path[min(5, len(path) - 1)] - path[0]
    s = env.reset(seed=seed, speed=2.5, roll=0.0, start_xy=path[0], yaw=float(np.arctan2(d[1], d[0])))
    head, cross = [], []
    info = {"progress": 0.0}
    off = False
    while env.data.time < max_t and not env.fallen and not f.finished:
        a, info = f.action(s, env.data.qpos[:2])
        s = env.step(a)
        head.append(abs(info["heading_error"]))
        cross.append(info["cross_track"])
        if info["cross_track"] > OFF_ROAD_M:
            off = True
            break
    done = f.finished and not env.fallen and not off
    return {"seed": seed, "route": route, "completed": bool(done), "fell": bool(env.fallen), "off_road": off,
            "progress": 1.0 if done else float(info["progress"]), "time_s": round(env.data.time, 2),
            "mean_heading_error_deg": round(float(np.degrees(np.mean(head))), 2) if head else None,
            "mean_cross_track_m": round(float(np.mean(cross)), 3) if cross else None,
            "max_cross_track_m": round(float(np.max(cross)), 3) if cross else None}


def courses(seeds, k=2, u_turns=0):
    from pai.bike.town import Town
    out = []
    for s in seeds:
        out += [(s, r) for r in routes_for(Town(n=4, seed=s), np.random.default_rng(s + 77), k)]
        if u_turns:
            out += [(s, r) for r in u_turn_routes(Town(n=4, seed=s), np.random.default_rng(s + 555), u_turns)]
    return out


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=12)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start-from", type=Path, help="start the search from the gains in this result file")
    ap.add_argument("--track-weight", type=float, default=0.0,
                    help="practice score = progress - weight x mean cross-track error (m), capped at 2 m")
    ap.add_argument("--final-towns", type=int, nargs=2, default=[4200, 4210])
    ap.add_argument("--seed", type=int, default=0, help="seed of the cross-entropy search (practice runs)")
    ap.add_argument("--arm-steer", action="store_true", help="the G1's arms steer (no fork servo)")
    ap.add_argument("--lane-changes", action="store_true", help="half of the routes change lane around parked cars")
    ap.add_argument("--u-turns", type=int, default=0, help="U-turn routes per town (practice, dev, final)")
    args = ap.parse_args()
    global ARM_STEER, LANE_CHANGES
    ARM_STEER = args.arm_steer
    LANE_CHANGES = args.lane_changes
    u = args.u_turns
    practice, dev, final = (courses(range(4000, 4004), u_turns=u), courses(range(4100, 4105), u_turns=u),
                            courses(range(*args.final_towns), u_turns=u))
    rng = np.random.default_rng(args.seed)
    start = DEFAULT.copy() if args.start_from is None else np.array(json.loads(args.start_from.read_text())["practised"])
    lo, hi = LO, HI
    if args.u_turns:
        if len(start) == 8:
            start = np.append(start, start[7])          # U-turn speed starts at the corner speed
        lo, hi = np.append(LO, 1.2), np.append(HI, 3.5)
    mean, sd = start.copy(), (hi - lo) / (6 if args.start_from is None else 12)

    def value(x):
        ct = x["mean_cross_track_m"] if x["mean_cross_track_m"] is not None else 2.0
        return x["progress"] - args.track_weight * min(ct, 2.0)
    with Pool(args.workers, initializer=_init_worker, initargs=(args.arm_steer, args.lane_changes)) as pool:
        def score(g):
            res = pool.map(ride, [(s, r, g, 90.0) for s, r in practice])
            return float(np.mean([value(x) for x in res]))
        best, best_score = start.copy(), score(start)
        history = [{"iteration": 0, "best_practice_score": best_score, "gains": best.round(3).tolist()}]
        print(f"iteration 0: starting practice score {best_score:.3f}", flush=True)
        for it in range(1, args.iterations + 1):
            cand = np.clip(mean + sd * rng.standard_normal((args.population, len(mean))), lo, hi)
            jobs = [(s, r, g, 90.0) for g in cand for s, r in practice]
            res = pool.map(ride, jobs, chunksize=2)
            scores = np.array([np.mean([value(x) for x in res[i * len(practice):(i + 1) * len(practice)]])
                               for i in range(len(cand))])
            elite = cand[np.argsort(scores)[-6:]]
            mean, sd = elite.mean(0), elite.std(0) + 0.02 * (hi - lo)
            if scores.max() > best_score:
                best, best_score = cand[int(scores.argmax())].copy(), float(scores.max())
            history.append({"iteration": it, "best_practice_score": best_score, "gains": best.round(3).tolist()})
            print(f"iteration {it}: best practice score {best_score:.3f}", flush=True)
        dev_res = pool.map(ride, [(s, r, best, 120.0) for s, r in dev])
        dev_start = pool.map(ride, [(s, r, start, 120.0) for s, r in dev])
        final_res = {name: pool.map(ride, [(s, r, g, 120.0) for s, r in final])
                     for name, g in (("practised", best), ("default", DEFAULT))
                     + ((("starting_gains", start),) if args.start_from is not None else ())}
    summary = {}
    for name, rows in final_res.items():
        k = sum(r["completed"] for r in rows)
        ok = [r for r in rows if r["completed"]]
        summary[name] = {"completed": k, "routes": len(rows), "wilson95": wilson(k, len(rows)),
                         "falls": sum(r["fell"] for r in rows), "off_road": sum(r["off_road"] for r in rows),
                         "mean_heading_error_deg_completed": round(float(np.mean([r["mean_heading_error_deg"] for r in ok])), 2) if ok else None,
                         "mean_cross_track_m_completed": round(float(np.mean([r["mean_cross_track_m"] for r in ok])), 3) if ok else None,
                         "max_cross_track_m_completed": round(float(np.max([r["max_cross_track_m"] for r in ok])), 3) if ok else None}
    report = {"follower": "LeanFollower", "arm_kp": ARM_KP, "steer_kp": STEER_KP, "arm_steer": ARM_STEER, "lane_changes": LANE_CHANGES, "corner_radius_m": CORNER_R,
              "practice_towns": [4000, 4003], "dev_towns": [4100, 4104], "final_towns": args.final_towns,
              "gains_names": ["roll", "roll_rate", "heading_to_lean", "steer_feedforward", "speed", "lookahead_m",
                              "cruise_mps", "corner_mps"] + (["u_turn_mps"] if args.u_turns else []),
              "practised": best.round(4).tolist(), "default": DEFAULT.tolist(), "history": history,
              "dev": {"completed": sum(r["completed"] for r in dev_res), "routes": len(dev_res),
                    "mean_cross_track_m": round(float(np.mean([r["mean_cross_track_m"] for r in dev_res])), 3),
                    "starting_gains_completed": sum(r["completed"] for r in dev_start),
                    "starting_gains_mean_cross_track_m": round(float(np.mean([r["mean_cross_track_m"] for r in dev_start])), 3)},
            "track_weight": args.track_weight, "search_seed": args.seed, "u_turns_per_town": u, "start_from": None if args.start_from is None else args.start_from.name,
              "summary": summary, "final": final_res}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1))
    print("dev", report["dev"])
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
