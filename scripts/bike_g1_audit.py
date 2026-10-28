"""Audit: are the bicycle rider's perception and decisions real? Three tests on development towns.

    python scripts/bike_g1_audit.py --towns 4300 4305 --output results/bike_g1_audit.json

- **Blindfold.** The head camera sees nothing (all depth points far away). If the rider's success comes
  from its camera, it should now do as badly as the blind rider (no looking at all), and its camera
  log should read every street as clear.
- **Tamper.** The town shown to the camera keeps its obstacles, but the hidden truth used for scoring
  and collisions is swapped for another town's obstacles on the same street map. If decisions come
  from the camera, its readings match what is shown, not the hidden truth, and the rider runs into
  obstacles it could not see.
- **Determinism.** The same delivery is ridden twice; outcome, time, route and every camera reading
  must be identical, so every reported ride and video can be reproduced from its command.

The truth guard (``pai/bike/town.py``: the hidden obstacles can only be read inside ``scoring()``)
is active throughout; any read by the rider raises an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.bike.delivery import Delivery, blocked_prior
from pai.bike.g1_rider import G1BikeEnv
from pai.bike.navigation import HeadCamera
from pai.bike.town import Town, scoring, town_scene

ARM_KP, STEER_KP = 5.0, 300.0


class Blindfold(HeadCamera):
    """A head camera that sees nothing: every depth reading is beyond the reading range."""

    def look(self, data, yaw):
        img, pts, z = super().look(data, yaw)
        return img * 0, pts, np.full_like(z, 1e3)


def readings(res):
    return [(e["street"], e["seen"], e["truth"]) for e in res["events"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--towns", type=int, nargs=2, default=[4300, 4305])
    ap.add_argument("--houses", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    gains = np.array(json.loads((ROOT / "results/bike_g1_turns_v3.json").read_text())["practised"])
    prior = blocked_prior(range(4000, 4004))
    rows = {"normal": [], "blindfold": [], "blind": [], "tamper": [], "determinism": []}
    for seed in range(*args.towns):
        town = Town(n=4, seed=seed)
        env = G1BikeEnv(scenery=town_scene(town), max_seconds=1e9, arm_kp=ARM_KP, steer_kp=STEER_KP)
        rng = np.random.default_rng(seed + 31)
        houses = [int(h) for h in rng.choice(sorted(town.houses), size=args.houses, replace=False)]
        normal = Delivery(town, env, gains, prior, HeadCamera(env.model))
        blindfold = Delivery(town, env, gains, prior, Blindfold(env.model))
        for h in houses:
            r = normal.run(h, seed=seed)
            rows["normal"].append({"town": seed, "house": h, "outcome": r["outcome"], "readings": readings(r)})
            r2 = normal.run(h, seed=seed)
            rows["determinism"].append({"town": seed, "house": h,
                                        "identical": (r2["outcome"], r2["time_s"], r2["events"]) == (r["outcome"], r["time_s"], r["events"])})
            b = blindfold.run(h, seed=seed)
            rows["blindfold"].append({"town": seed, "house": h, "outcome": b["outcome"], "readings": readings(b)})
            bl = normal.run(h, seed=seed, blind=True)
            rows["blind"].append({"town": seed, "house": h, "outcome": bl["outcome"]})
        # tamper: the scene keeps this town's obstacles; the hidden truth becomes another town's
        other = Town(n=4, seed=seed + 1000)
        shown = list(town.hidden)
        town.hidden = list(other.hidden)
        with scoring():
            hidden_truth = town.truth
        shown_truth = {"blocked": {o.edge for o in shown if o.kind == "blocking"},
                       "partial": {o.edge for o in shown if o.kind == "partial"}}
        for h in houses:
            r = normal.run(h, seed=seed)
            reads = []
            for (a, b_), seen, _ in readings(r):
                e = tuple(sorted((a, b_)))
                kind = lambda t: "blocking" if e in t["blocked"] else "partial" if e in t["partial"] else "clear"
                reads.append({"street": [a, b_], "seen": seen, "shown": kind(shown_truth), "hidden": kind(hidden_truth)})
            rows["tamper"].append({"town": seed, "house": h, "outcome": r["outcome"], "hit": r["hit"], "readings": reads})
        town.hidden = shown

    def rate(key):
        return sum(r["outcome"] == "delivered" for r in rows[key]), len(rows[key])

    reads_n = [x for r in rows["tamper"] for x in r["readings"]]
    summary = {
        "delivered": {k: rate(k) for k in ("normal", "blindfold", "blind", "tamper")},
        "collisions": {k: sum(r["outcome"] == "collision" for r in rows[k]) for k in ("normal", "blindfold", "blind", "tamper")},
        "blindfold_readings_clear": [sum(s == "clear" for _, s, _ in r["readings"]) for r in rows["blindfold"]],
        "blindfold_readings_total": [len(r["readings"]) for r in rows["blindfold"]],
        "tamper_readings_match_shown": sum(x["seen"] == x["shown"] for x in reads_n),
        "tamper_readings_match_hidden": sum(x["seen"] == x["hidden"] for x in reads_n),
        "tamper_readings_where_they_differ": sum(x["shown"] != x["hidden"] for x in reads_n),
        "tamper_where_differ_match_shown": sum(x["seen"] == x["shown"] for x in reads_n if x["shown"] != x["hidden"]),
        "tamper_readings": len(reads_n),
        "determinism_identical": sum(r["identical"] for r in rows["determinism"]),
        "determinism_runs": len(rows["determinism"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"towns": args.towns, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
