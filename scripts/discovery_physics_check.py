"""Physics check of PhysicsWorld: hand-written correct action sequences through the discovery interface.

    python scripts/discovery_physics_check.py                       # every train type, lock state and place
    python scripts/discovery_physics_check.py --types hd_knob_pull_left --states key --places drawer -v

Like the scripted oracle (pai.envs.home_oracle), this reads the ground truth (PhysicsWorld.truth(): which
part is the handle, which key fits, which way things turn) and writes down the solution; it only checks
that the actions of pai.discovery.physics carry it out and that features / outcomes say what they should
at each step (a latched door stalls; a wrong key goes in but will not turn; the hidden key is revealed
when the drawer opens). The discovery agent never does this. Prints a success table and per-action timing.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.discovery.physics import PhysicsWorld  # noqa: E402

TYPES = ["hd_knob_pull_left", "hd_knob_push_right", "hd_lever_pull_right", "hd_pushbar_push_left", "hd_ball_pull_left"]


def plan(w: PhysicsWorld, wrong_key: bool) -> list[tuple]:
    """(action or callable -> action, expectation, label) steps for the episode's ground truth."""
    tr = w.truth()
    ids = {v: k for k, v in tr["ids"].items()}
    p = tr["params"]
    push = p["swing"] == "push"
    h, door = ids["handle"], ids["door"]
    if p["handle"] == "push_bar":
        open_act = ("press_push", h)
    else:
        open_act = ("turn_push" if push else "turn_pull", h, 1)
    naive = ("pull", h)          # a latched pull door or any push door stalls
    opened = lambda o, w: w.goal_reached()            # noqa: E731
    stalls = lambda o, w: o.stalled and not w.goal_reached()   # noqa: E731
    steps = [(naive, stalls, "plain pull stalls")]
    locked = tr["deadbolt_locked"] or tr["key_locked"]
    if locked:
        steps.append((open_act, stalls, "right probe stalls (locked)"))
    if tr["key_locked"]:
        good = ids[next(n for n, k in tr["keys"].items() if k["matches"])]
        bad = [ids[n] for n, k in tr["keys"].items() if not k["matches"]]
        cyl = ids["cylinder"]
        if tr["key_place"] == "drawer":
            steps.append((("pull", ids["drawer"]), lambda o, w, g=good: any(e.id == g for e in o.revealed),
                          "drawer opens, key revealed"))
        if tr["key_place"] == "other_room":
            visible = [e.id for e in w.entities() if e.kind == "object"]
            steps.append((None, lambda o, w, g=good: g not in visible, "fitting key not in view"))
            return steps + [(open_act, stalls, "still locked (no solution)")]
        if wrong_key:
            vis_bad = [b for b in bad if b in {e.id for e in w.entities() if e.kind == "object"}]
            if vis_bad:
                b = vis_bad[0]
                steps += [(("pick", b), lambda o, w, b=b: w.features().get(f"obj:{b}.where") == "held", "pick wrong key"),
                          (("insert", b, cyl), lambda o, w, b=b, c=cyl: w.features().get(f"obj:{b}.where") == f"in:{c}",
                           "wrong key goes in"),
                          (("turn_held", b, 1), lambda o, w: o.stalled, "wrong key will not turn"),
                          (("pick", b), lambda o, w, b=b: w.features().get(f"obj:{b}.where") == "held", "withdraw it"),
                          (("release", b), lambda o, w, b=b: w.features().get(f"obj:{b}.where") == "free", "put it back")]
        steps += [(("pick", good), lambda o, w, g=good: w.features().get(f"obj:{g}.where") == "held", "pick key"),
                  (("turn_held", good, 1), lambda o, w, g=good: o.executed and not o.stalled and not o.changed
                   and w.features().get("body:hand") == g, "turn it in the hand (no effect)"),
                  (("insert", good, cyl), lambda o, w, g=good, c=cyl: w.features().get(f"obj:{g}.where") == f"in:{c}",
                   "key goes in"),
                  (("turn_held", good, 1), lambda o, w, g=good: not o.stalled and w.features().get(f"obj:{g}.angle") == "pos",
                   "key turns"),
                  (("release", good), lambda o, w, g=good, c=cyl: w.features().get(f"obj:{g}.where") == f"in:{c}",
                   "let go (key stays in)")]
    if tr["deadbolt_locked"]:
        t = ids["thumb"]
        steps.append((("turn", t, -1), lambda o, w, t=t: not o.stalled and w.features().get(f"part:{t}.angle") == "rest",
                      "thumb-turn back"))
    steps.append((open_act, opened, "door opens"))
    return steps


def run_case(w: PhysicsWorld, seed: int, state: str, place: str, wrong_key: bool, verbose: bool, timing: dict) -> dict:
    w.reset(seed, lock_state=state, key_place=place)
    rows, ok_all = [], True
    for act, check, label in plan(w, wrong_key):
        if act is None:
            ok = bool(check(None, w))
            rows.append((label, "-", ok, 0.0, 0.0, ""))
        else:
            o = w.execute(act)
            ok = bool(check(o, w))
            timing[act[0]].append((o.cost, w.last_wall))
            rows.append((label, act, ok, o.cost, w.last_wall, f"exec={int(o.executed)} stall={int(o.stalled)} "
                         f"{o.notes} changed={ {k: v[1] for k, v in o.changed.items()} }"))
        ok_all &= ok
        if verbose:
            r = rows[-1]
            print(f"      {'ok ' if r[2] else 'BAD'} {r[0]:30s} {str(r[1]):32s} {r[3]:5.1f}s sim {r[4]:4.1f}s wall  {r[5]}",
                  flush=True)
    opened = w.goal_reached()                # the goal condition on the features, after the last action
    fallen = sum(float(w.d.xpos[k["body"]][2]) < 0.3 for k in w.objects)   # knocked to the floor on the way
    return {"opened": opened, "as_expected": ok_all, "solvable": w.truth()["solvable"], "rows": rows,
            "sim": sum(r[3] for r in rows), "fallen": fallen}


def main() -> None:
    warnings.filterwarnings("ignore", message="Attach conflict")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", nargs="*", default=TYPES)
    ap.add_argument("--embodiment", default="robotiq_2f85")
    ap.add_argument("--states", nargs="*", default=["unlocked", "deadbolt", "key", "both"])
    ap.add_argument("--places", nargs="*", default=["table", "shelf", "drawer", "other_room"])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--n-keys", type=int, default=3, help="keys per scene (>1: a wrong key is tried first)")
    ap.add_argument("--no-wrong-key", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    timing: dict = defaultdict(list)
    results = []
    t0 = time.time()
    for t in args.types:
        w = PhysicsWorld(t, args.embodiment, seed=0, n_keys=args.n_keys)
        for st in args.states:
            if st not in w.rt.available_states():
                continue
            for pl in (args.places if st in ("key", "both") else ["table"]):
                for s in range(args.seeds):
                    if args.verbose:
                        print(f"  {t} {st}/{pl} seed {s}", flush=True)
                    r = run_case(w, s, st, pl, not args.no_wrong_key, args.verbose, timing)
                    key = f"{st}/{pl}" if st in ("key", "both") else st
                    results.append((t, key, s, r))
                    bad = [row[0] for row in r["rows"] if not row[2]]
                    print(f"  {t:22s} {key:18s} seed {s}: opened={int(r['opened'])} solvable={int(r['solvable'])} "
                          f"as_expected={int(r['as_expected'])} sim={r['sim']:.0f}s"
                          + (f"  unexpected: {bad}" if bad else ""), flush=True)
        w.env.close()

    print(f"\n{'type':22s} {'case':18s} {'seed':>4s} {'opened':>7s} {'expected':>9s} {'actions':>8s} {'sim s':>6s} "
          f"{'fallen':>7s}")
    for t, key, s, r in results:
        print(f"{t:22s} {key:18s} {s:4d} {int(r['opened']):7d} {int(r['as_expected']):9d} {len(r['rows']):8d} "
              f"{r['sim']:6.0f} {r['fallen']:7d}")
    print("(fallen: keys knocked to the floor during the episode)")
    solv = [r for *_, r in results if r["solvable"]]
    uns = [r for *_, r in results if not r["solvable"]]
    print(f"\nsolvable: opened {sum(r['opened'] for r in solv)}/{len(solv)}, every step as expected "
          f"{sum(r['as_expected'] for r in solv)}/{len(solv)}")
    if uns:
        print(f"unsolvable (key in the other room): opened {sum(r['opened'] for r in uns)}/{len(uns)}, "
              f"as expected {sum(r['as_expected'] for r in uns)}/{len(uns)}")
    print(f"\n{'action':12s} {'n':>4s} {'sim s mean':>11s} {'sim s max':>10s} {'wall s mean':>12s}")
    for k, v in sorted(timing.items()):
        sim = [a for a, _ in v]
        wall = [b for _, b in v]
        print(f"{k:12s} {len(v):4d} {sum(sim) / len(v):11.2f} {max(sim):10.2f} {sum(wall) / len(v):12.2f}")
    print(f"\ntotal wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
