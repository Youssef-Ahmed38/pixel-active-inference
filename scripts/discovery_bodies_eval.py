"""The discovery agent with other bodies: the same agent, unchanged, in PhysicsWorld with decoys on.

    python scripts/discovery_bodies_eval.py oracle          # the scripted oracle per body (physical upper reference)
    python scripts/discovery_bodies_eval.py agent           # the discovery agent per body
    python scripts/discovery_bodies_eval.py gif --body shadow --case latch --seed 200000
    python scripts/discovery_bodies_eval.py report          # results/discovery_bodies/report.md from the JSON

Bodies (pai.envs.embodiments): the Robotiq gripper (the reference the other physics numbers use), the
Panda arm with its parallel gripper, the Allegro and LEAP hands (4 fingers), the Shadow hand (5, human-like)
and the G1 humanoid with its Dex3 hand. Every body gets the same PhysicsWorld skills (the world side of
pai.discovery.interface): nothing about the agent, its parameters, the skills or what it sees changes
with the body. The scene is the home door with the look-alike decoys on (home_doors.DECOYS) and 2 keys.

Cases (as in scripts/discovery_eval.py, over the physical lock states):

    unlocked     no working latch (pai.discovery.bodies.BodyWorld(latch=False)): any move of the door opens it
    latch        the spring latch only (the home-door lock state "unlocked")
    deadbolt     latch + thumb-turn deadbolt thrown
    key_table    latch + key deadbolt, the fitting key on the cabinet top
    key_drawer   the same, the fitting key in the closed drawer
    no_solution  key-locked, the fitting key in the other room

Train types and fresh seeds (offset 200000: never used while the agent or the physics was developed).
Metrics per body: success, actions and simulated seconds to success, explanation accuracy
(pai.discovery.explain.score against the truth), and for each solvable door that stayed shut the cause
of the failure (pai.discovery.bodies.failure_cause: skill not executed / stalled / reasoning / unstable /
timeout) from the recorded outcomes and the simulator's events. The oracle (pai.envs.home_oracle, reads
the ground truth, fixed plan) on the same scenes is the physical upper reference of each body.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "discovery_bodies"
MEDIA = ROOT / "docs" / "media" / "discovery"
BODIES = ("robotiq_2f85", "panda_2f", "allegro", "leap", "shadow", "g1_hands")
TYPES = ("hd_knob_pull_left", "hd_lever_pull_right")
CASES = ("unlocked", "latch", "deadbolt", "key_table", "key_drawer", "no_solution")
SEED0 = 200000
# case -> (home-door lock state, key place, latch works)
CASE_SETUP = {"unlocked": ("unlocked", "table", False), "latch": ("unlocked", "table", True),
              "deadbolt": ("deadbolt", "table", True), "key_table": ("key", "table", True),
              "key_drawer": ("key", "drawer", True), "no_solution": ("key", "other_room", True)}
FIXTURE = {"n_keys": 2, "decoys": True}


def pool_map(fn, jobs: list, processes: int, label: str, sink=None) -> list:
    """fn over jobs in any order, with progress; sink(result) is called as each one finishes."""
    t0 = time.time()
    out = []
    if processes <= 1:
        it, pool = map(fn, jobs), None
    else:
        from multiprocessing import get_context
        pool = get_context("spawn").Pool(processes, maxtasksperchild=1)
        it = pool.imap_unordered(fn, jobs, chunksize=1)
    try:
        for k, r in enumerate(it):
            out.append(r)
            if sink is not None:
                sink(r)
            print(f"  {label}: {k + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


# ---------------------------------------------------------------------------------------- oracle
def _oracle_job(job: tuple) -> dict:
    warnings.filterwarnings("ignore")
    from pai.discovery.bodies import free_latch
    from pai.envs.door_scene import DoorSceneEnv
    from pai.envs.home_doors import home_door_fixture
    from pai.envs.home_oracle import run_oracle
    body, type_name, case, seed = job
    st, place, latch = CASE_SETUP[case]
    t0 = time.time()
    env = DoorSceneEnv(body, home_door_fixture(type_name, **FIXTURE), seed=seed)
    if not latch:
        free_latch(env.runtime)
    r = run_oracle(env, seed=seed, lock_state=st, key_place=place)
    env.close()
    out = {"body": body, "type": type_name, "case": case, "seed": seed, "opened": bool(r["opened"]),
           "stage": r["stage"], "why": r["why"], "stable": bool(r["stable"]), "sim": round(r["steps"] * 0.02, 1),
           "wall": round(time.time() - t0, 1), "solvable": case != "no_solution"}
    print(f"  oracle {body:13s} {type_name:20s} {case:11s} {seed}: opened={int(out['opened'])} {out['why']}", flush=True)
    return out


def run_oracle_table(args) -> None:
    """Runs are appended to oracle.partial.jsonl as they finish; a rerun skips the ones already there."""
    OUT.mkdir(parents=True, exist_ok=True)
    part = OUT / "oracle.partial.jsonl"
    done = {}
    if part.exists():
        for ln in part.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[_key(r)] = r
    jobs = [(b, t, c, SEED0 + s) for b in args.bodies for t in args.types for c in args.cases for s in range(args.seeds)]
    todo = [j for j in jobs if "|".join(map(str, j)) not in done]
    print(f"{len(jobs) - len(todo)} runs already done, {len(todo)} to run", flush=True)

    def sink(r):
        with part.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
    t0 = time.time()
    pool_map(_oracle_job, todo, args.processes, "oracle", sink)
    for ln in part.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            done[_key(r)] = r
    rows = [done["|".join(map(str, j))] for j in jobs if "|".join(map(str, j)) in done]
    out = {"config": {"bodies": args.bodies, "types": args.types, "cases": args.cases, "seeds": args.seeds,
                      "seed0": SEED0, "fixture": FIXTURE, "wall_s": round(time.time() - t0)},
           "runs": sorted(rows, key=lambda r: (BODIES.index(r["body"]), r["type"], CASES.index(r["case"]), r["seed"]))}
    (OUT / "oracle.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / 'oracle.json'} ({out['config']['wall_s']} s)")


# ---------------------------------------------------------------------------------------- agent
def _truth(pw) -> dict:
    from discovery_eval import physics_truth
    t = physics_truth(pw)
    t["latched"] = bool(pw.latch)
    return t


def _agent_job(job: tuple) -> dict:
    warnings.filterwarnings("ignore")
    from pai.discovery.agent import DiscoveryAgent
    from pai.discovery.bodies import BodyWorld, TracedWorld, WallLimit, failure_cause
    from pai.discovery.explain import explain, score
    body, type_name, case, seed, budget, limit = job
    st, place, latch = CASE_SETUP[case]
    t0 = time.time()
    pw = BodyWorld(type_name, body, seed=seed, lock_state=st, key_place=place, latch=latch, **FIXTURE)
    w = TracedWorld(pw, limit)
    init = w.bolts()
    truth = _truth(pw)
    r = {"body": body, "type": type_name, "case": case, "seed": seed, "solvable": bool(truth["solvable"])}
    try:
        log = DiscoveryAgent(seed=seed).run(w, budget)
        stopped = log.stopped_reason
        e = explain(log)
        s = score(e.claims, truth, log)
        r.update(success=bool(log.reached_goal), acc=s["accuracy"], rel=s["relevant"], false=s["false_claims"],
                 text=e.text)
    except WallLimit:
        stopped = "timeout"
        r.update(success=False, acc=None, rel=None, false=0, text="")
    tr = w.trace
    r.update(stopped=stopped, actions=len(tr), sim=round(sum(t["cost"] for t in tr), 1),
             wall=round(time.time() - t0, 1), unstable=bool(pw.unstable), opened_ever=bool(pw.truth()["opened"]))
    r["n_not_executed"] = sum(not t["executed"] for t in tr)
    r["n_stalled"] = sum(t["executed"] and t["stalled"] for t in tr)
    r["n_aborted"] = sum(t["notes"].startswith("aborted") for t in tr)
    r["notes_failed"] = dict(Counter(t["notes"] for t in tr if not t["executed"]))
    r["failure"] = failure_cause(tr, pw, init, stopped) if r["solvable"] and not r["success"] else None
    # the unlocking directions the failure analysis assumes, checked on every retraction
    r["unlock_signs"] = [[t["a"][0], t["a"][2] if len(t["a"]) > 2 else None, e["bolt"]]
                         for t in tr for e in t["events"] if e["type"] == "bolt_retracted"]
    r["trace"] = tr
    pw.env.close()
    f = r["failure"] or {}
    print(f"  agent {body:13s} {type_name:20s} {case:11s} {seed}: success={int(r['success'])} actions={r['actions']} "
          f"sim={r['sim']:.0f}s wall={r['wall']:.0f}s stopped={stopped} cause={f.get('cause')}@{f.get('blocked_at')}",
          flush=True)
    return r


def _key(r) -> str:
    return f"{r['body']}|{r['type']}|{r['case']}|{r['seed']}"


def run_agent(args) -> None:
    """Episodes are appended to <out>.partial.jsonl as they finish; a rerun with the same --tag skips the
    ones already there (resume). At the end all of them go into <out>.json and <out>_traces.json.gz."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"agent_{args.tag}.json" if args.tag else "agent.json")
    part = path.with_suffix(".partial.jsonl")
    done = {}
    if part.exists():
        for ln in part.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[_key(r)] = r
    nseeds = {b: args.seeds for b in args.bodies}
    nseeds.update({b: int(n) for b, n in (x.split("=") for x in args.seeds_for)})
    jobs = [(b, t, c, SEED0 + s, args.budget, args.limit)
            for b in args.bodies for t in args.types for c in args.cases for s in range(nseeds[b])]
    todo = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}|{j[3]}" not in done]
    # the slowest bodies and cases first, so the pool does not end on one long episode
    slow = {"g1_hands": 0, "shadow": 1, "leap": 2, "allegro": 3, "panda_2f": 4, "robotiq_2f85": 5}
    todo.sort(key=lambda j: (CASES.index(j[2]) < 2, slow.get(j[0], 9)))
    print(f"{len(done)} episodes already done, {len(todo)} to run", flush=True)

    def sink(r):
        with part.open("a") as fh:
            fh.write(json.dumps(r) + "\n")
    t0 = time.time()
    for r in pool_map(_agent_job, todo, args.processes, "agent", sink):
        done[_key(r)] = r
    want = {f"{j[0]}|{j[1]}|{j[2]}|{j[3]}" for j in jobs}
    rows = [done[k] for k in done if k in want]
    traces = {_key(r): r.pop("trace") for r in rows}
    walls = json.loads(path.read_text())["config"].get("wall_s_runs", []) if path.exists() else []
    cfg = {"bodies": sorted({r["body"] for r in rows}, key=BODIES.index), "types": args.types, "cases": args.cases,
           "seeds": args.seeds, "seeds_per_body": nseeds, "seed0": SEED0, "budget": args.budget, "limit_s": args.limit, "fixture": FIXTURE,
           "processes": args.processes, "wall_s_runs": walls + [round(time.time() - t0)]}
    out = {"config": cfg, "episodes": sorted(rows, key=lambda r: (BODIES.index(r["body"]), r["type"],
                                                                   CASES.index(r["case"]), r["seed"]))}
    path.write_text(json.dumps(out, indent=1))
    tpath = OUT / (path.stem + "_traces.json.gz")
    tpath.write_bytes(gzip.compress(json.dumps(traces).encode()))
    print(f"wrote {path} and {tpath} ({cfg['wall_s_runs'][-1]} s)")


# ---------------------------------------------------------------------------------------- the GIF
def run_gif(args) -> None:
    """One episode of the agent with a body, recorded as a GIF with the action, what changed and the most
    probable rule overlaid (as scripts/discovery_eval.py gif), then its explanation."""
    warnings.filterwarnings("ignore")
    from PIL import Image
    from discovery_eval import Recorder, _describe, _panel, _rule
    from pai.discovery.agent import DiscoveryAgent
    from pai.discovery.bodies import BodyWorld, TracedWorld
    from pai.discovery.explain import explain
    FG, DIM, OK, WARN = (235, 235, 235), (150, 155, 165), (110, 200, 120), (235, 150, 80)
    st, place, latch = CASE_SETUP[args.case]
    pw = BodyWorld(args.type, args.body, seed=args.seed, lock_state=st, key_place=place, latch=latch, **FIXTURE)
    rec = Recorder(pw, size=args.size, busy_frames=args.busy_frames)
    agent = DiscoveryAgent(seed=args.seed)
    rec.agent = agent
    w = TracedWorld(pw)

    class _Rec:            # Recorder around the traced world
        def __getattr__(self, n):
            return getattr(w, n)

        def execute(self, a):
            rec.begin(a)
            out = w.execute(a)
            rec.end(out)
            return out
    t0 = time.time()
    log = agent.run(_Rec(), args.budget)
    print(f"episode: success={log.reached_goal} actions={log.n_actions} wall={time.time() - t0:.0f}s")
    ents = dict(log.entities)
    frames = []
    n_all = len(rec.actions)
    boring = [i for i, x in enumerate(rec.actions) if not x[4]]
    keep_boring = set(boring[::max(1, int(np.ceil(len(boring) / args.boring_frames)))])
    for i, (a, out, fr, top, busy) in enumerate(rec.actions):
        if not busy and i not in keep_boring:
            continue
        res = ("stalled" if out.stalled else "done") if out.executed else "skill failed"
        ch = ", ".join(f"{k.split(':', 1)[1]}: {v[1]}" for k, v in out.changed.items() if not k.startswith("body:"))
        lines = [(f"{args.body} | action {i + 1}/{n_all}: {_describe(a, ents)} -> {res}", FG),
                 (f"changed: {ch}" if ch else "changed: nothing", OK if ch else DIM),
                 (f"best rule {_rule(top[0][0] if top[0] else None, ents)}", WARN),
                 (f"P(cause is something I have not thought of) = {top[1]:.2f}", DIM)]
        for f in (fr if busy else fr[-1:]):
            frames.append(_panel(f, lines, args.size))
    if len(frames) > args.max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, args.max_frames).astype(int)]
    text = explain(log).text
    frames += [_panel(rec.actions[-1][2][-1], [(f"{args.body}: " + text, FG)], args.size)] * 20
    MEDIA.mkdir(parents=True, exist_ok=True)
    path = MEDIA / f"{args.body}_{args.case}.gif"
    ims = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=160, loop=0, optimize=True)
    meta = {"body": args.body, "type": args.type, "case": args.case, "seed": args.seed, "success": bool(log.reached_goal),
            "actions": n_all, "sim_s": round(float(sum(x[1].cost for x in rec.actions)), 1), "frames": len(frames),
            "bytes": path.stat().st_size, "explanation": text, "actions_list": [[*x[0]] for x in rec.actions]}
    (MEDIA / (path.stem + ".json")).write_text(json.dumps(meta, indent=1))
    pw.env.close()
    print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------------------- report
def _mean(x):
    x = [v for v in x if v is not None]
    return f"{np.mean(x):.1f}" if x else "-"


def _part_names(scenes: list) -> dict:
    """"type|seed" -> {entity id: ground-truth name} (scoring only; the ids depend on the scene, not the body),
    cached in <OUT>/ids.json."""
    path = OUT / "ids.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    todo = [(t, s) for t, s in scenes if f"{t}|{s}" not in out]
    if todo:
        warnings.filterwarnings("ignore")
        from pai.discovery.bodies import BodyWorld
        for t, s in todo:
            pw = BodyWorld(t, BODIES[0], seed=s, **FIXTURE)
            out[f"{t}|{s}"] = pw.truth()["ids"]
            pw.env.close()
        path.write_text(json.dumps(out, indent=1))
    return out


def write_report(tag: str = "") -> None:
    lines = ["# The discovery agent with other bodies", "",
             "Generated by `scripts/discovery_bodies_eval.py report` from `results/discovery_bodies/*.json`. "
             "The same DiscoveryAgent (default parameters) in PhysicsWorld with decoys on and 2 keys; only the "
             "body changes. See the script's docstring for the cases and the failure causes.", ""]
    op = OUT / "oracle.json"
    if op.exists():
        o = json.loads(op.read_text())
        c = o["config"]
        lines += ["## Oracle (physical upper reference)", "",
                  f"Scripted oracle (reads the ground truth), same scenes: types {', '.join(c['types'])}; seeds "
                  f"{c['seed0']}..{c['seed0'] + c['seeds'] - 1}; decoys on. Opened / runs "
                  f"(no_solution: the oracle must fail). Total wall {c['wall_s']} s.", "",
                  "| body | " + " | ".join(c["cases"]) + " | solvable total |", "|---" * (len(c["cases"]) + 2) + "|"]
        by = defaultdict(list)
        for r in o["runs"]:
            by[(r["body"], r["case"])].append(r)
        for b in c["bodies"]:
            cells, tot = [], [0, 0]
            for cs in c["cases"]:
                x = by[(b, cs)]
                ok = sum(r["opened"] for r in x)
                cells.append(f"{ok}/{len(x)}")
                if cs != "no_solution":
                    tot[0] += ok
                    tot[1] += len(x)
            lines.append(f"| {b} | " + " | ".join(cells) + f" | {tot[0]}/{tot[1]} |")
        why = defaultdict(Counter)
        for r in o["runs"]:
            if not r["opened"] and r["solvable"]:
                why[r["body"]][r["why"] or r["stage"]] += 1
        if why:
            lines += ["", "Oracle failures on solvable doors:", ""]
            for b in c["bodies"]:
                if why[b]:
                    lines.append(f"- {b}: " + "; ".join(f"{k} ({n})" for k, n in why[b].most_common()))
        lines.append("")
    ap = OUT / (f"agent_{tag}.json" if tag else "agent.json")
    if ap.exists():
        a = json.loads(ap.read_text())
        c = a["config"]
        eps = a["episodes"]
        from pai.discovery.bodies import body_limited
        for r in eps:
            r["failure_raw"] = r["failure"]
            r["failure"] = body_limited(r["failure"], r["actions"], r["n_not_executed"])
        lines += ["## Discovery agent", "",
                  f"Types {', '.join(c['types'])}; seeds from {c['seed0']}, per body "
                  f"{c.get('seeds_per_body', c['seeds'])}; budget "
                  f"{c['budget']} actions or {c['limit_s']:.0f} s wall per episode; {c['processes']} processes, "
                  f"wall per run {c['wall_s_runs']} s.", "",
                  "Solvable doors opened per case (opened / episodes), and in total:", "",
                  "| body | " + " | ".join(x for x in c["cases"] if x != "no_solution") + " | total | "
                  "mean actions to success | mean sim s to success | explanation accuracy | no_solution: gave up |",
                  "|---" * (len(c["cases"]) + 5) + "|"]
        for b in c["bodies"]:
            e = [r for r in eps if r["body"] == b]
            if not e:
                continue
            sol = [r for r in e if r["solvable"]]
            cells = []
            for cs in c["cases"]:
                if cs == "no_solution":
                    continue
                x = [r for r in sol if r["case"] == cs]
                cells.append(f"{sum(r['success'] for r in x)}/{len(x)}")
            succ = [r for r in sol if r["success"]]
            ns = [r for r in e if not r["solvable"]]
            lines.append(f"| {b} | " + " | ".join(cells) + f" | {len(succ)}/{len(sol)} | {_mean([r['actions'] for r in succ])} "
                         f"| {_mean([r['sim'] for r in succ])} | {_mean([r['acc'] for r in e])} "
                         f"| {sum(r['stopped'] == 'no_solution' for r in ns)}/{len(ns)} |")
        lines += ["", "Why solvable doors stayed shut (pai.discovery.bodies.failure_cause: the first sub-goal of the "
                  "solution not met at the end, and what the agent's attempts at it did; a sub-goal never attempted "
                  "in an episode where >= 90% of the skills did not execute counts as not executed, "
                  "`body_limited`):", "",
                  "| body | failed | not executed | stalled | reasoning | unstable | timeout | blocked at |",
                  "|---|---|---|---|---|---|---|---|"]
        for b in c["bodies"]:
            f = [r for r in eps if r["body"] == b and r["solvable"] and not r["success"]]
            if not [r for r in eps if r["body"] == b]:
                continue
            cnt = Counter(r["failure"]["cause"] for r in f)
            at = Counter(f"{r['failure']['blocked_at']}: {r['failure']['cause']}" + (f" ({r['failure'].get('why')})" if r['failure'].get('why') else "")
                         for r in f)
            lines.append(f"| {b} | {len(f)} | {cnt['not_executed']} | {cnt['stalled']} | {cnt['reasoning']} | "
                         f"{cnt['unstable']} | {cnt['timeout']} | " + "; ".join(f"{k} x{n}" for k, n in at.most_common()) + " |")
        lines += ["", "Actions per body over all episodes: share of skills that did not execute, and that ran but "
                  "stalled (a stalled probe of a locked door is expected; these are all actions, not only failures):", "",
                  "| body | actions | not executed | stalled | aborted | most common skill failures |", "|---|---|---|---|---|---|"]
        for b in c["bodies"]:
            e = [r for r in eps if r["body"] == b]
            if not e:
                continue
            n = sum(r["actions"] for r in e)
            notes = Counter()
            for r in e:
                notes.update(r["notes_failed"])
            lines.append(f"| {b} | {n} | {sum(r['n_not_executed'] for r in e) / max(n, 1):.2f} | "
                         f"{sum(r['n_stalled'] for r in e) / max(n, 1):.2f} | {sum(r['n_aborted'] for r in e)} | "
                         + "; ".join(f"{k} ({v})" for k, v in notes.most_common(4)) + " |")
        traces = json.loads(gzip.decompress((OUT / (ap.stem + "_traces.json.gz")).read_bytes()))
        names = _part_names(sorted({(r["type"], r["seed"]) for r in eps}))
        lines += ["", "Where the skills did not execute: per body, the parts the agent acted on (ground-truth names, "
                  "keys pooled), not executed / actions, and the longest run of consecutive actions that all failed "
                  "as skills on the same part (median and max over episodes). The agent gets no evidence from a "
                  "skill that did not execute, so a part it cannot grasp stays as informative as before:", "",
                  "| body | not executed / actions per part | longest failed run: median, max |", "|---|---|---|"]
        for b in c["bodies"]:
            e = [r for r in eps if r["body"] == b]
            if not e:
                continue
            n, ne, runs = Counter(), Counter(), []
            for r in e:
                m = names[f"{r['type']}|{r['seed']}"]
                best = cur = 0
                prev = None
                for t in traces[_key(r)]:
                    p = m.get(t["a"][1], t["a"][1])
                    p = "key" if p.startswith("home_key") else p
                    n[p] += 1
                    ne[p] += not t["executed"]
                    cur = cur + 1 if not t["executed"] and p == prev else (0 if t["executed"] else 1)
                    prev = p if not t["executed"] else None
                    best = max(best, cur)
                runs.append(best)
            lines.append(f"| {b} | " + "; ".join(f"{p} {ne[p]}/{n[p]}" for p, _ in n.most_common())
                         + f" | {int(np.median(runs))}, {max(runs)} |")
        signs = Counter(tuple(s) for r in eps for s in r.get("unlock_signs", []))
        lines += ["", "Bolt retractions seen, by (action, sign, bolt): " + ", ".join(f"{k}: {v}" for k, v in sorted(signs.items(), key=str)), "",
                  "Per episode:", "", "| body | type | case | seed | opened | stopped | actions | sim s | wall s | acc | cause |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in eps:
            f = r["failure"] or {}
            cause = f"{f.get('cause')} @ {f.get('blocked_at')}" if f else ""
            acc = "-" if r["acc"] is None else f"{r['acc']:.2f}"
            lines.append(f"| {r['body']} | {r['type']} | {r['case']} | {r['seed']} | {int(r['success'])} | {r['stopped']} | "
                         f"{r['actions']} | {r['sim']:.0f} | {r['wall']:.0f} | {acc} | {cause} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("oracle", "agent"):
        p = sub.add_parser(name)
        p.add_argument("--bodies", nargs="*", default=list(BODIES))
        p.add_argument("--types", nargs="*", default=list(TYPES))
        p.add_argument("--cases", nargs="*", default=list(CASES))
        p.add_argument("--seeds", type=int, default=2)
        p.add_argument("--processes", type=int, default=12)
        if name == "agent":
            p.add_argument("--budget", type=int, default=300)
            p.add_argument("--limit", type=float, default=600.0, help="wall seconds per episode")
            p.add_argument("--tag", default="", help="write agent_<tag>.json instead of agent.json")
            p.add_argument("--seeds-for", nargs="*", default=[], metavar="BODY=N",
                           help="fewer (or more) seeds for some bodies, e.g. shadow=1")
    g = sub.add_parser("gif")
    g.add_argument("--body", required=True)
    g.add_argument("--type", default=TYPES[0])
    g.add_argument("--case", default="latch")
    g.add_argument("--seed", type=int, default=SEED0)
    g.add_argument("--budget", type=int, default=300)
    g.add_argument("--size", type=int, default=360)
    g.add_argument("--busy-frames", type=int, default=8)
    g.add_argument("--boring-frames", type=int, default=30)
    g.add_argument("--max-frames", type=int, default=120)
    r = sub.add_parser("report")
    r.add_argument("--tag", default="")
    args = ap.parse_args()
    {"oracle": run_oracle_table, "agent": run_agent, "gif": run_gif, "report": lambda a: write_report(a.tag)}[args.cmd](args)


if __name__ == "__main__":
    main()
