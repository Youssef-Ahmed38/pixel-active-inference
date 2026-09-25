"""Evaluation of the discovery agent: does it find out how to open a locked door, and does it say why?

    python scripts/discovery_eval.py mock                 # MockWorld: 200 scenes per case and split, 3 agents
    python scripts/discovery_eval.py mock --n 20          # a quick run
    python scripts/discovery_eval.py physics              # the same agent in PhysicsWorld (MuJoCo), a small set
    python scripts/discovery_eval.py gif                  # docs/media/discovery/: the agent finding the drawer key
    python scripts/discovery_eval.py report               # results/discovery_eval.md from the JSON files

A frozen-parameter re-run on fresh seeds, with the ablations (outputs in the --out folder, e.g. results/discovery_v2/):

    python scripts/discovery_eval.py mock --seed-offset 100000 --config default --out results/discovery_v2
    python scripts/discovery_eval.py mock --seed-offset 100000 --config both_off --agents discovery --out ...
    python scripts/discovery_eval.py physics --seed-offset 100000 --decoys --config default --out results/discovery_v2
    python scripts/discovery_eval.py report2 --out results/discovery_v2    # eval.md / eval.json there

--config picks the locality and the mock decoys (CONFIGS): default, loc_off (Locality.off(): no probe radius,
no distance priors, the evidence explanation filter), shared (MockWorld decoys shaped like the mechanisms),
both_off (both). Only the discovery agent's locality changes; the baselines keep their probe radius.
--seed-offset is added to every scene seed and every agent seed (the numbers in results/ used 0).

The agents get the same skills, the same budget (actions, not time) and never read the ground truth; only
the scoring here does (MockWorld.truth(), PhysicsWorld.truth()). Cases:

    unlocked    no latch catch, no bolt: any move in the right direction opens it
    latch       spring latch: the handle must be turned (pressed) while the door is moved
    deadbolt    latch + thumb-turn deadbolt thrown
    key_table   latch + key deadbolt; the fitting key on the cabinet top (1-3 keys, others are decoys)
    key_drawer  the same, the fitting key inside the closed drawer (not in view until it is opened)
    both        latch + thumb deadbolt + key deadbolt, the key on the table, the shelf or in the drawer
    no_solution key or both, the fitting key in the other room (behind the locked door)

Scenes come from pai.discovery.mock.sample_scenario (door types from the home-door registry, with its
train / held-out test split; distractors, including a dial and a hook on the door leaf, at random).

Metrics: success within the budget; actions and simulated seconds to success (mean and median over the
successes, and the mean with failures counted at the budget); explanation accuracy of the discovery
agent against the truth (pai.discovery.explain.score: all fields, and only the mechanisms present);
on no-solution scenes, how often it stops and says so (the baselines never stop); on solvable scenes,
how often it wrongly gives up. Second encounter: a recipe from one solved train scene, then a new scene
of the same case (a train scene, and a held-out test type), with and without the recipe, same agent seed.
Every mean comes with a 95% percentile bootstrap interval (2000 resamples; for reuse, of paired differences).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.discovery.agent import AGENTS, DiscoveryAgent, Locality  # noqa: E402
from pai.discovery.explain import explain, score  # noqa: E402
from pai.discovery.mock import MockWorld, sample_scenario  # noqa: E402
from pai.discovery.recipes import RecipeBook  # noqa: E402

RESULTS = ROOT / "results"
MEDIA = ROOT / "docs" / "media" / "discovery"
CASES = ("unlocked", "latch", "deadbolt", "key_table", "key_drawer", "both", "no_solution")
REUSE_LOCKS = ("latch", "deadbolt", "key", "both")
N_BOOT = 2000
# name -> (the discovery agent's locality, MockWorld shared-shape decoys); the baselines keep their radius
CONFIGS = {"default": (Locality(), False), "loc_off": (Locality.off(), False),
           "shared": (Locality(), True), "both_off": (Locality.off(), True)}


# ---------------------------------------------------------------------------------------- statistics
def boot(x, stat=np.mean, seed: int = 0) -> list:
    """95% percentile bootstrap interval of stat(x)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    s = stat(x[rng.integers(len(x), size=(N_BOOT, len(x)))], axis=1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def est(x, stat=np.mean) -> dict:
    x = np.asarray(x, float)
    return {"v": float(stat(x)) if len(x) else None, "ci": boot(x, stat), "n": int(len(x))}


# ---------------------------------------------------------------------------------------- scenes
def case_scene(rng: np.random.Generator, case: str, split: str, shared: bool = False):
    """A scene of the given case; key placements and solvability as the case says."""
    kw = {"shared_shapes": True} if shared else {}
    if case in ("unlocked", "latch", "deadbolt"):
        return sample_scenario(rng, split, lock_state=case, **kw)
    if case in ("key_table", "key_drawer"):
        return sample_scenario(rng, split, lock_state="key", key_place=case[4:], **kw)
    if case == "no_solution":
        return sample_scenario(rng, split, lock_state=str(rng.choice(["key", "both"])), key_place="other_room", **kw)
    while True:
        sc = sample_scenario(rng, split, lock_state=case, **kw)
        if sc.solvable:
            return sc


def scenes(case: str, split: str, n: int, seed: int, shared: bool = False) -> list:
    rng = np.random.default_rng([seed, CASES.index(case), split == "test"])
    return [case_scene(rng, case, split, shared) for _ in range(n)]


def make_agent(agent: str, seed: int, config: str = "default"):
    """The agent under a config: only the discovery agent's locality changes (the baselines keep theirs)."""
    if agent == "discovery":
        return DiscoveryAgent(seed=seed, locality=CONFIGS[config][0])
    return AGENTS[agent](seed=seed)


def code_hashes() -> dict:
    """sha1 of the agent and world code a run used (parallel work may change it between runs)."""
    import hashlib
    fs = sorted((ROOT / "pai" / "discovery").glob("*.py")) + [ROOT / "pai" / "envs" / "home_doors.py", Path(__file__)]
    return {f.resolve().relative_to(ROOT).as_posix(): hashlib.sha1(f.read_bytes()).hexdigest()[:12] for f in fs}


# ---------------------------------------------------------------------------------------- episodes
def summary_of(log, truth: dict | None, explain_it: bool) -> dict:
    r = {"success": bool(log.reached_goal), "actions": int(log.n_actions), "time": float(log.time_cost),
         "stopped": log.stopped_reason, "wall": float(log.wall_time)}
    if explain_it and truth is not None:
        e = explain(log)
        s = score(e.claims, truth, log)
        r.update(acc=s["accuracy"], rel=s["relevant"], false=s["false_claims"], text=e.text)
    return r


def _mock_job(job: tuple) -> tuple:
    key, agent, sc, budget, seed, config = job
    w = MockWorld(sc)
    log = make_agent(agent, seed, config).run(w, budget)
    return key, summary_of(log, w.truth(), agent == "discovery")


def _reuse_job(job: tuple) -> tuple:
    lock, i, seed, budget, off, config = job
    shared = CONFIGS[config][1]
    rng = np.random.default_rng([seed, 77, REUSE_LOCKS.index(lock), i])
    first = case_scene(rng, lock, "train", shared)
    book = RecipeBook()
    lg1 = make_agent("discovery", off + i, config).run(MockWorld(first), budget)
    r = book.learn(lg1)
    out = {"lock": lock, "i": i, "first_actions": lg1.n_actions, "first_success": lg1.reached_goal,
           "recipe": r is not None}
    if r is None:
        return lock, out
    for split in ("train", "test"):
        second = case_scene(rng, lock, split, shared)
        for name, rec in (("cold", None), ("warm", book)):
            lg = make_agent("discovery", off + i, config).run(MockWorld(second), budget, recipes=rec)
            out[f"{split}_{name}"] = {"success": lg.reached_goal, "actions": lg.n_actions if lg.reached_goal else budget,
                                      "time": lg.time_cost}
    return lock, out


def _is_oom(e: BaseException) -> bool:
    """Out of memory (numpy's MemoryError, MuJoCo's 'Could not allocate memory'): the machine's, not the agent's."""
    return isinstance(e, MemoryError) or "allocate memory" in str(e)


def _retrying(job: tuple):
    """(id, fn(job)), run again (the same seeds, so the same episode) after an out-of-memory error, up to 4 tries."""
    fn, jid, job = job
    for k in range(4):
        try:
            return jid, fn(job)
        except Exception as e:             # noqa: PERF203
            if not _is_oom(e) or k == 3:
                raise
            import gc
            gc.collect()
            time.sleep(60 * (k + 1))


def pool_map(fn, jobs: list, processes: int, label: str, partial: Path | None = None, ids: list | None = None,
             restore=lambda r: r):
    """Results of fn over jobs, in any order, with progress. With `partial` (a .jsonl file) and `ids` (one
    string per job), every result is appended there as it arrives and a re-run skips the jobs already in it
    (restore() turns a stored result back into fn's return value), so an interrupted run resumes."""
    t0 = time.time()
    out = []
    ids = list(range(len(jobs))) if ids is None else ids
    if partial is not None:
        partial.parent.mkdir(parents=True, exist_ok=True)
        done = {}
        if partial.exists():
            for line in partial.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:   # a line cut short by the interruption
                    continue
                done[d["id"]] = restore(d["r"])
        keep = [k for k, i in enumerate(ids) if i not in done]
        out = [done[i] for i in ids if i in done]
        if out:
            print(f"  {label}: {len(out)}/{len(jobs)} already in {partial.name}", flush=True)
        jobs, ids = [jobs[k] for k in keep], [ids[k] for k in keep]
        fh = partial.open("a", encoding="utf-8")
    wrapped = [(fn, i, j) for i, j in zip(ids, jobs)]
    if processes <= 1:
        it = map(_retrying, wrapped)
        pool = None
    else:
        from multiprocessing import get_context
        # the agent iterates over sets of strings, so an episode depends on the string hash seed (measured:
        # the same episode took 984, 993 or 1000 actions under PYTHONHASHSEED 0-3); fix it for the workers
        os.environ.setdefault("PYTHONHASHSEED", "0")
        pool = get_context("spawn").Pool(processes)
        it = pool.imap_unordered(_retrying, wrapped, chunksize=1 if partial is not None else 2)
    try:
        for k, (jid, r) in enumerate(it):
            out.append(r)
            if partial is not None:
                fh.write(json.dumps({"id": jid, "r": r}, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
                fh.flush()
            if (k + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {label}: {k + 1}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        if partial is not None:
            fh.close()
    return out


def _as_keyed(r):
    """A (key, result) pair back from JSON (lists -> tuples)."""
    k, v = r
    return (tuple(k) if isinstance(k, list) else k), v


def _partial(args, name: str) -> Path | None:
    """The checkpoint file of a run in an --out folder (none for the old results/ layout)."""
    d, _ = _out(args, name, name)
    return None if d == RESULTS else d / "partial" / name


def agg(rows: list[dict], budget: int, solvable: bool) -> dict:
    succ = [r for r in rows if r["success"]]
    a = {"n": len(rows), "success": est([r["success"] for r in rows]),
         "actions_capped": est([r["actions"] if r["success"] else budget for r in rows]),
         "actions_to_success": est([r["actions"] for r in succ]),
         "actions_to_success_median": est([r["actions"] for r in succ], np.median),
         "time_to_success": est([r["time"] for r in succ]),
         "time_to_success_median": est([r["time"] for r in succ], np.median),
         "gave_up": est([r["stopped"] == "no_solution" for r in rows]),
         "wall": float(np.mean([r["wall"] for r in rows]))}
    if not solvable:
        gave = [r for r in rows if r["stopped"] == "no_solution"]
        a["actions_to_give_up"] = est([r["actions"] for r in gave])
        a["actions_to_give_up_median"] = est([r["actions"] for r in gave], np.median)
        a["time_to_give_up"] = est([r["time"] for r in gave])
    if rows and "acc" in rows[0]:
        a["explanation"] = est([r["acc"] for r in rows])
        a["explanation_relevant"] = est([r["rel"] for r in rows])
        a["false_claims"] = int(sum(r["false"] for r in rows))
    return a


def run_mock(args) -> None:
    jobs = []
    off = args.seed_offset
    shared = CONFIGS[args.config][1]
    for split in ("train", "test"):
        for case in CASES:
            for i, sc in enumerate(scenes(case, split, args.n, args.seed + off, shared)):
                for agent in args.agents:
                    jobs.append(((split, case, agent, i), agent, sc, args.budget, off + i, args.config))
    # the slow episodes (discovery on key-locked doors) first, so the pool does not end on a straggler
    jobs.sort(key=lambda j: (j[1] != "discovery", j[0][1] not in ("key_table", "key_drawer", "both", "no_solution")))
    hashes = code_hashes()
    t0 = time.time()
    res = pool_map(_mock_job, jobs, args.processes, "episodes", _partial(args, f"mock_{args.config}.episodes.jsonl"),
                   ["/".join(map(str, j[0])) + f"/{j[4]}" for j in jobs], _as_keyed)
    t_eval = time.time() - t0
    by = defaultdict(list)
    for (split, case, agent, i), r in res:
        by[(split, case, agent)].append((i, r))
    table = {}
    examples = {}
    for (split, case, agent), rows in sorted(by.items()):
        rows = [r for _, r in sorted(rows, key=lambda x: x[0])]
        table.setdefault(split, {}).setdefault(case, {})[agent] = agg(rows, args.budget, case != "no_solution")
        if agent == "discovery":
            examples[f"{split}/{case}"] = rows[0].get("text", "")
    overall = {}
    for split in ("train", "test"):
        for agent in args.agents:
            rows = [r for (s, c, a), v in by.items() if s == split and a == agent and c != "no_solution" for _, r in v]
            overall.setdefault(split, {})[agent] = agg(rows, args.budget, True)

    t1 = time.time()
    rjobs = [(lock, i, args.seed + off, args.budget, off, args.config) for lock in REUSE_LOCKS for i in range(args.reuse)]
    rres = pool_map(_reuse_job, rjobs, args.processes, "reuse", _partial(args, f"mock_{args.config}.reuse.jsonl"),
                    [f"{j[0]}/{j[1]}/{j[2]}" for j in rjobs], _as_keyed) if args.reuse else []
    t_reuse = time.time() - t1
    reuse = {}
    for lock in REUSE_LOCKS:
        rows = [r for lk, r in rres if lk == lock]
        ok = [r for r in rows if r["recipe"]]
        d = {"pairs": len(rows), "recipes": len(ok)}
        for split in ("train", "test"):
            cold = [r[f"{split}_cold"]["actions"] for r in ok]
            warm = [r[f"{split}_warm"]["actions"] for r in ok]
            d[split] = {"cold": est(cold), "warm": est(warm),
                        "cold_median": est(cold, np.median), "warm_median": est(warm, np.median),
                        "diff": est(np.subtract(warm, cold)),
                        "cold_success": est([r[f"{split}_cold"]["success"] for r in ok]),
                        "warm_success": est([r[f"{split}_warm"]["success"] for r in ok]),
                        "warm_fewer": est(np.less(warm, cold)), "warm_more": est(np.greater(warm, cold))}
        reuse[lock] = d
    out = {"config": {"n": args.n, "budget": args.budget, "seed": args.seed, "seed_offset": off,
                      "config": args.config, "locality": dict(vars(CONFIGS[args.config][0])),
                      "shared_shapes": shared, "agents": list(args.agents),
                      "reuse_pairs": args.reuse, "bootstrap": N_BOOT, "processes": args.processes,
                      "wall_eval_s": round(t_eval), "wall_reuse_s": round(t_reuse), "code": hashes},
           "cases": table, "overall": overall, "reuse": reuse, "examples": examples}
    out_dir, name = _out(args, f"mock_{args.config}.json", "discovery_eval.json")
    (out_dir / name).write_text(json.dumps(out, indent=1))
    print(f"wrote {out_dir / name} ({t_eval:.0f} s episodes, {t_reuse:.0f} s reuse)")
    if out_dir == RESULTS:
        write_report()


def _out(args, new_name: str, old_name: str) -> tuple:
    """(folder, file name): results/ keeps the old file names, another --out folder gets per-config names."""
    d = Path(args.out).resolve() if args.out else RESULTS
    d.mkdir(parents=True, exist_ok=True)
    return d, (old_name if d == RESULTS else new_name)


# ---------------------------------------------------------------------------------------- physics
PHYS_STATE = {"latch": "unlocked"}   # the mock's "latch" is the physics' "unlocked" (latch only)


def physics_truth(w) -> dict:
    """PhysicsWorld.truth() in the shape explain.score() reads (the mock's truth())."""
    tr = w.truth()
    ids = tr["ids"]                        # id -> name
    pid = {n: i for i, n in ids.items() if i.startswith("p")}
    oid = {n: i for i, n in ids.items() if i.startswith("o")}
    keys = {oid[n]: {"matches": k["matches"], "place": k["slot"][0], "fits": True} for n, k in tr["keys"].items() if n in oid}
    match = next((o for o, k in keys.items() if k["matches"]), None) if tr["key_locked"] else None
    return {"parts": pid, "solvable": tr["solvable"], "latched": True, "deadbolt": tr["deadbolt_locked"],
            "key_locked": tr["key_locked"], "matching_key": match, "key_place": tr["key_place"] if match else None,
            "keys": keys, "events": tr["events"]}


class _Timeout(Exception):
    pass


class TimedWorld:
    """A world wrapper: stops the episode (raises _Timeout) once `limit` wall seconds have passed, and
    keeps a count of what ran (so a timed-out episode still has numbers). Optionally records frames."""

    def __init__(self, world, limit: float, recorder=None):
        self.w, self.limit, self.rec = world, limit, recorder
        self.t0 = time.time()
        self.n, self.sim = 0, 0.0

    def __getattr__(self, name):
        return getattr(self.w, name)

    def execute(self, action):
        if time.time() - self.t0 > self.limit:
            raise _Timeout()
        if self.rec is not None:
            self.rec.begin(action)
        out = self.w.execute(action)
        if self.rec is not None:
            self.rec.end(out)
        self.n += 1
        self.sim += out.cost
        return out


def _physics_job(job: tuple) -> dict:
    warnings.filterwarnings("ignore")
    from pai.discovery.physics import PhysicsWorld
    type_name, case, seed, budget, limit, split, decoys, config = job
    st = PHYS_STATE.get(case, case)
    if case.startswith("key_"):
        st, place = "key", case[4:]
    elif case == "no_solution":
        st, place = "key", "other_room"
    else:
        place = {"both": "table"}.get(case, "table")
    pw = PhysicsWorld(type_name, seed=seed, lock_state=st, key_place=place, n_keys=2,
                      **({"decoys": True} if decoys else {}))
    w = TimedWorld(pw, limit)
    truth = physics_truth(pw)
    t0 = time.time()
    r = {"type": type_name, "split": split, "case": case, "seed": seed, "lock_state": st, "key_place": place,
         "decoys": bool(decoys), "config": config}
    try:
        log = make_agent("discovery", seed, config).run(w, budget)
        r.update(summary_of(log, truth, True))
        r["unstable"] = bool(pw.unstable)
        r["actions_list"] = [list(a) for a in log.actions]
    except _Timeout:
        r.update(success=False, actions=w.n, time=w.sim, stopped="timeout", acc=None, rel=None, false=0, text="")
    except Exception as e:                 # one crashed episode is recorded, not the end of the run
        if _is_oom(e):
            pw.env.close()
            raise
        import traceback
        r.update(success=False, actions=w.n, time=w.sim, stopped="error", acc=None, rel=None, false=0, text="",
                 error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    r["wall"] = time.time() - t0
    r["opened_ever"] = bool(pw.truth()["opened"])
    pw.env.close()
    print(f"  {type_name:22s} {case:11s} seed {seed}: success={int(r['success'])} actions={r['actions']} "
          f"sim={r['time']:.0f}s wall={r['wall']:.0f}s stopped={r['stopped']}", flush=True)
    return r


def run_physics(args) -> None:
    jobs = []
    off, extra = args.seed_offset, (args.decoys, args.config)
    for t in args.types:
        for case in args.cases:
            for s in range(args.seeds):
                jobs.append((t, case, off + s, args.budget, args.limit, "train", *extra))
    for t in args.test_types:
        for case in args.cases:
            jobs.append((t, case, off, args.budget, args.limit, "test", *extra))
    jobs = [j for j in jobs if _physics_ok(j[0], j[1])]
    # the long episodes (key locks, no solution) first, so the pool does not end on a straggler
    jobs.sort(key=lambda j: j[1] in ("latch", "deadbolt"))
    hashes = code_hashes()
    t0 = time.time()
    name = f"physics_{args.config}{'_decoys' if args.decoys else ''}"
    rows = pool_map(_physics_job, jobs, args.processes, "physics", _partial(args, f"{name}.jsonl"),
                    [f"{j[0]}/{j[1]}/{j[2]}/{j[3]}" for j in jobs])
    out = {"config": {"types": args.types, "test_types": args.test_types, "cases": args.cases, "seeds": args.seeds,
                      "budget": args.budget, "limit_s": args.limit, "wall_s": round(time.time() - t0),
                      "embodiment": "robotiq_2f85", "n_keys": 2, "seed_offset": off, "decoys": args.decoys,
                      "config": args.config, "locality": dict(vars(CONFIGS[args.config][0])),
                      "processes": args.processes, "code": hashes},
           "episodes": sorted(rows, key=lambda r: (r["split"], r["type"], CASES.index(r["case"]), r["seed"]))}
    out_dir, name = _out(args, f"{name}.json", "discovery_physics.json")
    (out_dir / name).write_text(json.dumps(out, indent=1))
    print(f"wrote {out_dir / name} ({out['config']['wall_s']} s)")
    if out_dir == RESULTS:
        write_report()


def _physics_ok(type_name: str, case: str) -> bool:
    from pai.envs.home_doors import HOME_DOOR_TYPES
    f = HOME_DOOR_TYPES[type_name].fixed
    need = {"deadbolt": f["has_thumb"], "both": f["has_thumb"] and f["has_key"]}
    return case != "unlocked" and need.get(case, f["has_key"] if case.startswith("key") or case == "no_solution" else True)


# ---------------------------------------------------------------------------------------- the GIF
class Recorder:
    """Frames of a PhysicsWorld episode: every `every` control steps while an action runs; after the
    action only some are kept (more of the actions that changed something)."""

    def __init__(self, pw, size: int = 320, every: int = 8, busy_frames: int = 8):
        self.pw, self.size, self.every, self.busy_frames = pw, size, every, busy_frames
        self.cur: list = []
        self.actions: list = []        # [(action, outcome, frames, belief top)]
        self.agent = None
        env = pw.env
        step = env.step
        self.k = 0

        def rec_step(a):
            obs = step(a)
            self.k += 1
            if self.k % self.every == 0:
                self.cur.append(env.render("overview", self.size))
            return obs
        env.step = rec_step

    def begin(self, action) -> None:
        self.cur = [self.pw.env.render("overview", self.size)]
        bel = self.agent.belief if self.agent is not None else None
        self.top = bel.top(1) if bel is not None else []
        self.p_unknown = float(bel.weights()[1]) if bel is not None else 1.0

    def end(self, out) -> None:
        self.cur.append(self.pw.env.render("overview", self.size))
        busy = bool({k for k in out.changed if not k.startswith("body:")}) or bool(out.revealed)
        keep = self.busy_frames if busy else 2
        fr = [self.cur[i] for i in np.unique(np.linspace(0, len(self.cur) - 1, min(keep, len(self.cur))).astype(int))]
        self.actions.append((out.action, out, fr, (self.top, self.p_unknown), busy))
        self.cur = []


def _describe(a, ents) -> str:
    look = lambda e: f"{e} ({ents[e].attrs.get('shape', '?')})" if e in ents else e   # noqa: E731
    if a[0] == "insert":
        return f"insert {look(a[1])} into {look(a[2])}"
    d = {1: " +", -1: " -"}.get(a[2], "") if len(a) > 2 else ""
    return f"{a[0].replace('_', ' ')}{d} {look(a[1])}"


def _rule(h, ents) -> str:
    if not h:
        return "-"
    lits = ", ".join(f"{f.split(':', 1)[1]}={v}" for f, v in h["literals"].items()) or "nothing"
    return f"p={h['p']:.2f}: {_describe(h['probe'], ents)} works if {lits}"


def _panel(img, lines: list, width: int, height: int = 118) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", (width, img.shape[0] + height), (18, 20, 26))
    canvas.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=12)
    y = img.shape[0] + 5
    for text, colour in lines:
        for ln in _wrap(text, font, width - 12, d):
            if y > img.shape[0] + height - 14:
                break
            d.text((6, y), ln, fill=colour, font=font)
            y += 14
    return np.asarray(canvas)


def _wrap(text: str, font, width: int, d) -> list:
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = t
    return out + ([cur] if cur else [])


def run_gif(args) -> None:
    """Record the agent in PhysicsWorld (by default on a door locked with a key hidden in the drawer) and
    write a GIF with what it did, what changed and its most probable rule overlaid, then its explanation."""
    warnings.filterwarnings("ignore")
    from PIL import Image
    from pai.discovery.physics import PhysicsWorld
    FG, DIM, OK, WARN = (235, 235, 235), (150, 155, 165), (110, 200, 120), (235, 150, 80)
    pw = PhysicsWorld(args.type, seed=args.seed, lock_state=args.state, key_place=args.place, n_keys=2)
    rec = Recorder(pw, size=args.size, busy_frames=args.busy_frames)
    agent = DiscoveryAgent(seed=args.seed)
    rec.agent = agent
    w = TimedWorld(pw, args.limit, rec)
    t0 = time.time()
    try:
        log = agent.run(w, args.budget)
    except _Timeout:
        log = None
    print(f"episode: success={log is not None and log.reached_goal} actions={len(rec.actions)} wall={time.time() - t0:.0f}s")
    ents = dict(log.entities) if log is not None else {e.id: e for e in pw.entities()}
    frames = []
    n_all = len(rec.actions)
    # boring actions (nothing changed): one frame each, and only every few of them, to fit the size budget
    boring = [i for i, x in enumerate(rec.actions) if not x[4]]
    stride = max(1, int(np.ceil(len(boring) / args.boring_frames)))
    keep_boring = set(boring[::stride])
    for i, (a, out, fr, top, busy) in enumerate(rec.actions):
        if not busy and i not in keep_boring:
            continue
        mode = log.modes[i] if log is not None and i < len(log.modes) else ""
        res = ("stalled" if out.stalled else "done") if out.executed else "skill failed"
        ch = ", ".join(f"{k.split(':', 1)[1]}: {v[1]}" for k, v in out.changed.items() if not k.startswith("body:"))
        if out.revealed:
            ch = (ch + "; " if ch else "") + "new in view: " + ", ".join(e.id for e in out.revealed)
        lines = [(f"action {i + 1}/{n_all} ({mode}): {_describe(a, ents)} -> {res}", FG),
                 (f"changed: {ch}" if ch else "changed: nothing", OK if ch else DIM),
                 (f"best rule {_rule(top[0][0] if top[0] else None, ents)}", WARN),
                 (f"P(cause is something I have not thought of) = {top[1]:.2f}", DIM)]
        for f in (fr if busy else fr[-1:]):
            frames.append(_panel(f, lines, args.size))
    if len(frames) > args.max_frames:     # evenly thinned, to fit the size budget
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, args.max_frames).astype(int)]
    if log is not None:
        text = explain(log).text
        last = rec.actions[-1][2][-1]
        end = _panel(last, [("explanation: " + text, FG)], args.size, height=118)
        frames += [end] * 25
    MEDIA.mkdir(parents=True, exist_ok=True)
    path = MEDIA / args.out
    ims = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=160, loop=0, optimize=True)
    Image.fromarray(frames[len(frames) // 2]).save(path.with_suffix(".png"))
    meta = {"type": args.type, "seed": args.seed, "lock_state": args.state, "key_place": args.place, "success": bool(log is not None and log.reached_goal),
            "actions": n_all, "sim_s": float(sum(x[1].cost for x in rec.actions)), "frames": len(frames),
            "bytes": path.stat().st_size, "explanation": explain(log).text if log is not None else None,
            "actions_list": [[*x[0]] for x in rec.actions]}
    (MEDIA / (path.stem + ".json")).write_text(json.dumps(meta, indent=1))
    print(f"wrote {path} ({len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------------------- report
def _f(e, digits: int = 1, pct: bool = False) -> str:
    if e is None or e.get("v") is None:
        return "-"
    k = 100 if pct else 1
    lo, hi = e["ci"]
    if pct:
        return f"{k * e['v']:.0f}% [{k * lo:.0f}, {k * hi:.0f}]"
    return f"{e['v']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_report() -> None:
    lines = ["# Discovering how to open a locked door: evaluation", "",
             "Generated by `scripts/discovery_eval.py` (see its docstring for the cases and metrics). "
             "Brackets are 95% bootstrap intervals. Actions and simulated seconds *to success* are over the "
             "successful episodes only; *capped* counts a failure at the budget.", ""]
    mj = RESULTS / "discovery_eval.json"
    if mj.exists():
        m = json.loads(mj.read_text())
        cfg = m["config"]
        B = cfg["budget"]
        lines += [f"## MockWorld ({cfg['n']} scenes per case and split, budget {B} actions)", "",
                  f"Wall time: {cfg['wall_eval_s']} s for the episodes, {cfg['wall_reuse_s']} s for reuse, "
                  f"{cfg['processes']} processes.", ""]
        for split in ("train", "test"):
            lines += [f"### {'Train door types' if split == 'train' else 'Held-out test door types'}", "",
                      "| case | agent | success | actions to success (mean) | median | actions capped (mean) "
                      "| sim s to success (mean) | median |",
                      "|---|---|---|---|---|---|---|---|"]
            for case in CASES:
                if case == "no_solution":
                    continue
                for agent, a in m["cases"][split][case].items():
                    lines.append(f"| {case} | {agent} | {_f(a['success'], pct=True)} | {_f(a['actions_to_success'])} "
                                 f"| {_f(a['actions_to_success_median'])} | {_f(a['actions_capped'])} "
                                 f"| {_f(a['time_to_success'])} | {_f(a['time_to_success_median'])} |")
            for agent, a in m["overall"][split].items():
                lines.append(f"| **all solvable** | {agent} | {_f(a['success'], pct=True)} | {_f(a['actions_to_success'])} "
                             f"| {_f(a['actions_to_success_median'])} | {_f(a['actions_capped'])} "
                             f"| {_f(a['time_to_success'])} | {_f(a['time_to_success_median'])} |")
            lines.append("")
        lines += ["### Explanations and giving up (discovery agent)", "",
                  "| split | case | explanation accuracy | on the mechanisms present | false claims | wrongly gave up "
                  "| correctly gave up | actions to give up (mean) | median |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for split in ("train", "test"):
            for case in CASES:
                a = m["cases"][split][case]["discovery"]
                sol = case != "no_solution"
                lines.append(f"| {split} | {case} | {_f(a['explanation'], 3)} | {_f(a['explanation_relevant'], 3)} "
                             f"| {a['false_claims']} | {_f(a['gave_up'], pct=True) if sol else '-'} "
                             f"| {'-' if sol else _f(a['gave_up'], pct=True)} "
                             f"| {'-' if sol else _f(a['actions_to_give_up'])} "
                             f"| {'-' if sol else _f(a['actions_to_give_up_median'])} |")
        lines.append("")
        lines += ["No-solution scenes for the baselines: they never stop before the budget "
                  "(correct give-up 0%; they open none).", ""]
        for split in ("train", "test"):
            ns = m["cases"][split]["no_solution"]
            lines.append(f"- {split}: " + "; ".join(f"{ag} opened {_f(a['success'], pct=True)}, gave up "
                                                     f"{_f(a['gave_up'], pct=True)}" for ag, a in ns.items()))
        lines.append("")
        lines += ["### Second encounter: reuse of a recipe", "",
                  f"A recipe from one solved train scene (discovery agent), then a new scene of the same lock state: "
                  f"another train scene, and a held-out test type. Mean actions (failures at {B}), same agent seed; "
                  f"'diff' is the paired mean of with minus without.", "",
                  "| lock | recipes / pairs | new scene | without recipe | with recipe | diff | median without "
                  "| median with | fewer / more actions with it |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for lock, d in m["reuse"].items():
            for split in ("train", "test"):
                x = d[split]
                lines.append(f"| {lock} | {d['recipes']} / {d['pairs']} | {split} | {_f(x['cold'])} | {_f(x['warm'])} "
                             f"| {_f(x['diff'])} | {_f(x['cold_median'])} | {_f(x['warm_median'])} "
                             f"| {100 * x['warm_fewer']['v']:.0f}% / {100 * x['warm_more']['v']:.0f}% |")
        lines.append("")
        lines += ["### Example explanations (first scene of each case, discovery agent)", ""]
        for k, t in m["examples"].items():
            lines.append(f"- **{k}**: {t}")
        lines.append("")
    pj = RESULTS / "discovery_physics.json"
    if pj.exists():
        p = json.loads(pj.read_text())
        cfg = p["config"]
        eps = p["episodes"]
        lines += [f"## PhysicsWorld (MuJoCo, {cfg['embodiment']}, {cfg['n_keys']} keys)", "",
                  f"The same agent, unchanged, on the physics door. Budget {cfg['budget']} actions or "
                  f"{cfg['limit_s']} s wall per episode; total wall {cfg['wall_s']} s. The physics 'unlocked' "
                  "state has the spring latch (the mock's 'latch'); 'both' has the key on the table.", "",
                  "| split | type | case | seed | opened | stopped | actions | sim s | wall s | explanation accuracy "
                  "| on mechanisms present |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in eps:
            acc = "-" if r.get("acc") is None else f"{r['acc']:.2f}"
            rel = "-" if r.get("rel") is None else f"{r['rel']:.2f}"
            lines.append(f"| {r['split']} | {r['type']} | {r['case']} | {r['seed']} | {int(r['success'])} | {r['stopped']} "
                         f"| {r['actions']} | {r['time']:.0f} | {r['wall']:.0f} | {acc} | {rel} |")
        sol = [r for r in eps if r["case"] != "no_solution"]
        uns = [r for r in eps if r["case"] == "no_solution"]
        succ = [r for r in sol if r["success"]]
        lines += ["", f"Solvable: opened {len(succ)}/{len(sol)}"
                  + (f", mean {np.mean([r['actions'] for r in succ]):.1f} actions and "
                     f"{np.mean([r['time'] for r in succ]):.0f} s simulated to success" if succ else "")
                  + (f"; mean explanation accuracy {np.mean([r['acc'] for r in sol if r.get('acc') is not None]):.3f}"
                     if any(r.get("acc") is not None for r in sol) else "") + "."]
        if uns:
            lines.append(f"No solution: gave up {sum(r['stopped'] == 'no_solution' for r in uns)}/{len(uns)}.")
        lines.append("")
        ex = [r for r in eps if r.get("text")]
        if ex:
            lines += ["Explanations:", ""] + [f"- {r['type']} {r['case']} seed {r['seed']}: {r['text']}" for r in ex]
            lines.append("")
    (RESULTS / "discovery_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS / 'discovery_eval.md'}")


# ---------------------------------------------------------------------------------------- report of a re-run
def physics_summary(eps: list) -> dict:
    """Per case and in total: opened / episodes, mean actions and sim s to success, explanation accuracy."""
    def one(rows):
        succ = [r for r in rows if r["success"]]
        acc = [r["acc"] for r in succ if r.get("acc") is not None]
        return {"n": len(rows), "opened": len(succ),
                "actions_to_success": float(np.mean([r["actions"] for r in succ])) if succ else None,
                "time_to_success": float(np.mean([r["time"] for r in succ])) if succ else None,
                "explanation_on_success": float(np.mean(acc)) if acc else None,
                "gave_up": sum(r["stopped"] == "no_solution" for r in rows),
                "timeouts": sum(r["stopped"] == "timeout" for r in rows),
                "errors": sum(r["stopped"] == "error" for r in rows),
                "unstable": sum(bool(r.get("unstable")) for r in rows),
                "wall_mean_s": float(np.mean([r["wall"] for r in rows])) if rows else None}
    cases = sorted({r["case"] for r in eps}, key=CASES.index)
    out = {c: one([r for r in eps if r["case"] == c]) for c in cases}
    out["all_solvable"] = one([r for r in eps if r["case"] != "no_solution"])
    return out


def _pc(e: dict) -> str:
    return "-" if e.get("v") is None else f"{100 * e['v']:.0f}%"


def _cell(a: dict, B: int) -> str:
    """success rate (when below 100%) and mean actions with failures at the budget."""
    s = a["success"]["v"]
    return f"{_f(a['actions_capped'])}" + ("" if s == 1 else f" ({100 * s:.1f}%)")


def write_report2(out_dir: Path) -> None:
    """eval.md / eval.json in out_dir from its mock_<config>.json and physics_*.json, against the old results/."""
    mocks = {c: json.loads((out_dir / f"mock_{c}.json").read_text()) for c in CONFIGS if (out_dir / f"mock_{c}.json").exists()}
    phys = {f.stem[len("physics_"):]: json.loads(f.read_text()) for f in sorted(out_dir.glob("physics_*.json"))}
    old_m = json.loads((RESULTS / "discovery_eval.json").read_text()) if (RESULTS / "discovery_eval.json").exists() else None
    old_p = json.loads((RESULTS / "discovery_physics.json").read_text()) if (RESULTS / "discovery_physics.json").exists() else None
    L = ["# Discovery agent: clean re-run on fresh seeds, with the locality and decoy ablations", "",
         "Generated by `scripts/discovery_eval.py report2` from the JSON files in this folder. Frozen parameters: "
         "the agents are the reported ones, unchanged; only the seeds (every scene seed and every agent seed "
         "offset, see each file's config) and the switches named below differ. Brackets are 95% percentile "
         "bootstrap intervals (2000 resamples). Unless said otherwise a cell is the mean number of actions until "
         "the door opens with a failure counted at the budget, and in parentheses the success rate when below 100%.", "",
         "Configs: **default** (the reported agent); **loc_off** (discovery agent with `Locality.off()`: probes on "
         "any part in view, no distance penalty on probes or literals, evidence-based explanation filter); "
         "**shared** (MockWorld decoys shaped like the mechanisms: a non-graspable disc like the lock cylinder, a "
         "turning wing on the leaf like the thumb-turn, the door dial shaped like the handle); **both_off** (both). "
         "The baselines keep their 0.7 m probe radius in every config.", ""]
    notes = out_dir / "notes.md"          # a hand-written summary of the tables, kept across regenerations
    if notes.exists():
        L += [notes.read_text(encoding="utf-8").rstrip(), ""]
    L += ["## Runs", ""]
    for c, m in mocks.items():
        cfg = m["config"]
        L.append(f"- mock `{c}`: {cfg['n']} scenes per case and split, seed {cfg['seed']} + offset {cfg['seed_offset']}, "
                 f"agents {', '.join(cfg['agents'])}, {cfg['reuse_pairs']} reuse pairs per lock, budget {cfg['budget']}, "
                 f"wall {cfg['wall_eval_s']} s + {cfg['wall_reuse_s']} s reuse on {cfg.get('processes', '?')} processes")
    for c, p in phys.items():
        cfg = p["config"]
        L.append(f"- physics `{c}`: {len(p['episodes'])} episodes, seed offset {cfg.get('seed_offset', 0)}, decoys {cfg.get('decoys', False)}, "
                 f"budget {cfg['budget']} actions / {cfg['limit_s']:.0f} s wall per episode, wall {cfg['wall_s']} s "
                 f"on {cfg.get('processes', '?')} processes")
    L.append("")
    hs = {k: {f: h for f, h in (v["config"].get("code") or {}).items() if not f.startswith("scripts/")}
          for k, v in list(mocks.items()) + list(phys.items())}
    if len({json.dumps(h, sort_keys=True) for h in hs.values()}) > 1:
        L += ["**Note:** the code hashes differ between runs (other work changed files in between); see `code` in "
              "each config.", ""]
        cc = out_dir / "code_check.json"
        if cc.exists():  # replay of earlier runs' episodes under the newer code, see code_check.json
            c = json.loads(cc.read_text())
            sh, bo = c.get("shared", {}), c.get("both_off", [])
            L += [f"Code check (`code_check.json`): under the newer code, the `shared` discovery means on the train split "
                  f"match exactly for {', '.join(k for k, v in sh.items() if v['old_actions'] == v['new_actions'])} "
                  f"({len(sh)} cases x 200 scenes re-run); of {len(bo)} `both_off` episodes replayed one by one, "
                  f"{sum(r['same'] for r in bo)} are identical (actions, success, stop reason, explanation)"
                  + "".join(f"; {r['split']} {r['case']} #{r['i']}: {r['old']} -> {r['new']} actions"
                            for r in bo if not r["same"]) + ".", ""]
    B = next(iter(mocks.values()))["config"]["budget"] if mocks else 1000
    if "default" in mocks:
        m = mocks["default"]
        for split in ("train", "test"):
            ags = list(m["cases"][split]["unlocked"])
            L += [f"## MockWorld, default: {'train' if split == 'train' else 'held-out test'} door types", "",
                  "| case | " + " | ".join(ags) + " |", "|---|" + "---|" * len(ags)]
            for case in CASES[:-1]:
                L.append(f"| {case} | " + " | ".join(_cell(m["cases"][split][case][a], B) for a in ags) + " |")
            L.append("| **all solvable** | " + " | ".join(_cell(m["overall"][split][a], B) for a in ags) + " |")
            L.append("")
    if mocks:
        L += ["## MockWorld, discovery agent under each config", ""]
        for split in ("train", "test"):
            cs = list(mocks)
            L += [f"### {'Train' if split == 'train' else 'Held-out test'} door types", "",
                  "| case | " + " | ".join(cs) + " |", "|---|" + "---|" * len(cs)]
            for case in CASES[:-1]:
                L.append(f"| {case} | " + " | ".join(_cell(mocks[c]["cases"][split][case]["discovery"], B) for c in cs) + " |")
            L.append("| **all solvable** | " + " | ".join(_cell(mocks[c]["overall"][split]["discovery"], B) for c in cs) + " |")
            L += ["", "Explanation accuracy (all fields) / false claims (total), and giving up:", "",
                  "| case | " + " | ".join(cs) + " |", "|---|" + "---|" * len(cs)]
            for case in CASES:
                L.append(f"| {case} | " + " | ".join(
                    f"{_f(mocks[c]['cases'][split][case]['discovery']['explanation'], 3)} / "
                    f"{mocks[c]['cases'][split][case]['discovery']['false_claims']}" for c in cs) + " |")
            L.append("| wrongly gave up (solvable) | " + " | ".join(
                str(sum(round(mocks[c]["cases"][split][case]["discovery"]["gave_up"]["v"] * mocks[c]["cases"][split][case]["discovery"]["n"])
                        for case in CASES[:-1])) + f" / {sum(mocks[c]['cases'][split][case]['discovery']['n'] for case in CASES[:-1])}"
                for c in cs) + " |")
            L.append("| no solution: gave up | " + " | ".join(
                _f(mocks[c]["cases"][split]["no_solution"]["discovery"]["gave_up"], pct=True) for c in cs) + " |")
            L.append("| no solution: actions to give up | " + " | ".join(
                _f(mocks[c]["cases"][split]["no_solution"]["discovery"]["actions_to_give_up"]) for c in cs) + " |")
            L.append("")
    if "shared" in mocks and len(mocks["shared"]["config"]["agents"]) > 1:
        m = mocks["shared"]
        L += ["## MockWorld, shared shapes: baselines", ""]
        for split in ("train", "test"):
            ags = list(m["cases"][split]["unlocked"])
            L += [f"### {split}", "", "| case | " + " | ".join(ags) + " |", "|---|" + "---|" * len(ags)]
            for case in CASES[:-1]:
                L.append(f"| {case} | " + " | ".join(_cell(m["cases"][split][case][a], B) for a in ags) + " |")
            L.append("| **all solvable** | " + " | ".join(_cell(m["overall"][split][a], B) for a in ags) + " |")
            L.append("")
    reuse = {c: m["reuse"] for c, m in mocks.items() if m["config"]["reuse_pairs"]}
    if reuse:
        L += ["## Second encounter: recipe reuse", "",
              f"A recipe from one solved train scene, then a new scene of the same lock state (train, and a held-out "
              f"type), without and with the recipe, same agent seed; failures at {B}. 'diff' is the paired mean.", "",
              "| config | lock | recipes / pairs | new scene | without | with | diff | median without | median with "
              "| fewer / more with it |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c, rr in reuse.items():
            for lock, d in rr.items():
                for split in ("train", "test"):
                    x = d[split]
                    L.append(f"| {c} | {lock} | {d['recipes']} / {d['pairs']} | {split} | {_f(x['cold'])} | {_f(x['warm'])} "
                             f"| {_f(x['diff'])} | {_f(x['cold_median'])} | {_f(x['warm_median'])} "
                             f"| {_pc(x['warm_fewer'])} / {_pc(x['warm_more'])} |")
        L.append("")
    psum = {c: physics_summary(p["episodes"]) for c, p in phys.items()}
    if phys:
        L += ["## PhysicsWorld (MuJoCo, Robotiq 2F-85, 2 keys, physical decoys where marked)", "",
              "Opened / episodes; mean actions and simulated seconds to success over the successes; explanation "
              "accuracy over the successes; no solution: gave up / episodes.", "",
              "| config | case | opened | actions to success | sim s to success | explanation | gave up | timeouts | errors |",
              "|---|---|---|---|---|---|---|---|---|"]
        for c, ps in psum.items():
            for case, a in ps.items():
                num = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"   # noqa: E731
                L.append(f"| {c} | {case} | {a['opened']}/{a['n']} | {num(a['actions_to_success'])} | "
                         f"{num(a['time_to_success'], 0)} | {num(a['explanation_on_success'], 2)} | "
                         f"{str(a['gave_up']) + '/' + str(a['n']) if case == 'no_solution' else '-'} | {a['timeouts']} | {a['errors']} |")
        L += ["", "Episodes:", "", "| config | split | type | case | seed | opened | stopped | actions | sim s | wall s "
              "| explanation | on mechanisms present |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, p in phys.items():
            for r in p["episodes"]:
                acc = "-" if r.get("acc") is None else f"{r['acc']:.2f}"
                rel = "-" if r.get("rel") is None else f"{r['rel']:.2f}"
                L.append(f"| {c} | {r['split']} | {r['type']} | {r['case']} | {r['seed']} | {int(r['success'])} "
                         f"| {r['stopped']} | {r['actions']} | {r['time']:.0f} | {r['wall']:.0f} | {acc} | {rel} |")
        L.append("")
    comp = {}
    if old_m is not None and "default" in mocks:
        m = mocks["default"]
        L += ["## Against the old numbers (results/discovery_eval.json, seeds 0..199; results/discovery_physics.json)", "",
              "MockWorld, default config: old -> new (mean actions, failures at the budget; success rate).", "",
              "| split | case | agent | old | new |", "|---|---|---|---|---|"]
        for split in ("train", "test"):
            for case in list(CASES[:-1]) + ["all solvable"]:
                for a in m["cases"][split]["unlocked"]:
                    o = old_m["overall"][split][a] if case == "all solvable" else old_m["cases"][split][case][a]
                    n = m["overall"][split][a] if case == "all solvable" else m["cases"][split][case][a]
                    comp[f"{split}/{case}/{a}"] = {"old": o["actions_capped"], "new": n["actions_capped"],
                                                   "old_success": o["success"]["v"], "new_success": n["success"]["v"]}
                    L.append(f"| {split} | {case} | {a} | {_cell(o, B)} | {_cell(n, B)} |")
        L.append("")
        L += ["Explanation accuracy and giving up, discovery agent: old -> new.", "",
              "| split | case | explanation old | new | false claims old | new | gave up old | new |",
              "|---|---|---|---|---|---|---|---|"]
        for split in ("train", "test"):
            for case in CASES:
                o, n = old_m["cases"][split][case]["discovery"], m["cases"][split][case]["discovery"]
                L.append(f"| {split} | {case} | {_f(o['explanation'], 3)} | {_f(n['explanation'], 3)} | {o['false_claims']} "
                         f"| {n['false_claims']} | {_f(o['gave_up'], pct=True)} | {_f(n['gave_up'], pct=True)} |")
        L += ["", "Reuse, default config: paired diff (with minus without), old -> new.", "",
              "| lock | new scene | old without -> with | new without -> with | old diff | new diff |", "|---|---|---|---|---|---|"]
        for lock in REUSE_LOCKS:
            if lock not in m["reuse"] or lock not in old_m["reuse"]:
                continue
            for split in ("train", "test"):
                o, n = old_m["reuse"][lock][split], m["reuse"][lock][split]
                v = lambda e: "-" if e["v"] is None else f"{e['v']:.1f}"   # noqa: E731
                L.append(f"| {lock} | {split} | {v(o['cold'])} -> {v(o['warm'])} | {v(n['cold'])} -> "
                         f"{v(n['warm'])} | {_f(o['diff'])} | {_f(n['diff'])} |")
        L.append("")
    if old_p is not None and psum:
        po = physics_summary(old_p["episodes"])
        comp["physics_old"] = po
        L += ["PhysicsWorld, opened / episodes per case: old (no decoys, seeds 0-1) and new.", "",
              "| case | old | " + " | ".join(psum) + " |", "|---|---|" + "---|" * len(psum)]
        for case in list(po):
            L.append(f"| {case} | {po[case]['opened']}/{po[case]['n']} | "
                     + " | ".join(f"{ps[case]['opened']}/{ps[case]['n']}" if case in ps else "-" for ps in psum.values()) + " |")
        L.append("")
    out = {"mock": mocks, "physics": phys, "physics_summary": psum, "comparison": comp}
    (out_dir / "eval.json").write_text(json.dumps(out, indent=1))
    (out_dir / "eval.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'eval.md'} and eval.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mock")
    m.add_argument("--n", type=int, default=200, help="scenes per case and split")
    m.add_argument("--budget", type=int, default=1000)
    m.add_argument("--reuse", type=int, default=100, help="recipe pairs per lock state")
    m.add_argument("--agents", nargs="*", default=list(AGENTS))
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--processes", type=int, default=16)
    m.add_argument("--seed-offset", type=int, default=0, help="added to every scene and agent seed (fresh seeds)")
    m.add_argument("--config", choices=list(CONFIGS), default="default", help="locality and MockWorld decoys")
    m.add_argument("--out", default=None, help="output folder (default results/, with the old file names)")
    p = sub.add_parser("physics")
    p.add_argument("--types", nargs="*", default=["hd_knob_pull_left", "hd_lever_pull_right"])
    p.add_argument("--test-types", nargs="*", default=["hd_knob_pull_right"])
    p.add_argument("--cases", nargs="*", default=["latch", "deadbolt", "key_table", "key_drawer", "both", "no_solution"])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--budget", type=int, default=400)
    p.add_argument("--limit", type=float, default=900.0, help="wall seconds per episode")
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--seed-offset", type=int, default=0, help="added to every env and agent seed (fresh seeds)")
    p.add_argument("--config", choices=["default", "loc_off"], default="default", help="the agent's locality")
    p.add_argument("--decoys", action="store_true", help="the fixture's physical decoys (home_doors.DECOYS)")
    p.add_argument("--out", default=None, help="output folder (default results/, with the old file names)")
    g = sub.add_parser("gif")
    g.add_argument("--type", default="hd_knob_pull_left")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--budget", type=int, default=400)
    g.add_argument("--limit", type=float, default=1800.0)
    g.add_argument("--state", default="key")
    g.add_argument("--place", default="drawer")
    g.add_argument("--size", type=int, default=360)
    g.add_argument("--busy-frames", type=int, default=8, help="frames kept of each action that changed something")
    g.add_argument("--boring-frames", type=int, default=40, help="at most this many frames of actions that changed nothing")
    g.add_argument("--max-frames", type=int, default=160, help="frames before the explanation at the end")
    g.add_argument("--out", default="key_in_drawer.gif")
    sub.add_parser("report")
    r2 = sub.add_parser("report2")
    r2.add_argument("--out", default=str(RESULTS / "discovery_v2"))
    args = ap.parse_args()
    {"mock": run_mock, "physics": run_physics, "gif": run_gif, "report": lambda a: write_report(),
     "report2": lambda a: write_report2(Path(a.out).resolve())}[args.cmd](args)


if __name__ == "__main__":
    main()
