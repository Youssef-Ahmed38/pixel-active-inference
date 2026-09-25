"""The discovery agent in other bodies: PhysicsWorld with any embodiment, a latch-free door, and a trace.

PhysicsWorld (pai.discovery.physics) already takes any body from pai.envs.embodiments: every skill runs
through the common palm-velocity action and the oracle's P-controllers on the palm target. Nothing here
changes how a skill runs, what the agent sees or what it may do. This module adds, for the evaluation
over bodies (scripts/discovery_bodies_eval.py):

    BodyWorld      PhysicsWorld with `latch=False`: the door's spring latch never catches (as if its
                   bolt were taped back), so "unlocked" means what it means in the mock: any move in the
                   right direction opens it. The home-door lock state "unlocked" still has the latch.
    TracedWorld    a wrapper that records, after every action, the Outcome and the ground truth the
                   agent never sees (the runtime's events with their details, which bolts are thrown),
                   and stops the episode at a wall-time limit. It returns the Outcome unchanged.
    failure_cause  why a solvable door was not opened, from that trace (scoring only, see below).

Failure causes. The solution of a door is a chain of sub-goals, in this order: the key bolt retracted
(itself: the drawer opened if the key is in it, the fitting key picked up, inserted into the lock,
turned), the thumb bolt retracted, the door opened with the latch released. A bolt the agent threw
itself (the thumb-turn turned the locking way, a key turned back) has to be undone as well, so a bolt
sub-goal is met when that bolt is retracted at the end of the episode. The first sub-goal not met is
the one that blocked the door, and the agent's attempts at it (the actions that do it when the body
does them right: pick that key, insert it into the lock, turn it the unlocking way, turn the thumb-turn
the unlocking way, a latch-releasing probe of the handle in the swing direction while every bolt is
retracted) give the cause:

    not_executed   every attempt failed as a skill (no grasp, could not lift it, dropped it)
    stalled        at least one attempt ran but the part did not move (an insert that missed, a turn
                   or a door probe that stalled although nothing held it any more)
    reasoning      no attempt at all (it never tried the needed action, or gave up: "no solution"),
                   or the sub-goal was met once and then undone by the agent
    unstable       the simulation went unstable (the world stops acting)
    timeout        the wall-time limit ran out (the agent was not done)

A "reasoning" or "timeout" failure with no attempt at all is not always the agent's: if nearly every
skill of the episode failed to execute (a body that cannot grasp the handle cannot learn that the door is
locked, so it never gets to the thumb-turn), `body_limited` reclassifies it as not_executed (why
"no_skill_ran").

The unlocking directions (key: turn_held +1, thumb-turn: turn -1) follow from the joint conventions of
pai.envs.home_doors (a thrown thumb-turn reads "pos"; the oracle turns the key plug positively), and
the evaluation checks them against every bolt_retracted event in its traces.
"""

from __future__ import annotations

import time

from pai.discovery.physics import PhysicsWorld
from pai.envs.door_scene import DoorSceneEnv
from pai.envs.home_doors import home_door_fixture

UNLOCK_SIGN = {"key": 1, "thumb": -1}
CAUSES = ("not_executed", "stalled", "reasoning", "unstable", "timeout")


def free_latch(runtime) -> None:
    """Keep the door's latch equality off at every control step (the rest of the runtime is unchanged)."""
    update = runtime.update

    def no_latch():
        update()
        runtime.data.eq_active[runtime.latch] = False
    runtime.update = no_latch


class BodyWorld(PhysicsWorld):
    """PhysicsWorld with any embodiment; latch=False: a door without a working latch."""

    def __init__(self, type_name: str = "hd_knob_pull_left", embodiment: str = "robotiq_2f85", seed: int = 0,
                 lock_state: str | None = None, key_place: str | None = None, latch: bool = True,
                 max_action_steps: int = 3000, **fixture_overrides):
        env = DoorSceneEnv(embodiment, home_door_fixture(type_name, **fixture_overrides), seed=seed)
        if not latch:
            free_latch(env.runtime)
        self.latch = latch
        super().__init__(type_name, embodiment, seed=seed, lock_state=lock_state, key_place=key_place, env=env,
                         max_action_steps=max_action_steps)


class WallLimit(Exception):
    pass


class TracedWorld:
    """World wrapper: after every action, the Outcome plus ground truth (for scoring only); raises WallLimit
    once `limit` wall seconds have passed."""

    def __init__(self, world: PhysicsWorld, limit: float = float("inf")):
        self.w, self.limit = world, limit
        self.t0 = time.time()
        self.trace: list[dict] = []
        self._n_events = len(world.rt.events)

    def __getattr__(self, name):
        return getattr(self.w, name)

    def bolts(self) -> dict:
        rt = self.w.rt
        return {k: rt.bolt_thrown(k) for k in rt.rec.bolts}

    def execute(self, action):
        if time.time() - self.t0 > self.limit:
            raise WallLimit()
        before = self.bolts()
        t1 = time.time()
        out = self.w.execute(action)
        ev = self.w.rt.events[self._n_events:]
        self._n_events = len(self.w.rt.events)
        self.trace.append({"a": list(action), "executed": bool(out.executed), "stalled": bool(out.stalled),
                           "notes": out.notes, "cost": float(out.cost), "wall": round(time.time() - t1, 2),
                           "changed": {k: list(v) for k, v in out.changed.items()},
                           "events": [{k: v for k, v in e.items() if k != "truth"} for e in ev],
                           "bolts_before": before, "bolts": self.bolts(), "opened": bool(self.w.truth()["opened"])})
        return out


def _solution_ids(world: PhysicsWorld) -> dict:
    """Ground-truth ids of the parts and of the fitting key (scoring only)."""
    tr = world.truth()
    ids = {name: i for i, name in tr["ids"].items()}
    match = next((n for n, k in tr["keys"].items() if k["matches"]), None)
    return {"door": ids["door"], "handle": ids["handle"], "drawer": ids["drawer"], "thumb": ids.get("thumb"),
            "cylinder": ids.get("cylinder"), "key": ids.get(match) if match else None,
            "key_place": tr["keys"][match]["slot"][0] if match else None}


def failure_cause(trace: list[dict], world: PhysicsWorld, init: dict, stopped: str) -> dict:
    """Why a solvable door stayed shut (see the module docstring). init: the bolts thrown at the start
    (TracedWorld.bolts() before the first action). Returns {"cause", "blocked_at", "attempts", "notes"}."""
    if any(t["opened"] for t in trace):
        return {"cause": None, "blocked_at": None, "attempts": 0, "notes": []}
    if world.unstable or any(t["notes"].startswith("aborted: unstable") for t in trace):
        return {"cause": "unstable", "blocked_at": None, "attempts": 0, "notes": []}
    ids = _solution_ids(world)
    end = trace[-1]["bolts"] if trace else init
    ever = lambda kind, **kw: any(e["type"] == kind and all(e.get(k) == v for k, v in kw.items())   # noqa: E731
                                  for t in trace for e in t["events"])
    a_is = lambda t, kind, *args: t["a"][0] == kind and list(t["a"][1:1 + len(args)]) == list(args)  # noqa: E731
    # where the fitting key was before each action (the trace holds changes only)
    where = "free"
    for t in trace:
        t["_in_cyl"] = where == f"in:{ids['cylinder']}"
        ch = t["changed"].get(f"obj:{ids['key']}.where") if ids["key"] else None
        if ch:
            where = ch[1]
    steps = []                      # (sub-goal, met at the end, met ever, attempt filter)
    if ids["key"] is not None and (init.get("key") or end.get("key")):
        if ids["key_place"] == "drawer":
            steps.append(("open drawer", ever("drawer_opened"), ever("drawer_opened"),
                          lambda t: t["a"][0] in ("pull", "turn_pull") and t["a"][1] == ids["drawer"]))
        picked = any(a_is(t, "pick", ids["key"]) and t["executed"] and t["notes"] == "lifted" for t in trace)
        steps.append(("pick key", picked or ever("key_inserted", matches=True), picked,
                      lambda t: a_is(t, "pick", ids["key"]) and not t["_in_cyl"]))
        ins = ever("key_inserted", matches=True)
        steps.append(("insert key", ins, ins, lambda t: a_is(t, "insert", ids["key"], ids["cylinder"])))
        steps.append(("turn key", not end.get("key"), ever("bolt_retracted", bolt="key"),
                      lambda t: a_is(t, "turn_held", ids["key"], UNLOCK_SIGN["key"]) and t["_in_cyl"]))
    if ids["thumb"] is not None and (init.get("thumb") or end.get("thumb")):
        steps.append(("turn thumb", not end.get("thumb"), ever("bolt_retracted", bolt="thumb"),
                      lambda t: t["a"][0] in ("turn", "turn_pull", "turn_push") and t["a"][1] == ids["thumb"]
                      and t["a"][2] == UNLOCK_SIGN["thumb"]))
    swing = world.rt.rec.spec.params["swing"]
    kinds = {"pull": ("turn_pull",), "push": ("turn_push", "press_push")}[swing]
    if not getattr(world, "latch", True):
        kinds = kinds + (swing,)
    steps.append(("open door", False, False,
                  lambda t: t["a"][0] in kinds and t["a"][1] in (ids["handle"], ids["door"])
                  and not any(t["bolts_before"].values())))
    out = {"cause": "reasoning", "blocked_at": None, "attempts": 0, "notes": [], "why": None}
    for name, met_end, met_ever, is_attempt in steps:
        if met_end:
            continue
        att = [t for t in trace if is_attempt(t)]
        notes = sorted({t["notes"].split("=")[0] if "=" in t["notes"] else t["notes"] for t in att})
        why = None
        if met_ever:
            cause, why = "reasoning", "undone"
        elif not att:
            cause = "timeout" if stopped == "timeout" else "reasoning"
            why = {"no_solution": "gave_up", "timeout": None}.get(stopped, "not_tried")
        elif not any(t["executed"] for t in att):
            cause = "not_executed"
        else:
            cause = "stalled"
        out = {"cause": cause, "blocked_at": name, "attempts": len(att), "notes": notes, "why": why}
        break
    for t in trace:
        t.pop("_in_cyl", None)
    return out


def body_limited(failure: dict | None, n_actions: int, n_not_executed: int, share: float = 0.9) -> dict | None:
    """failure_cause's result, with a "reasoning" or "timeout" failure of a sub-goal never attempted
    reclassified as not_executed when at least `share` of the episode's actions did not execute as skills."""
    if not failure or failure["cause"] not in ("reasoning", "timeout") or failure["attempts"] \
            or failure.get("why") == "undone":
        return failure
    if n_actions and n_not_executed / n_actions >= share:
        return {**failure, "cause": "not_executed", "why": "no_skill_ran"}
    return failure
