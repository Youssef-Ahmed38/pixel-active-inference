"""MockWorld: a fast symbolic home door with the same mechanics as pai.envs.home_doors.

It implements pai.discovery.interface.World so the discovery agent can be developed and tested in
milliseconds per episode. The mechanics mirror HomeDoorRuntime, discretised:

- latch: a spring latch holds a closed door unless the handle is turned (knob), pressed down (lever) or
  pressed in (push bar) *while* the door is moved, i.e. only 'turn_pull' / 'turn_push' with a releasing
  direction (either for a two-way knob, +1 for a one-way knob, -1 for a lever) or 'press_push' (bar)
  moves a latched door. The handle is spring-returned, so turning it alone leaves no trace in the
  features. Lock state "unlocked" has no latch catch at all (the door just swings); "latch" is the
  physics' "unlocked" (latch only).
- swing: a pull door moves only with pull-kind probes, a push door only with push-kind ones.
- thumb-turn deadbolt: a hinge part whose angle ("pos" = a quarter turn = bolt thrown) drives the bolt.
- key deadbolt: a lock cylinder (hinge, not graspable) whose plug turns only while a matching key is
  inserted: insert(key, cylinder), then turn_held(key, dir). As in the physics the plug drives the
  tailpiece rigidly (tail = a0 - u * plug, a0 fixed when the key engages), so a key has to stay turned
  to keep the bolt back, a wrong key inserts but stalls when turned, and a turned key cannot be pulled.
- keys (1-3, one matching) lie on the cabinet top ("table"), the wall shelf, inside the drawer (hidden
  until the drawer is pulled open) or in the other room (hidden behind the door: no solution when the
  door is key-locked). Decoys lie on reachable places.
- distractors: a dial on the cabinet (a persistent hinge part), a small cabinet door with a knob-like
  handle (an empty container), a coat hook, a pen-like object that fits nowhere; and, so that "close to
  the door" is not an oracle for "matters", door-mounted ones: a persistent round dial on the leaf next
  to the handle (a privacy indicator) and a coat hook on the leaf.
- shared shapes (Scenario.shared_shapes, off by default): so that a part's shape does not say which
  mechanism it is, decoys take the mechanisms' shapes: a non-graspable disc on the cabinet front like
  the lock cylinder (nothing goes into it), a persistent wing-shaped turn on the leaf like the
  thumb-turn (it bolts nothing), and the door dial (then always present) takes the handle's shape and
  size. Without the option each shape belongs to one mechanism (the lock cylinder is the only
  non-graspable disc, the thumb-turn the only wing).
- parts mounted on the leaf (handle, thumb-turn, door dial, door hook, door wing) carry the door when pulled or
  pushed, as in pai.discovery.physics (their carrier is the door hinge); only the handle releases the
  latch. The thumb-turn's thrown position reads "pos" (turned +1), as in physics.py where the thumb's
  angle is its physical orientation.
- one hand: part skills need an empty hand, object skills need that object in hand.
- skill noise: each action fails to execute with probability p_fail (nothing happens, time is spent);
  blocked motions stall.

Entity ids are opaque and shuffled per seed; features and attributes carry no semantic names.
Scenario / sample_scenario draw door types from home_doors' registry (with its train/test splits),
lock states and key placements. truth() is for scoring only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pai.discovery.interface import Action, Entity, Outcome
from pai.envs.home_doors import HOME_DOOR_TYPES

LOCK_STATES = ("unlocked", "latch", "deadbolt", "key", "both")
PLACEMENTS = ("table", "shelf", "drawer", "other_room")
BASE_COST = {"pick": 3.0, "release": 1.5, "insert": 4.0, "turn_held": 2.0, "turn": 2.0, "pull": 3.0,
             "push": 3.0, "turn_pull": 4.0, "turn_push": 4.0, "press_push": 3.5}
TRAVEL = 1.2        # s per metre the hand travels between targets
STALL_EXTRA = 1.0   # s spent pushing against something that does not give


def scenario_types(split: str | None = None) -> list[str]:
    return [n for n, t in HOME_DOOR_TYPES.items() if split is None or t.split == split]


def available_states(type_name: str) -> list[str]:
    f = HOME_DOOR_TYPES[type_name].fixed
    return [s for s in LOCK_STATES if s in ("unlocked", "latch") or (s == "deadbolt" and f["has_thumb"])
            or (s == "key" and f["has_key"]) or (s == "both" and f["has_thumb"] and f["has_key"])]


@dataclass
class Scenario:
    type_name: str
    lock_state: str
    key_place: str | None           # where the matching key lies (None without a key lock)
    n_keys: int = 1
    seed: int = 0
    dial: bool = True
    cabinet: bool = True            # the small cabinet door (an empty container)
    pen: bool = False               # a non-key elongated object
    door_dial: bool = False         # a persistent dial on the door leaf, next to the handle
    door_hook: bool = False         # a coat hook on the door leaf
    shared_shapes: bool = False     # decoys shaped like the mechanisms (a disc, a wing, a handle-like dial)
    p_fail: float = 0.07
    extra: dict = field(default_factory=dict)

    @property
    def fixed(self) -> dict:
        return HOME_DOOR_TYPES[self.type_name].fixed

    @property
    def solvable(self) -> bool:
        return not (self.lock_state in ("key", "both") and self.key_place == "other_room")


def sample_scenario(rng: np.random.Generator, split: str = "train", lock_state: str | None = None,
                    key_place: str | None = None, type_name: str | None = None, p_fail: float = 0.07,
                    shared_shapes: bool = False) -> Scenario:
    """A random scene: a door type from the split that supports `lock_state`, placements, distractors.
    shared_shapes: decoys shaped like the mechanisms (the door dial then always present); the random
    draws here are the same as without it (the MockWorld built from it differs in more than the decoys)."""
    names = [n for n in scenario_types(split) if lock_state is None or lock_state in available_states(n)]
    if type_name is None:
        if not names:
            raise ValueError(f"no {split} type supports {lock_state!r}")
        type_name = names[int(rng.integers(len(names)))]
    states = available_states(type_name)
    state = lock_state or states[int(rng.integers(len(states)))]
    if state not in states:
        raise ValueError(f"{type_name}: lock state {state!r} not available (have {states})")
    has_key = HOME_DOOR_TYPES[type_name].fixed["has_key"]
    place = None
    if has_key:
        place = key_place or str(rng.choice(PLACEMENTS, p=[0.3, 0.25, 0.3, 0.15]))
    return Scenario(type_name, state, place, n_keys=int(rng.integers(1, 4)) if has_key else 0,
                    seed=int(rng.integers(1 << 30)), dial=bool(rng.random() < 0.8), cabinet=bool(rng.random() < 0.6),
                    pen=bool(rng.random() < 0.5), p_fail=p_fail, door_dial=bool(rng.random() < 0.5) or shared_shapes,
                    door_hook=bool(rng.random() < 0.3), shared_shapes=shared_shapes)


class MockWorld:
    """One episode of a symbolic home door (interface.World)."""

    def __init__(self, scenario: Scenario):
        self.sc = sc = scenario
        self.rng = np.random.default_rng(sc.seed)
        f = sc.fixed
        rng = self.rng
        self.swing, self.handle = f["swing"], f["handle"]
        self.knob_dir = f["knob_dir"]
        ls = 1.0 if f["hinge"] == "left" else -1.0          # latch side (+y when hinged on the left)
        w = float(rng.uniform(0.82, 0.92))
        hz = float(rng.uniform(0.95, 1.02))
        hy = ls * (w / 2 - 0.07)
        cab_y = ls * float(rng.uniform(0.95, 1.25))          # the side cabinet stands on the latch side
        # internal name -> (joint, graspable, pos, attrs)
        spec: dict[str, tuple] = {"door": ("hinge", False, (0.0, 0.0, 1.0), {"shape": "panel", "size": w})}
        hattrs = {"knob_round": ("hinge", {"shape": "round", "size": 0.058}),
                  "knob_ball": ("hinge", {"shape": "ball", "size": 0.064}),
                  "lever": ("hinge", {"shape": "bar", "size": 0.12}),
                  "push_bar": ("slide", {"shape": "bar", "size": 0.6})}[self.handle]
        spec["handle"] = (hattrs[0], True, (0.07, hy, hz), hattrs[1])
        if f["has_thumb"]:
            spec["thumb"] = ("hinge", True, (0.05, hy, hz + float(rng.uniform(0.27, 0.31))), {"shape": "wing", "size": 0.07})
        if f["has_key"]:
            spec["cylinder"] = ("hinge", False, (0.01, hy, hz + float(rng.uniform(0.13, 0.16))), {"shape": "disc", "size": 0.044})
        spec["drawer"] = ("slide", True, (0.25, cab_y, 0.62), {"shape": "panel", "size": 0.4})
        if sc.cabinet:
            spec["cab_door"] = ("hinge", False, (0.25, cab_y + ls * 0.3, 0.35), {"shape": "panel", "size": 0.3})
            spec["cab_knob"] = ("hinge", True, (0.28, cab_y + ls * 0.2, 0.4), {"shape": "round", "size": 0.03})
        if sc.dial:
            spec["dial"] = ("hinge", True, (0.1, cab_y - ls * 0.1, 0.83), {"shape": "round", "size": 0.05})
        spec["hook"] = ("none", True, (0.0, -ls * float(rng.uniform(0.6, 0.8)), 1.6), {"shape": "peg", "size": 0.06})
        if sc.door_dial:   # on the leaf between the handle and the hinge side, a hand's width away
            spec["door_dial"] = ("hinge", True, (0.05, hy - ls * float(rng.uniform(0.1, 0.16)),
                                                 hz + float(rng.uniform(-0.05, 0.22))), {"shape": "round", "size": 0.05})
        if sc.door_hook:
            spec["door_hook"] = ("none", True, (0.04, float(rng.uniform(-0.15, 0.15)), 1.75), {"shape": "peg", "size": 0.06})
        if sc.shared_shapes:
            # their places from a separate stream (the ids, colours and draws after this still differ)
            srng = np.random.default_rng([sc.seed, 1])
            spec["cab_disc"] = ("hinge", False, (0.26, cab_y - ls * float(srng.uniform(0.08, 0.14)), 0.72),
                                {"shape": "disc", "size": 0.044})
            spec["door_wing"] = ("hinge", True, (0.05, hy - ls * float(srng.uniform(0.1, 0.3)),
                                                 hz + float(srng.uniform(0.3, 0.45))), {"shape": "wing", "size": 0.07})
            if "door_dial" in spec:
                j, g, pos, _ = spec["door_dial"]
                spec["door_dial"] = (j, g, pos, {"shape": hattrs[1]["shape"], "size": hattrs[1]["size"]})
        names = list(spec)
        ids = [f"p{i + 1}" for i in rng.permutation(len(names))]
        self.pid = dict(zip(names, ids))
        self.parts = {self.pid[n]: Entity(self.pid[n], "part", j, g, tuple(float(x) for x in pos),
                                          {**a, "colour": _colour(rng)}) for n, (j, g, pos, a) in spec.items()}
        self.pname = {v: k for k, v in self.pid.items()}
        # places for objects
        self.places = {"table": (0.2, cab_y, 0.86), "shelf": (0.1, cab_y + ls * 0.1, 1.35),
                       "drawer": (0.25, cab_y, 0.6), "other_room": (-1.5, 0.3, 0.75)}
        # objects: keys (index 0 of the shuffled list matches when there is a lock) and the pen
        objs = []
        match_i = int(rng.integers(sc.n_keys)) if sc.n_keys else -1
        decoy_places = ("table", "shelf", "drawer")
        for i in range(sc.n_keys):
            place = sc.key_place if i == match_i else str(rng.choice(decoy_places))
            objs.append({"name": f"key{i}", "fits": True, "matches": i == match_i, "place": place,
                         "attrs": {"shape": "elongated", "size": round(float(rng.uniform(0.09, 0.12)), 3)}})
        if sc.pen:
            objs.append({"name": "pen", "fits": False, "matches": False, "place": str(rng.choice(decoy_places)),
                         "attrs": {"shape": "elongated", "size": round(float(rng.uniform(0.14, 0.16)), 3)}})
        oids = [f"o{i + 1}" for i in rng.permutation(len(objs))]
        self.objs = {}
        for o, oid in zip(objs, oids):
            jit = rng.uniform(-0.05, 0.05, size=3) * (1, 1, 0)
            pos = tuple(float(x) for x in np.add(self.places[o["place"]], jit))
            self.objs[oid] = {**o, "id": oid, "where": "slot", "angle": 0, "pos": pos, "moved": False,
                              "attrs": {**o["attrs"], "colour": _colour(rng)}}
        # mechanism state
        st = sc.lock_state
        self.latch = st != "unlocked"
        self.thumb = 1 if st in ("deadbolt", "both") and "thumb" in self.pid else 0
        self.tail = 1 if st in ("key", "both") and "cylinder" in self.pid else 0   # key deadbolt thrown
        self.unlock_sign = int(rng.choice([-1, 1]))
        self.plug, self.a0 = 0, self.tail
        self.door_open = self.drawer_open = self.cab_open = False
        self.dial_q = {n: 0 for n in ("dial", "door_dial", "door_wing") if n in self.pid}   # persistent dials
        self.hand: str | None = None
        self.hand_pos = np.array([0.6, 0.0, 1.0])
        self.t = 0.0
        self.n = 0
        self.events: list[dict] = []
        self._log("reset", truth_state=st)

    # ------------------------------------------------------------------ observation
    def _visible(self, o: dict) -> bool:
        if o["moved"]:
            return True
        return not ((o["place"] == "drawer" and not self.drawer_open) or (o["place"] == "other_room" and not self.door_open))

    def entities(self) -> list[Entity]:
        out = list(self.parts.values())
        for oid, o in self.objs.items():
            if self._visible(o):
                out.append(Entity(oid, "object", "none", True, o["pos"], dict(o["attrs"])))
        return out

    def features(self) -> dict[str, str]:
        ang = lambda q: "rest" if q == 0 else ("pos" if q > 0 else "neg")   # noqa: E731
        f = {}
        P = self.pid
        f[f"part:{P['door']}.open"] = "open" if self.door_open else "closed"
        f[f"part:{P['handle']}.angle"] = "rest"                 # spring-returned: never stays turned
        if "thumb" in P:
            f[f"part:{P['thumb']}.angle"] = ang(self.thumb)
        if "cylinder" in P:
            f[f"part:{P['cylinder']}.angle"] = ang(self.plug)
        f[f"part:{P['drawer']}.open"] = "open" if self.drawer_open else "closed"
        if "cab_door" in P:
            f[f"part:{P['cab_door']}.open"] = "open" if self.cab_open else "closed"
            f[f"part:{P['cab_knob']}.angle"] = "rest"
        for n, q in self.dial_q.items():
            f[f"part:{P[n]}.angle"] = ang(q)
        for oid, o in self.objs.items():
            if not self._visible(o):
                continue
            w = o["where"]
            f[f"obj:{oid}.where"] = {"slot": "free", "free": "free", "held": "held"}.get(w, w)
            f[f"obj:{oid}.angle"] = ang(o["angle"])
        f["body:hand"] = self.hand or "empty"
        return f

    def actions(self) -> list[Action]:
        acts: list[Action] = []
        if self.hand is None:
            for pid, e in self.parts.items():
                if e.joint == "hinge" and e.graspable:
                    acts += [("turn", pid, 1), ("turn", pid, -1), ("pull", pid), ("push", pid)]
                    acts += [(k, pid, d) for k in ("turn_pull", "turn_push") for d in (1, -1)]
                elif e.joint == "slide" and e.graspable:
                    acts += [("pull", pid), ("push", pid), ("press_push", pid)]
                elif e.joint != "none":
                    acts.append(("push", pid))
                else:
                    acts += [("pull", pid), ("push", pid)]
            acts += [("pick", oid) for oid, o in self.objs.items() if self._visible(o)]
        else:
            o = self.objs[self.hand]
            acts += [("release", self.hand), ("turn_held", self.hand, 1), ("turn_held", self.hand, -1)]
            if o["where"] == "held":
                acts += [("insert", self.hand, pid) for pid in self.parts]
        return acts

    def goal(self) -> dict[str, str]:
        return {f"part:{self.pid['door']}.open": "open"}

    def goal_reached(self) -> bool:
        return self.door_open

    # ------------------------------------------------------------------ acting
    def target_pos(self, action: Action) -> np.ndarray:
        eid = action[2] if action[0] == "insert" else action[1]
        return np.asarray(self.parts[eid].pos if eid in self.parts else self.objs[eid]["pos"], float)

    def execute(self, action: Action) -> Outcome:
        if action not in self.actions():
            raise ValueError(f"action {action} not available")
        before = self.features()
        n_ent = {e.id for e in self.entities()}
        kind = action[0]
        tgt = self.target_pos(action)
        cost = BASE_COST[kind] + TRAVEL * float(np.linalg.norm(tgt - self.hand_pos))
        self.hand_pos = tgt
        self.n += 1
        if self.rng.random() < self.sc.p_fail:
            self.t += cost * 0.6
            return Outcome(action, False, False, {}, [], cost * 0.6, "skill failed")
        stalled, notes = self._do(action)
        if stalled:
            cost += STALL_EXTRA
        self.t += cost
        after = self.features()
        changed = {k: (before.get(k), v) for k, v in after.items() if before.get(k) != v}
        revealed = [e for e in self.entities() if e.id not in n_ent]
        return Outcome(action, True, stalled, changed, revealed, cost, notes)

    def _do(self, a: Action) -> tuple[bool, str]:
        kind, eid = a[0], a[1]
        name = self.pname.get(eid)
        if kind in ("pick", "release", "insert", "turn_held"):
            return self._object(a)
        d = a[2] if len(a) > 2 else 0
        if kind == "turn":
            return self._turn(name, d), ""
        if kind in ("turn_pull", "turn_push"):
            st = self._turn(name, d)
            fam = "pull" if kind == "turn_pull" else "push"
            released = name == "handle" and not st and self.handle != "push_bar"
            return self._move(name, fam, released) or st, ""
        if kind == "press_push":
            if name == "handle":
                return self._move(name, "push", True), ""
            return self._move(name, "push", False), ""
        return self._move(name, kind, False), ""

    def _turn(self, name: str, d: int) -> bool:
        """Turn a hinge part; returns stalled."""
        if name == "handle":
            if self.handle == "lever":
                return d != -1
            return self.knob_dir == "one" and d != 1
        if name == "thumb":
            if d == 1 and self.thumb == 0:
                self.thumb = 1
                self._log("bolt_extended", bolt="thumb")
                return False
            if d == -1 and self.thumb == 1:
                self.thumb = 0
                self._log("bolt_retracted", bolt="thumb")
                return False
            return True
        if name in self.dial_q:
            if abs(self.dial_q[name] + d) > 1:
                return True
            self.dial_q[name] += d
            return False
        return name != "cab_knob"   # the cabinet knob turns (and springs back); a hook does not

    def _move(self, name: str, fam: str, released: bool) -> bool:
        """Pull or push a part (moving whatever it is attached to); returns stalled."""
        if name in ("door", "handle", "thumb", "door_dial", "door_hook", "door_wing"):   # the leaf and what is on it
            if self.door_open:
                return fam == "pull" if self.swing == "push" else False
            if name == "door" and fam == "pull":
                return True
            if released:
                self._log("knob_turned")
            latched = self.latch and not released
            bolted = self.thumb == 1 or self.tail == 1
            if fam == self.swing and not latched and not bolted:
                if released and self.latch:
                    self._log("latch_released")
                self.door_open = True
                self._log("door_opened")
                return False
            return True
        if name == "drawer":
            if fam == "pull" and not self.drawer_open:
                self.drawer_open = True
                self._log("drawer_opened")
                return False
            if fam == "push" and self.drawer_open:
                self.drawer_open = False
                return False
            return True
        if name in ("cab_knob", "cab_door"):
            if fam == "pull" and name == "cab_knob" and not self.cab_open:
                self.cab_open = True
                return False
            if fam == "push" and self.cab_open:
                self.cab_open = False
                return False
            return True
        return True    # dial, hook, cylinder, cabinet disc: fixed to the furniture

    def _object(self, a: Action) -> tuple[bool, str]:
        kind, oid = a[0], a[1]
        o = self.objs[oid]
        cyl = self.pid.get("cylinder")
        if kind == "pick":
            if o["where"] in ("slot", "free"):
                o["where"], o["moved"] = "held", True
                self.hand = oid
                return False, ""
            if self.plug == 0:                                  # in the cylinder at home: pulls out
                o["where"] = "held"
                self.hand = oid
                self._log("key_removed", key=oid, matches=o["matches"])
                return False, ""
            self.hand = oid                                     # turned: grasp it where it is
            return False, "grasped in place"
        if kind == "release":
            self.hand = None
            if o["where"] == "held":
                o["where"] = "free"
                o["pos"] = tuple(float(x) for x in np.add(self.places["table"], self.rng.uniform(-0.05, 0.05, 3) * (1, 1, 0)))
            return False, ""
        if kind == "insert":
            p = a[2]
            occupied = any(x["where"] == f"in:{p}" for x in self.objs.values())
            if p == cyl and o["fits"] and self.plug == 0 and not occupied:
                o["where"] = f"in:{p}"
                o["pos"] = self.parts[p].pos
                if o["matches"]:
                    self.a0 = self.tail
                self._log("key_inserted", key=oid, matches=o["matches"])
                return False, ""
            return True, ""
        # turn_held
        d = a[2]
        if o["where"] == "held":
            return False, "turned in the hand"
        if not o["matches"]:
            return True, ""
        plug = self.plug + d
        tail = self.a0 - self.unlock_sign * plug
        if abs(plug) > 1 or tail not in (0, 1):
            return True, ""
        prev = self.tail
        self.plug, self.tail = plug, tail
        o["angle"] = plug
        if plug != 0:
            self._log("key_turned", key=oid)
        if prev != tail:
            self._log("bolt_retracted" if tail == 0 else "bolt_extended", bolt="key")
        return False, ""

    def _log(self, kind: str, **details) -> None:
        self.events.append({"n": self.n, "type": kind, **details})

    # ------------------------------------------------------------------ scoring only
    def truth(self) -> dict:
        sc = self.sc
        match = next((oid for oid, o in self.objs.items() if o["matches"]), None)
        rel = {"knob_round": "turn", "knob_ball": "turn", "lever": "turn", "push_bar": "press"}[self.handle]
        dirs = {"push_bar": [], "lever": [-1]}.get(self.handle, [1, -1] if self.knob_dir == "both" else [1])
        st = sc.lock_state
        return {"type": sc.type_name, "lock_state": st, "swing": self.swing, "handle_kind": self.handle,
                "latched": st != "unlocked", "deadbolt": st in ("deadbolt", "both") and "thumb" in self.pid,
                "key_locked": st in ("key", "both"), "key_place": sc.key_place if match else None,
                "matching_key": match, "release": rel, "release_dirs": dirs, "unlock_sign": self.unlock_sign,
                "keys": {oid: {"matches": o["matches"], "place": o["place"], "fits": o["fits"]}
                         for oid, o in self.objs.items()},
                "parts": dict(self.pid), "solvable": sc.solvable, "events": list(self.events)}


def _colour(rng: np.random.Generator) -> tuple:
    return tuple(round(float(x), 2) for x in rng.uniform(0.1, 0.9, 3))
