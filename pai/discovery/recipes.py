"""Recipes: a solved episode, abstracted by generic roles, reused as a prior on the next scene.

After a success the explanation (pai.discovery.explain) says which probe moved the goal part and which
conditions had to hold. A Recipe stores that without any ids, by the *role* each entity played, described
only by what any scene shows about it: part or object, joint (hinge / slide / none), graspable, shape,
whether it sits near the goal part, and where on the goal part it sits (its offset from the goal's
centre relative to the goal's size). For example:

    probe       a 'released' move (turn while pulling) of a graspable round hinge part near the goal
    conditions  a graspable wing-shaped hinge part near the goal turned away from how it was (rest);
                a small elongated object in a non-graspable disc-shaped part near the goal, turned
    search      that object was found inside a container; objects of the same look that went in but
                did not turn are wrong ones
    steps       the order: open container, pick, insert, turn, release, turn thumb, probe

On a new scene (other ids, layout, door type) RecipeBook.priors() turns the book into additive log-prior
bonuses for the belief: per probe (released vs plain, pull vs push family, direction, role similarity of
the part) and per literal (the same feature kind and value class on an entity of a similar role, e.g.
"a disc-like part near the goal turned either way" or "an elongated object inside such a part"), plus a
bias on the value of opening containers.

Shape is one cue among five (weight b_shape = 0.2 of 1.3), never a key on its own: nothing here looks
an entity up by its shape, so a scene where a decoy shares a mechanism's shape (MockWorld with
Scenario.shared_shapes: a disc on the cabinet like the lock cylinder, a wing on the leaf like the
thumb-turn, a door dial shaped like the handle) only makes the shape term equal for the two, and the
other cues (joint, graspable, near, offset on the goal) have to tell them apart. b_shape = 0 removes
shape altogether (an ablation). The belief stays a belief: a recipe that does not fit (a door
without a lock) is overruled by the first failed probes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pai.discovery.belief import entity_of
from pai.discovery.explain import FAMILY, RELEASED, explain

NEAR = 0.7


def role(e, goal, ents: dict | None = None) -> dict:
    """Generic description of an entity's role, relative to the goal part: kind, joint, graspable, shape,
    near the goal, and where on the goal it sits ('offset': distance from the goal's centre in units of
    the goal's half-size, e.g. ~0.85 for a part by a door leaf's edge, ~0.5 for one half-way in).
    'offset' replaces a rank ("the graspable part closest to the goal"), which a second part on the goal
    (a dial beside the handle) would steal."""
    d = lambda x: float(np.linalg.norm(np.subtract(x.pos, goal.pos))) if goal is not None else np.inf   # noqa: E731
    near = d(e) <= NEAR
    half = float(goal.attrs.get("size", 1.0)) / 2 if goal is not None else 1.0
    return {"kind": e.kind, "joint": e.joint, "graspable": bool(e.graspable), "shape": e.attrs.get("shape"),
            "near": bool(near), "is_goal": goal is not None and e.id == goal.id,
            "offset": d(e) / max(half, 1e-6) if near else None}


def similarity(a: dict, b: dict, offset_scale: float = 0.1, shape: float = 0.2) -> float:
    """0..1: how well two roles match (different kinds do not match at all); offsets match smoothly.
    `shape`: the weight of the shape cue (0 = ignore shapes)."""
    if a is None or b is None or a["kind"] != b["kind"]:
        return 0.0
    w = {"joint": 0.2, "graspable": 0.2, "shape": shape, "near": 0.3, "is_goal": 0.1}
    s = sum(wt for k, wt in w.items() if a.get(k) == b.get(k))
    oa, ob = a.get("offset"), b.get("offset")
    s += 0.3 * (float(np.exp(-abs(oa - ob) / offset_scale)) if oa is not None and ob is not None else float(oa == ob))
    return s / (sum(w.values()) + 0.3)


def value_class(feature: str, value: str) -> str:
    kind = feature.rsplit(".", 1)[-1]
    if kind == "angle":
        return "rest" if value == "rest" else "turned"
    if kind == "where":
        return "in" if value.startswith("in:") else value
    return value


@dataclass
class Recipe:
    probe_kind: str
    probe_dir: int | None
    probe_role: dict
    conditions: list = field(default_factory=list)   # [(feature kind, value class, entity role, in-part role)]
    key_role: dict | None = None
    key_found: str | None = None                      # "container" | "visible" | None
    wrong_objects: int = 0
    steps: list = field(default_factory=list)          # [(kind, role)] of the actions that mattered
    source: str = ""

    @property
    def released(self) -> bool:
        return self.probe_kind in RELEASED

    @property
    def family(self) -> str:
        return FAMILY[self.probe_kind]


def recipe_from_log(log, source: str = "") -> Recipe | None:
    """Abstract a successful episode into a Recipe (None if the episode did not reach the goal)."""
    if not log.reached_goal or not log.actions:
        return None
    c = explain(log).claims
    ents = log.entities
    (gf, _), = log.goal.items()
    goal = ents.get(entity_of(gf))
    R = lambda eid: role(ents[eid], goal, ents) if eid in ents else None   # noqa: E731
    p = c["probe"]
    conds = []
    for f, v in {**c["probably"], **c["conditions"]}.items():
        e = entity_of(f)
        inpart = R(v[3:]) if v.startswith("in:") else None
        conds.append((f.rsplit(".", 1)[-1], value_class(f, v), R(e), inpart))
    key = c["key_used"]
    steps = []
    wanted = {e for e in (c["deadbolt_part"], c["key_part"], key, c["key_found_in"]) if e}
    for a, o in zip(log.actions, log.outcomes):
        if o.executed and not o.stalled and (set(x for x in a[1:] if isinstance(x, str)) & wanted):
            steps.append((a[0], R(a[2] if a[0] == "insert" else a[1])))
    steps.append((p[0], R(p[1])))
    return Recipe(p[0], p[2] if len(p) > 2 else None, R(p[1]), conds, R(key) if key else None,
                  ("container" if c["key_found_in"] else "visible") if key else None,
                  len(c["wrong_keys"]), steps, source)


class RecipeBook:
    """A library of recipes and the priors they induce on a new scene."""

    def __init__(self, b_released: float = 1.5, b_family: float = 0.5, b_dir: float = 0.3, b_role: float = 1.5,
                 b_lit: float = 3.0, rho_container: float = 0.4, b_shape: float = 0.2):
        self.recipes: list[Recipe] = []
        self.b_released, self.b_family, self.b_dir, self.b_role = b_released, b_family, b_dir, b_role
        self.b_lit, self.rho_container, self.b_shape = b_lit, rho_container, b_shape

    def __len__(self) -> int:
        return len(self.recipes)

    def add(self, r: Recipe | None) -> None:
        if r is not None:
            self.recipes.append(r)

    def learn(self, log, source: str = "") -> Recipe | None:
        r = recipe_from_log(log, source)
        self.add(r)
        return r

    def priors(self, ents: dict, goal_id: str):
        """(probe_bonus(probes) -> array, lit_bonus(literals) -> array, reveal bias) for a new scene."""
        goal = ents.get(goal_id)
        roles = {eid: role(e, goal, ents) for eid, e in ents.items()}

        sim = lambda a, b: similarity(a, b, shape=self.b_shape)   # noqa: E731

        def rl(eid):
            if eid not in roles and eid in ents:
                roles[eid] = role(ents[eid], goal, ents)
            return roles.get(eid)

        def probe_bonus(probes):
            out = np.zeros(len(probes))
            for i, p in enumerate(probes):
                best = 0.0
                for r in self.recipes:
                    b = (self.b_released * ((p[0] in RELEASED) == r.released) + self.b_family * (FAMILY[p[0]] == r.family)
                         + self.b_dir * (len(p) > 2 and p[2] == r.probe_dir) + self.b_role * sim(rl(p[1]), r.probe_role))
                    best = max(best, b)
                out[i] = best
            return out

        def lit_bonus(lits):
            out = np.zeros(len(lits))
            for i, (f, v) in enumerate(lits):
                kind, vc, e = f.rsplit(".", 1)[-1], value_class(f, v), entity_of(f)
                best = 0.0
                for r in self.recipes:
                    for ck, cvc, crole, cin in r.conditions:
                        if ck != kind or cvc != vc:
                            continue
                        s = sim(rl(e), crole)
                        if cin is not None:
                            s *= sim(rl(v[3:]), cin)
                        best = max(best, self.b_lit * s * s)
                    # the object that fitted, wherever it ends up: also 'in' the part that turned
                    if r.key_role is not None and kind == "where" and vc == "in":
                        cin = next((c[3] for c in r.conditions if c[3] is not None), None)
                        part_role = next((c[2] for c in r.conditions if c[0] == "angle" and c[1] == "turned"), cin)
                        s = sim(rl(e), r.key_role) * sim(rl(v[3:]), part_role)
                        best = max(best, self.b_lit * s * s)
                out[i] = best
            return out

        rho = self.rho_container if any(r.key_found == "container" for r in self.recipes) else 0.0
        return probe_bonus, lit_bonus, rho
