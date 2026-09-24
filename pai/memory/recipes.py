"""Recipes: what worked, stored so it can be done again (repeating a success, "explaining the expected").

A recipe is distilled from ONE successful episode:

    goal        the relation and the roles it binds: on(<block>, <plate>)
    steps       per subgoal, where the hand ended up, relative to the entity it was acting on
                (the object, its start position, or the target): a sketch of the solution in terms
                of the scene's roles, not of absolute positions, so it transfers to new layouts
    preconditions  facts true at the start that the success relied on (hand open and empty, the
                object not already there); checked before reuse
    why         the factors the success depended on, found by per-phase counterfactual replay in
                the world model (pai.causes.credit): "skipping the lift breaks 'lifted'",
                "carrying at half speed does not matter"; each judged by the subgoal it serves

Reuse: in a new episode with the same kind of goal, the recipe turns into a *proposal* for the
planner: a straight-line hand motion towards the current step's waypoint. The planner scores it
alongside its own random samples, so a recipe can only help: if the proposal is bad, MPPI simply
does not pick it. The measurable benefit is that a good proposal lets a much cheaper planner
(fewer samples) succeed, i.e. the agent needs less thinking for a task it has solved before.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pai.world.entities import GRIP, POS

GRIP_CLOSED = 0.05  # finger opening below which the object counts as held (pai.goals.relations)


@dataclass
class RecipeStep:
    subgoal: str
    reference: str          # "object" (where it is now), "object_start" or "target"
    offset: list[float]     # hand position at the end of the step minus the reference (m)
    duration: int           # planner steps the step took in the source episode


@dataclass
class Recipe:
    relation: str
    object_kind: str
    target_kind: str
    steps: list[RecipeStep]
    preconditions: list[str]
    why: list[dict] = field(default_factory=list)
    source_episode: int = -1
    uses: int = 0
    successes: int = 0

    def step(self, subgoal: str) -> RecipeStep | None:
        return next((s for s in self.steps if s.subgoal == subgoal), None)

    @property
    def success_rate(self) -> float:
        return (self.successes + 1) / (self.uses + 2)  # Laplace: a new recipe starts at 0.5

    def describe(self) -> str:
        lines = [f"Recipe for {self.relation}({self.object_kind}, {self.target_kind}), "
                 f"learned from episode {self.source_episode}, reused {self.uses}x ({self.successes} successes):"]
        for s in self.steps:
            off = ", ".join(f"{1000 * v:+.0f}" for v in s.offset)
            lines.append(f"  {s.subgoal:12s} hand at {s.reference} + ({off}) mm, ~{s.duration} steps")
        if self.preconditions:
            lines.append("  needs: " + "; ".join(self.preconditions))
        for w in self.why:
            judge = w.get("phase") or "goal"  # older recipes were judged by the final goal
            verdict = f"breaks '{judge}'" if w["necessary"] else "does not matter"
            lines.append(f"  why: {w['description']} -> {verdict} ({judge} cost {w['importance']:+.4f})")
        return "\n".join(lines)


def _kinds(names: list[str], objects: dict, obj: str, target: str) -> tuple[str, str]:
    return objects[obj].kind, objects[target].kind


def preconditions_of(tokens0: np.ndarray, i_o: int, i_t: int, grip_open: float = 0.07) -> list[str]:
    """Facts at the start of the source episode that the solution relied on."""
    facts = []
    if float(tokens0[0, GRIP][0]) > grip_open:
        facts.append("hand open")
    if np.linalg.norm(tokens0[i_o, POS][:2] - tokens0[i_t, POS][:2]) > 0.05:
        facts.append("object not already at the target")
    return facts


def check_preconditions(recipe: Recipe, tokens0: np.ndarray, i_o: int, i_t: int) -> list[str]:
    """The preconditions that do NOT hold now (empty = the recipe applies)."""
    now = set(preconditions_of(tokens0, i_o, i_t))
    return [p for p in recipe.preconditions if p not in now]


def extract_recipe(record, relation: str, obj: str, target: str, names: list[str], objects: dict,
                   subgoal_order: list[str], why: list[dict] | None = None) -> Recipe:
    """record: a successful EpisodeRecord with tokens (T+1, N, D) and phases {subgoal: (first, last)}."""
    tokens, i_o, i_t = record.tokens, names.index(obj), names.index(target)
    start = tokens[0, i_o, POS]
    steps = []
    for name in subgoal_order:
        if name not in record.phases:
            continue
        first, last = record.phases[name]
        hand = tokens[min(last, len(tokens) - 1), 0, POS]
        end = tokens[min(last, len(tokens) - 1)]
        # the object itself is a reference only while it is not in the hand (the hand may nudge it
        # before the grasp); once held, "relative to the object" would always be zero
        refs = {"object": end[i_o, POS]} if float(end[0, GRIP][0]) > GRIP_CLOSED else {}
        refs.update({"object_start": start, "target": end[i_t, POS]})  # ties go to the object
        # the reference is the entity the hand is acting on at the end of the step: the nearer one
        ref = min(refs, key=lambda k: np.linalg.norm((hand - refs[k])[:2]))
        steps.append(RecipeStep(name, ref, (hand - refs[ref]).round(4).tolist(), int(last - first)))
    o_kind, t_kind = _kinds(names, objects, obj, target)
    return Recipe(relation, o_kind, t_kind, steps, preconditions_of(tokens[0], i_o, i_t), why or [],
                  source_episode=record.episode)


def proposal(recipe: Recipe, subgoal: str, tokens: np.ndarray, start_obj: np.ndarray, i_t: int,
             horizon: int, dt: float, max_speed: float = 0.25, grip: float | None = None,
             i_o: int | None = None) -> np.ndarray | None:
    """Action sequence (horizon, 4) that moves the hand in a straight line towards the waypoint of
    the current step; None if the recipe has no such step."""
    st = recipe.step(subgoal)
    if st is None:
        return None
    ref = {"object_start": start_obj, "target": tokens[i_t, POS],
           "object": tokens[i_o, POS] if i_o is not None else start_obj}[st.reference]
    wp = np.asarray(ref) + np.asarray(st.offset)
    g = tokens[0, POS].astype(np.float64).copy()
    out = np.zeros((horizon, 4), np.float32)
    for h in range(horizon):
        v = np.clip((wp - g) / dt, -max_speed, max_speed)
        out[h, :3] = v
        g = g + v * dt
    out[:, 3] = 0.0 if grip is None else grip
    return out


class RecipeLibrary:
    """All recipes, recalled by the kind of goal. Saved as JSON."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.recipes: list[Recipe] = []
        if self.path and self.path.exists():
            for r in json.loads(self.path.read_text()):
                r["steps"] = [RecipeStep(**s) for s in r["steps"]]
                self.recipes.append(Recipe(**r))

    def add(self, recipe: Recipe) -> None:
        self.recipes.append(recipe)
        self.save()

    def recall(self, relation: str, object_kind: str, target_kind: str) -> Recipe | None:
        match = [r for r in self.recipes if (r.relation, r.object_kind, r.target_kind) == (relation, object_kind, target_kind)]
        return max(match, key=lambda r: r.success_rate) if match else None

    def record_use(self, recipe: Recipe, success: bool) -> None:
        recipe.uses += 1
        recipe.successes += int(success)
        self.save()

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(r) for r in self.recipes], indent=1))

    def __len__(self) -> int:
        return len(self.recipes)
