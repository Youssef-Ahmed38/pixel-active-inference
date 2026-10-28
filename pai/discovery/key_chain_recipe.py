"""Observation-only recipe induction and repair for the discovery skill interface.

No fixture roles, shape vocabulary, insertion ordering, or key-specific repair rule.
The supplied goal, motor primitives, discrete feature vocabulary and Bayesian rule
family remain given. The old integrated chain is deliberately left as a baseline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
from collections import deque

import numpy as np

from pai.discovery.agent import DiscoveryAgent, Locality
from pai.discovery.belief import context, entity_of
from pai.discovery.interface import Entity, Outcome
from pai.discovery.recipes import RecipeBook


class EvidenceWorld:
    """Narrow body interface; copy observations and discard semantic outcome notes.

    Slots prevent accidental passthrough of pid/truth/events. The policy still has
    Python access to its body implementation; this is a tested API boundary, not
    a security sandbox. Generic positions and joint/graspability are supplied.
    """
    __slots__ = ("_entities", "_features", "_actions", "_execute", "_goal")

    def __init__(self, world):
        self._entities, self._features = world.entities, world.features
        self._actions, self._execute, self._goal = world.actions, world.execute, world.goal

    @staticmethod
    def clean(e):
        return Entity(e.id, e.kind, e.joint, e.graspable, tuple(e.pos), {})

    def entities(self):
        return [self.clean(e) for e in self._entities()]

    def features(self):
        return dict(self._features())

    def actions(self):
        return list(self._actions())

    def goal(self):
        return dict(self._goal())

    def goal_reached(self):
        state = self.features()
        return all(state.get(f) == v for f, v in self.goal().items())

    def execute(self, action):
        o = self._execute(action)
        return Outcome(tuple(action), o.executed, o.stalled, copy.deepcopy(o.changed),
                       [self.clean(e) for e in o.revealed], o.cost, "")


def _translate(value, bindings):
    if not isinstance(value, str):
        return value
    if value in bindings:
        return bindings[value]
    if value.startswith("in:"):
        return "in:" + bindings.get(value[3:], value[3:])
    if value.startswith(("part:", "obj:")):
        prefix, rest = value.split(":", 1)
        eid, suffix = rest.split(".", 1)
        return f"{prefix}:{bindings.get(eid, eid)}.{suffix}"
    return value


@dataclass
class RecipeStep:
    action: tuple
    preconditions: dict
    effects: dict
    reveals: list = field(default_factory=list)
    source_step: int = 0


@dataclass
class LearnedChain:
    """A backward slice of actual successful transitions, with opaque variables."""
    goal_id: str
    entities: dict
    steps: list[RecipeStep]
    rule: dict
    source: str

    @classmethod
    def learn(cls, log, source=""):
        if not log.reached_goal or not log.actions:
            return None
        wanted = dict(log.goal)
        needed_entities = set()
        selected = []
        # These are the discovery belief's conditions, not simulator conditions.
        causal = (log.final_map or {}).get("literals", {})
        for i in reversed(range(len(log.actions))):
            a, out, before = log.actions[i], log.outcomes[i], log.states[i]
            if not out.executed:
                continue
            eff = {f: pair[1] for f, pair in out.changed.items()}
            reveals = {e.id for e in out.revealed}
            if not (any(f in wanted and wanted[f] == v for f, v in eff.items())
                    or reveals & needed_entities):
                continue
            for f, v in eff.items():
                if wanted.get(f) == v:
                    del wanted[f]
            pre = dict(context(a, before))
            if i == len(log.actions) - 1:
                pre.update(causal)
                # One success cannot identify every necessary condition. Keep
                # the demonstrated changes as conservative contextual guards;
                # unlike the MAP rule these are not claimed to be causal facts.
                pre.update({f: v for f, v in before.items()
                            if f != "body:hand" and log.states[0].get(f) != v})
            selected.append(RecipeStep(tuple(a), pre, eff, sorted(reveals), i))
            wanted.update(pre)
            needed_entities.difference_update(reveals)
            needed_entities.update(x for x in a[1:] if isinstance(x, str))
        selected.reverse()
        if not selected:
            return None
        # Plan through the empirical transition graph instead of copying all
        # exploratory loops. Operators and their contexts come ONLY from this
        # episode. The finite search cap is a given computational limit.
        initial = dict(log.states[0])
        queue = deque([(initial, [])])
        visited = {frozenset(initial.items())}
        compact = None
        while queue and len(visited) < 10000:
            st, path = queue.popleft()
            if all(st.get(f) == v for f, v in log.goal.items()):
                compact = path
                break
            for s in selected:
                if all(st.get(f) == v for f, v in s.preconditions.items()):
                    after = {**st, **s.effects}
                    key = frozenset(after.items())
                    if key not in visited:
                        visited.add(key)
                        queue.append((after, [*path, s]))
        if compact:
            selected = compact
        goal_id = entity_of(next(iter(log.goal)))
        goal = log.entities[goal_id]
        used = {goal_id}
        for s in selected:
            used.update(x for x in s.action[1:] if isinstance(x, str))
            for f, v in s.preconditions.items():
                if f != "body:hand":
                    used.add(entity_of(f))
                if v.startswith("in:"):
                    used.add(v[3:])
                elif v in log.entities:
                    used.add(v)
        # Names become r0, r1, ...: neither numeric pid nor lexical equality
        # across episodes can bind a role. Position similarity is exemplar based.
        bindings = {eid: f"r{i}" for i, eid in enumerate(sorted(used))}
        ents = {}
        for eid, rid in bindings.items():
            e = log.entities[eid]
            reveal_index = next((j for j, s in enumerate(selected) if eid in s.reveals), None)
            if reveal_index is not None:
                original = log.outcomes[selected[reveal_index].source_step].revealed
                e = next(x for x in original if x.id == eid)
            ents[rid] = {"kind": e.kind, "joint": e.joint, "graspable": e.graspable,
                         "offset": list(np.subtract(e.pos, goal.pos)), "revealed_after": reveal_index}
        def tr(d):
            return {_translate(k, bindings): _translate(v, bindings) for k, v in d.items()}
        return cls(bindings[goal_id], ents,
                   [RecipeStep(tuple(_translate(x, bindings) for x in s.action),
                               tr(s.preconditions), tr(s.effects),
                               [bindings[x] for x in s.reveals if x in bindings], s.source_step)
                    for s in selected], tr(causal), source)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = copy.deepcopy(d)
        d["steps"] = [RecipeStep(tuple(s["action"]), s["preconditions"], s["effects"],
                                 s["reveals"], s["source_step"]) for s in d["steps"]]
        return cls(**d)


class ChainMemory:
    def __init__(self):
        self.chains = []
        # Existing abstract causal-rule priors, with the shape term removed.
        self.priors = RecipeBook(b_shape=0.0)

    def learn(self, log, source=""):
        chain = LearnedChain.learn(log, source)
        if chain is not None:
            self.chains.append(chain)
            self.priors.learn(log, source)
        return chain


class RecipeDiscoveryAgent(DiscoveryAgent):
    """Try learned transitions; mismatches return control to Bayesian discovery.

    No key-specific stall interpretation: any failed action, unbound role,
    precondition mismatch or unexpected effect invalidates the remaining replay.
    Discovery then learns/repairs the new scene using its ordinary action model.
    """
    def __init__(self, seed=0, memory=None, **kwargs):
        super().__init__(seed=seed, locality=Locality.off(), extent=False,
                         object_trials=False, **kwargs)
        self.memory = memory
        self.chain = memory.chains[-1] if memory and memory.chains else None
        self.cursor = 0
        self.bindings = {}
        self.disabled = False
        self.repair_reason = ""
        self.replay_actions = 0
        self._pending = None

    def _bind(self, ep):
        c = self.chain
        self.bindings[c.goal_id] = ep.goal_part
        goal = ep.ents[ep.goal_part]
        # Assignment is by demonstrated generic joint/graspability and metric
        # offset, not by shape, size, fixture name or container membership.
        for rid, desc in c.entities.items():
            if rid in self.bindings:
                continue
            if desc.get("revealed_after") is not None and self.cursor <= desc["revealed_after"]:
                continue
            candidates = [e for e in ep.ents.values() if e.id not in self.bindings.values()
                          and (e.kind, e.joint, e.graspable) ==
                          (desc["kind"], desc["joint"], desc["graspable"])]
            if candidates:
                distances = [np.linalg.norm(np.subtract(e.pos, goal.pos) - desc["offset"])
                             for e in candidates]
                self.bindings[rid] = candidates[int(np.argmin(distances))].id

    def _invalidate(self, reason, withdraw_prior=False):
        self.disabled, self.repair_reason = True, reason
        if withdraw_prior and self.belief is not None:
            # A wrong transferred transition must not leave a strong recipe
            # prior steering the repair. Rebuild the ordinary uniform-locality
            # belief and replay its OWN observations; keep the learned effects.
            self.belief.probe_prior = lambda ps: np.zeros(len(ps))
            self.belief.lit_prior = lambda ls: np.zeros(len(ls))
            self.belief._build()
            self._plans.clear()
            self._commit = None
            self._failed.clear()

    def _exploit(self, ep, acts):
        if self.chain and not self.disabled:
            self._bind(ep)
            if self._pending is not None:
                out = ep.log.outcomes[-1]
                expected = {_translate(k, self.bindings): _translate(v, self.bindings)
                            for k, v in self._pending.items()}
                if (not out.executed or (out.stalled and not expected)
                        or any(ep.state.get(f) != v for f, v in expected.items())):
                    self._invalidate("observed transition contradicted the recipe", withdraw_prior=True)
                self._pending = None
            if not self.disabled and self.cursor < len(self.chain.steps):
                s = self.chain.steps[self.cursor]
                tr = lambda x: _translate(x, self.bindings)
                a = tuple(tr(x) for x in s.action)
                pre = {tr(k): tr(v) for k, v in s.preconditions.items()}
                if a in acts and all(ep.state.get(f) == v for f, v in pre.items()):
                    self.cursor += 1
                    self.replay_actions += 1
                    self._pending = s.effects
                    return a, "recipe"
                self._invalidate("roles unavailable or observed preconditions changed")
        if self.disabled and sum(s == ep.state for s in ep.log.states[-20:]) >= 3:
            # A transferred plan can strand discovery in a reversible cycle.
            # Once the same observation recurs three times, test an untried
            # action/context instead. This is generic epistemic novelty, with
            # no special release/turn/key/shape sequence.
            novel = [a for a in acts if self.model.untested(a, ep.state)]
            if novel:
                return novel[int(self.rng.integers(len(novel)))], "repair_experiment"
        return super()._exploit(ep, acts)

    def run(self, world, budget_actions=800, recipes=None):
        return super().run(EvidenceWorld(world), budget_actions,
                           recipes=self.memory.priors if self.memory else None)


FAILURE_CAUSES = ("locked", "latched", "wrong_key", "key_not_turned", "blocked")


def failure_tokens(log, i):
    """After-episode evidence, including later interventions; never event labels.

    Entity-relative tokens discard opaque IDs. Later evidence is legitimate for
    a retrospective explanation, not available for an online prediction.
    """
    a, st, out = log.actions[i], log.states[i], log.outcomes[i]
    goal = next(iter(log.goal))
    tokens = ["kind=" + a[0], "stalled=" + str(out.stalled)]
    oid = a[1] if a[0] == "turn_held" else st.get("body:hand", "empty")
    where = st.get(f"obj:{oid}.where", "none")
    tokens += ["where=" + ("in" if where.startswith("in:") else where),
               "angle=" + st.get(f"obj:{oid}.angle", "none")]
    inserted = [f.split(":", 1)[1].split(".")[0] for f, v in st.items()
                if f.endswith(".where") and v.startswith("in:")]
    tokens.append("inserted=" + str(bool(inserted)))
    tokens.append("inserted_turned=" + str(any(st.get(f"obj:{e}.angle") != "rest" for e in inserted)))
    tokens.append("turned_parts=" + str(min(2, sum(f.startswith("part:") and f.endswith(".angle")
                                                      and v != "rest" for f, v in st.items()))))
    future = list(zip(log.actions[i + 1:], log.outcomes[i + 1:]))
    tokens.append("object_later_turns=" + str(any(b[0] == "turn_held" and b[1] == a[1]
                      and o.executed and not o.stalled and o.changed for b, o in future)))
    tokens.append("later_object_effect=" + str(any(b[0] == "turn_held" and o.executed
                                                  and o.changed for b, o in future)))
    successful = [b for b, o in future if goal in o.changed and o.changed[goal][1] == log.goal[goal]]
    tokens.append("later_goal=" + str(bool(successful)))
    tokens.append("same_final_kind=" + str(bool(successful) and successful[-1][0] == a[0]))
    tokens.append("same_final_entity=" + str(bool(successful) and successful[-1][1] == a[1]))
    return tokens


class FailureLikelihood:
    """Teacher-labelled categorical likelihoods fitted only on development logs.

    Symmetric Dirichlet smoothing .5, uniform class prior, posterior threshold
    .65 and unknown abstention are given, rather than learned mechanisms.
    """
    def __init__(self, threshold=0.65):
        self.threshold = threshold
        self.counts = {c: {} for c in FAILURE_CAUSES}
        self.totals = {c: 0 for c in FAILURE_CAUSES}
        self.vocab = {}

    def fit(self, examples):
        for tokens, label in examples:
            if label not in self.counts:
                raise ValueError(label)
            self.totals[label] += 1
            for tok in tokens:
                field, value = tok.split("=", 1)
                self.vocab.setdefault(field, set()).add(value)
                self.counts[label][tok] = self.counts[label].get(tok, 0) + 1
        return self

    def predict(self, tokens):
        lp = []
        for c in FAILURE_CAUSES:
            score = 0.0
            for tok in tokens:
                field = tok.split("=", 1)[0]
                den = self.totals[c] + .5 * (len(self.vocab.get(field, ())) + 1)
                score += np.log((self.counts[c].get(tok, 0) + .5) / den)
            lp.append(score)
        p = np.exp(np.asarray(lp) - max(lp)); p /= p.sum()
        index = int(np.argmax(p))
        return {"cause": FAILURE_CAUSES[index] if p[index] >= self.threshold else "unknown",
                "posterior": dict(zip(FAILURE_CAUSES, map(float, p))), "evidence": tokens}

    def to_dict(self):
        return {"threshold": self.threshold, "counts": self.counts, "totals": self.totals,
                "vocab": {k: sorted(v) for k, v in self.vocab.items()}}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["threshold"])
        m.counts, m.totals = copy.deepcopy(d["counts"]), dict(d["totals"])
        m.vocab = {k: set(v) for k, v in d["vocab"].items()}
        return m
