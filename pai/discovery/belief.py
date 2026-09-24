"""Belief over causal rules for reaching a goal, a learned action model, and information-gain scores.

Hypotheses. h = (probe, C) reads: "the goal feature is reached by the probe action `probe` (a move of
a part near the goal part, e.g. ('turn_pull', 'p2', +1)) while the conjunction C of feature literals
holds", e.g. C = {part:p4.angle = rest, part:p5.angle = neg}. Literals range over the current feature
vocabulary: every value of a part's angle / opening, and 'free' / 'in:<part>' (every part in view) for
every visible object ('held' is left out: probes need the hand, so such a rule could never be tested).
C has up to max_lits literals on distinct features (3-literal rules only over literals already seen
true, to stay tractable). The vocabulary grows as entities are revealed or new values are seen; the
hypothesis set is then rebuilt and the evidence replayed, and hypotheses the evidence has ruled out
are pruned. One more hypothesis, UNKNOWN, stands for "a cause I cannot express yet" (a hidden object,
or no solution at all): it predicts that every probe fails.

Prior. Occam: a penalty per literal (log L + 1.5 nats, so each size class carries e^-1.5/k! of the
mass of the empty rule), a proximity prior over probes and literals (causes act locally: parts on or
close to the goal part, measured from its extent, so everything mounted on a door leaf is equally
local), plus an additive transfer bonus per probe and per literal from recipes
(pai.discovery.recipes).

Likelihood. A probe that ran (reached, grasped; stalled counts, since a stall is a failure of the goal)
succeeds with 1 - eps if the probe is h's and C holds in the state before it, else with eps.

Action model. Which action changes which features is learned from experience: effects are counted per
(action, context), where the context is the values of the features of the entities the action involves
(its arguments, the part a held object sits in, objects inside those parts, and the hand). Untried
(action, context) pairs are the open questions of the action model; an entity's literals count as
reachable while such a pair exists for an action on it, or once they have been seen true.

Scores (score_actions), in bits per second of estimated action time. For a probe: expected information
gain (EIG) about h, charged the time to restore the state if the probe also changes it. For an action
with a known effect: one-step lookahead, the EIG of the best probe after the predicted change. For an
untried (action, context): P(change) (that action's record in other contexts, shrunk towards the rate
over all actions) times the mean lookahead over the changes it could make, plus a
value for revealing new entities (opening something that may hide things) proportional to the UNKNOWN
mass. Picking something up is also worth what it makes possible (insert, turn it). Any action that may
change the state while the probes here are still informative is charged the time to come back (the
state it leaves is one the other probes still need): without that, curiosity about a new (action,
context) pair walks away from a state that was just made worth testing.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import numpy as np

from pai.discovery.interface import PROBE_KINDS, Action

OBJECT_KINDS = ("insert", "turn_held", "release")
DOMAIN = {"angle": ("rest", "pos", "neg"), "open": ("closed", "ajar", "open")}
Literal = tuple  # (feature, value)


def entity_of(feature: str) -> str:
    """'part:p3.angle' -> 'p3'; 'obj:o2.where' -> 'o2'; 'body:hand' -> 'body'."""
    head = feature.split(":", 1)[1] if ":" in feature else feature
    return head.split(".", 1)[0]


def is_scene_feature(feature: str) -> bool:
    return feature.startswith("part:") or feature.startswith("obj:")


def action_entities(a: Action) -> list[str]:
    return [x for x in a[1:] if isinstance(x, str)]


def hb(p: np.ndarray | float) -> np.ndarray:
    """Binary entropy in bits."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ---------------------------------------------------------------------------------------- vocabulary
class Vocabulary:
    """Feature literals the hypotheses may use; grows with what comes into view.

    Angles and openings take their generic domains; an object's 'where' takes 'free' and 'in:<part>' for
    every part in view. 'held' is left out: the probes need the hand, so a rule "while I hold X" could
    never be tested (the agent knows its own body)."""

    def __init__(self, goal: dict):
        self.goal = dict(goal)
        self.values: dict[str, list[str]] = {}
        self.parts: list[str] = []

    def update(self, state: dict, parts: list[str] = ()) -> bool:
        changed = False
        for p in parts:
            if p not in self.parts:
                self.parts.append(p)
                for f, vals in self.values.items():
                    if f.endswith(".where"):
                        vals.append(f"in:{p}")
                changed = True
        for f, v in state.items():
            if not is_scene_feature(f) or f in self.goal:
                continue
            kind = f.rsplit(".", 1)[-1]
            vals = self.values.get(f)
            if vals is None:
                dom = ("free", *[f"in:{p}" for p in self.parts]) if kind == "where" else DOMAIN.get(kind, ())
                vals = self.values[f] = list(dom)
                changed = True
            if v not in vals and v != "held":
                vals.append(v)
                changed = True
        return changed

    def literals(self) -> list[Literal]:
        return [(f, v) for f in sorted(self.values) for v in self.values[f]]


# ---------------------------------------------------------------------------------------- belief
class Belief:
    """Posterior over h = (probe, C) plus UNKNOWN, vectorised over an enumerated hypothesis set."""

    def __init__(self, goal: dict, probes: list[Action], max_lits: int = 2, eps: float = 0.005,
                 p_unknown: float = 0.1, probe_prior=None, lit_prior=None, prune: float = 25.0,
                 occam: float = 1.5):
        self.goal = dict(goal)
        self.probes = list(probes)
        self.pidx = {a: i for i, a in enumerate(self.probes)}
        self.max_lits, self.eps, self.p_unknown, self.prune, self.occam = max_lits, eps, p_unknown, prune, occam
        self.probe_prior = probe_prior or (lambda probes: np.zeros(len(probes)))
        self.lit_prior = lit_prior or (lambda lits: np.zeros(len(lits)))
        self.vocab = Vocabulary(goal)
        self.history: list[tuple[int, dict, bool]] = []
        self.achieved: set[Literal] = set()
        self.lits: list[Literal] = []
        self.n_enumerated = 0

    # ------------------------------------------------------------------ building
    def update_vocab(self, state: dict, parts: list[str] = ()) -> bool:
        self.achieved |= {(f, v) for f, v in state.items() if is_scene_feature(f)}
        if self.vocab.update(state, parts) or not self.lits:
            self._build()
            return True
        return False

    def _build(self) -> None:
        self.lits = self.vocab.literals()
        self.lit_index = {l: i for i, l in enumerate(self.lits)}
        L, P = len(self.lits), len(self.probes)
        feat = [f for f, _ in self.lits]
        combos: list[tuple] = [()]
        for k in range(1, self.max_lits + 1):
            pool = range(L) if k <= 2 else [i for i in range(L) if self.lits[i] in self.achieved]
            combos += [c for c in itertools.combinations(pool, k) if len({feat[i] for i in c}) == k]
        M = len(combos)
        lits = np.full((M, max(1, self.max_lits)), L, dtype=np.int32)      # index L = padding (always true)
        for j, c in enumerate(combos):
            lits[j, :len(c)] = c
        size = np.array([len(c) for c in combos], float)
        alpha = np.log(max(L, 2)) + self.occam
        lb = np.r_[np.asarray(self.lit_prior(self.lits), float), 0.0]
        pb = np.asarray(self.probe_prior(self.probes), float)
        self.h_probe = np.repeat(np.arange(P, dtype=np.int32), M)
        self.h_lits = np.tile(lits, (P, 1))
        self.h_size = np.tile(size, P)
        self.log_prior = pb[self.h_probe] - alpha * self.h_size + lb[self.h_lits].sum(1)
        m = self.log_prior.max()
        self.log_prior_unk = m + np.log(np.exp(self.log_prior - m).sum()) + np.log(self.p_unknown / (1 - self.p_unknown))
        self.log_lik = np.zeros_like(self.log_prior)
        self.h_err = np.zeros(len(self.log_prior), np.int16)    # observations each hypothesis got wrong
        self.log_lik_unk = 0.0
        self.n_enumerated = len(self.log_prior)
        for pi, st, y in self.history:
            self._update(pi, st, y)
        self._compact()

    def _compact(self) -> None:
        """Drop hypotheses the evidence has ruled out (they cannot come back: likelihoods only multiply)."""
        post = self.log_prior + self.log_lik
        top = max(post.max(), self.log_prior_unk + self.log_lik_unk)
        keep = post > top - self.prune
        if keep.all():
            return
        for name in ("h_probe", "h_lits", "h_size", "log_prior", "log_lik", "h_err"):
            setattr(self, name, getattr(self, name)[keep])

    # ------------------------------------------------------------------ evidence
    def truth(self, state: dict) -> np.ndarray:
        t = np.array([state.get(f) == v for f, v in self.lits] + [True])
        return t

    def holds(self, state: dict) -> np.ndarray:
        return self.truth(state)[self.h_lits].all(1)

    def _update(self, pi: int, state: dict, success: bool) -> None:
        pred = (self.h_probe == pi) & self.holds(state)
        le, l1 = np.log(self.eps), np.log(1 - self.eps)
        ok = pred == success
        self.log_lik += np.where(ok, l1, le)
        self.h_err += ~ok
        self.log_lik_unk += le if success else l1

    def observe(self, probe: Action, state: dict, success: bool) -> None:
        pi = self.pidx.get(probe)
        if pi is None:
            return
        st = {f: v for f, v in state.items() if is_scene_feature(f)}
        self.history.append((pi, st, bool(success)))
        self._update(pi, st, success)
        self._compact()

    # ------------------------------------------------------------------ posterior
    def weights(self) -> tuple[np.ndarray, float]:
        """Normalised posterior of the enumerated hypotheses and of UNKNOWN."""
        post = self.log_prior + self.log_lik
        pu = self.log_prior_unk + self.log_lik_unk
        m = max(post.max(), pu)
        w = np.exp(post - m)
        wu = float(np.exp(pu - m))
        z = w.sum() + wu
        return w / z, wu / z

    def p_unknown_post(self) -> float:
        return self.weights()[1]

    def success_probs(self, state: dict, w: np.ndarray | None = None) -> np.ndarray:
        """P(goal | probe p now) for every probe."""
        if w is None:
            w = self.weights()[0]
        h = self.holds(state)
        q = np.bincount(self.h_probe[h], w[h], minlength=len(self.probes))
        return q * (1 - self.eps) + (1 - q) * self.eps

    def eig_probes(self, state: dict, w: np.ndarray | None = None) -> np.ndarray:
        return hb(self.success_probs(state, w)) - hb(self.eps)

    def literal_false_mass(self, state: dict, w: np.ndarray) -> np.ndarray:
        """Per literal: posterior mass of hypotheses in which it appears and is false now."""
        t = self.truth(state)
        out = np.zeros(len(self.lits) + 1)
        for k in range(self.h_lits.shape[1]):
            l = self.h_lits[:, k]
            m = ~t[l]
            out += np.bincount(l[m], w[m], minlength=len(self.lits) + 1)
        return out[:-1]

    def mass(self, lit_mask: np.ndarray, w: np.ndarray | None = None) -> float:
        """Posterior mass of hypotheses all of whose literals satisfy lit_mask (bool per literal)."""
        if w is None:
            w = self.weights()[0]
        m = np.r_[np.asarray(lit_mask, bool), True]
        return float(w[m[self.h_lits].all(1)].sum())

    def describe(self, i: int, w: np.ndarray | None = None) -> dict:
        if w is None:
            w = self.weights()[0]
        lits = [self.lits[j] for j in self.h_lits[i] if j < len(self.lits)]
        return {"probe": self.probes[self.h_probe[i]], "literals": dict(lits), "p": float(w[i])}

    def top(self, k: int = 5, mask: np.ndarray | None = None) -> list[dict]:
        w, _ = self.weights()
        ww = w if mask is None else np.where(mask, w, -1.0)
        idx = np.argsort(-ww)[:k]
        return [self.describe(int(i), w) for i in idx if ww[i] >= 0]

    def map(self, mask: np.ndarray | None = None) -> dict | None:
        t = self.top(1, mask)
        return t[0] if t else None

    def snapshot(self, k: int = 3) -> dict:
        w, wu = self.weights()
        return {"top": self.top(k), "p_unknown": wu, "n_hyps": int(len(w)), "n_enumerated": self.n_enumerated,
                "n_literals": len(self.lits)}


# ---------------------------------------------------------------------------------------- action model
def related(a: Action, state: dict) -> set[str]:
    """Entities an action involves: its arguments, the part a held/inserted object sits in, and the
    objects that sit in any of those parts (what is inside a part is part of its state)."""
    ents = set(action_entities(a))
    for e in list(ents):
        w = state.get(f"obj:{e}.where", "")
        if w.startswith("in:"):
            ents.add(w[3:])
    inside = {f"in:{e}" for e in ents}
    ents |= {entity_of(f) for f, v in state.items() if f.endswith(".where") and v in inside}
    return ents


def context(a: Action, state: dict) -> frozenset:
    ents = related(a, state)
    return frozenset((f, v) for f, v in state.items() if (is_scene_feature(f) and entity_of(f) in ents)
                     or f == "body:hand")


class ActionModel:
    """Counts of the effects of (action, context) pairs; generalises to unseen contexts by overlap."""

    def __init__(self, reveal_prior: tuple[float, float] = (1.0, 3.0)):
        self.effects: dict[Action, dict[frozenset, Counter]] = defaultdict(dict)
        self.stalls: Counter = Counter()
        self.seen: dict[tuple, set[str]] = {}         # (action, ctx) -> related entities, for every offered pair
        self.tried: set[tuple] = set()
        self.known: set[Action] = set()               # every action ever offered
        self.offered: set[tuple] = set()              # (action, hand empty?) pairs it was offered with
        self.openings = 0
        self.reveals = 0
        self.reveal_prior = reveal_prior              # Beta(a, b) prior: opening something reveals things
        self.version = 0                              # grows whenever an (action, context) shows a new effect
        self.failed_runs: Counter = Counter()         # (action, ctx) -> skill failures in a row there
        self.max_failed = 3                           # a skill that failed this often here counts as tried

    def observe_available(self, actions: list[Action], state: dict) -> None:
        hand = state.get("body:hand", "empty")
        for a in actions:
            self.known.add(a)
            self.offered.add((a, hand == "empty"))
            key = (a, context(a, state))
            if key not in self.seen:
                self.seen[key] = related(a, state)

    def record(self, a: Action, before: dict, after: dict, executed: bool, stalled: bool, revealed: int) -> None:
        ctx = context(a, before)
        if not executed:
            # a failed skill tells nothing about the world, unless it keeps failing in this context (an
            # insert the body cannot do there, an object it cannot lift from where it lies): from the
            # max_failed-th failure in a row, what it did (usually nothing) counts as what it does here
            self.failed_runs[(a, ctx)] += 1
            if self.failed_runs[(a, ctx)] < self.max_failed:
                return
        else:
            self.failed_runs[(a, ctx)] = 0
        eff = frozenset((f, after[f]) for f in before if f in after and after[f] != before[f])
        cnt = self.effects[a].setdefault(ctx, Counter())
        self.version += eff not in cnt                # something new was learned
        cnt[eff] += 1
        self.tried.add((a, ctx))
        self.seen.setdefault((a, ctx), related(a, before))
        if stalled:
            self.stalls[(a, ctx)] += 1
        if any(f.endswith(".open") and v == "open" for f, v in eff):
            self.openings += 1
            self.reveals += revealed > 0

    def change_rate(self) -> float:
        """How often an executed, untried (action, context) changed some feature (Beta(1, 1) prior)."""
        n = len(self.tried)
        k = sum(1 for (a, ctx) in self.tried if any(len(e) for e in self.effects[a].get(ctx, {})))
        return (1 + k) / (2 + n)

    def action_change_rate(self, a: Action, prior: float, strength: float = 2.0) -> float:
        """P(a changes something in a context it was not tried in): its own record across the contexts
        it was tried in, shrunk towards `prior` (the rate over all actions). An insert that stalled in
        four contexts is not worth much in a fifth."""
        table = self.effects.get(a, {})
        n = len(table)
        k = sum(1 for cnt in table.values() if any(len(e) for e in cnt))
        return (k + strength * prior) / (n + strength)

    def entity_change_rate(self, e: str) -> float:
        """How often executed actions on entity e changed one of e's own features (Beta(1, 1) prior)."""
        n = k = 0
        for a, table in self.effects.items():
            if not a[1:] or a[1] != e:
                continue
            for cnt in table.values():
                for eff, m in cnt.items():
                    n += m
                    k += m * any(entity_of(f) == e for f, _ in eff)
        return (1 + k) / (2 + n)

    def reveal_rate(self) -> float:
        a, b = self.reveal_prior
        return (a + self.reveals) / (a + b + self.openings)

    def predict(self, a: Action, state: dict) -> tuple[dict | None, bool]:
        """(predicted changes {feature: value}, exact) or (None, False) if the action was never run."""
        table = self.effects.get(a)
        if not table:
            return None, False
        ctx = context(a, state)
        if ctx in table:
            eff, exact = table[ctx].most_common(1)[0][0], True
        else:
            best = max(table, key=lambda c: (len(c & ctx), sum(table[c].values())))
            eff, exact = table[best].most_common(1)[0][0], False
        return {f: v for f, v in eff if f in state}, exact

    def side_effect(self, a: Action, state: dict) -> bool | None:
        """Does a probe change its own part's features here? Known from the probe itself or, for a
        turn-while-moving probe, from turning the same part the same way alone; None if unknown."""
        own = lambda eff: any(entity_of(f) == a[1] for f in eff)   # noqa: E731
        for b in (a, ("turn", a[1], a[2]) if a[0] in ("turn_pull", "turn_push") else None):
            if b is None:
                continue
            ctx = context(b, state)
            if ctx in self.effects.get(b, {}):
                return own(dict(self.effects[b][ctx].most_common(1)[0][0]))
        return None

    def offered_with(self, a: Action, state: dict) -> bool:
        """Has this action been offered with the hand as it is in `state` (empty or not)?"""
        return (a, state.get("body:hand", "empty") == "empty") in self.offered

    def untested(self, a: Action, state: dict) -> bool:
        return (a, context(a, state)) not in self.tried

    def open_pairs(self) -> list[tuple]:
        return [k for k in self.seen if k not in self.tried]

    def exhausted(self) -> bool:
        return not self.open_pairs()

    def possible(self, lits: list[Literal], state: dict, achieved: set) -> np.ndarray:
        """Literals that are true now, have been seen true, or sit on an entity an untried pair touches."""
        open_ents = set().union(*[self.seen[k] for k in self.open_pairs()]) if self.seen else set()
        return np.array([state.get(f) == v or (f, v) in achieved or entity_of(f) in open_ents for f, v in lits], bool)


def apply(state: dict, changes: dict) -> dict:
    s = dict(state)
    s.update(changes)
    return s


# ---------------------------------------------------------------------------------------- scores
def _free_hand(state: dict) -> tuple[dict, bool]:
    """The state after letting go of whatever is held (an object held in the air lands 'free'; one
    inserted in a part stays there): probes need the hand."""
    h = state.get("body:hand", "empty")
    if h == "empty":
        return state, False
    s = dict(state)
    s["body:hand"] = "empty"
    if s.get(f"obj:{h}.where") == "held":
        s[f"obj:{h}.where"] = "free"
    return s, True


def candidate_changes(a: Action, state: dict, vocab: Vocabulary) -> list[dict]:
    """Single-feature changes an untried action might make: another value of an angle / opening of an
    entity it involves; an inserted object ends up in the target part, a released one lies free."""
    ents = related(a, state)
    out = []
    for f, v in state.items():
        if is_scene_feature(f) and entity_of(f) in ents and not f.endswith(".where"):
            out += [{f: u} for u in vocab.values.get(f, ()) if u != v]
    if a[0] == "insert":
        out.append({f"obj:{a[1]}.where": f"in:{a[2]}"})
    elif a[0] == "release" and state.get(f"obj:{a[1]}.where") != "free":
        out.append({f"obj:{a[1]}.where": "free"})
    return out


def hand_ok(a: Action, state: dict) -> bool:
    """Body knowledge: part skills need an empty hand; pick needs it empty (or holding that object, to
    pull it back out of a part); object skills need that object in hand (turn_held may also take hold
    of an object sitting in a part)."""
    h = state.get("body:hand", "empty")
    if a[0] == "pick":
        return h in ("empty", a[1])
    if a[0] in OBJECT_KINDS:
        where = state.get(f"obj:{a[1]}.where", "")
        if a[0] == "insert":
            return h == a[1] and where == "held"
        return h == a[1] or (a[0] == "turn_held" and h == "empty" and where.startswith("in:"))
    return h == "empty"


def score_actions(belief: Belief, model: ActionModel, state: dict, actions: list[Action], cost_fn,
                  kappa_reveal: float = 2.0, rho_bias: float = 0.0, gamma: float = 0.9,
                  base_novel: float = 0.05, release_cost: float = 1.5, depth: int = 1,
                  restore_cost: float = 8.0) -> list[dict]:
    """Rate (bits / s) of every available action: probe EIG + lookahead EIG + action-model + reveal value.

    Lookahead: the EIG of the best probe in the state after the action (after letting go of a held
    object if needed, at the cost of a release), per second of both actions. For an untried (action,
    context) the change is unknown: P(change) (learned) times the mean lookahead over the single-feature
    changes it could make. An action that changes what the hand holds (pick) is also worth what it
    enables: the best rate of the actions that become possible after it (one level deeper). A probe that
    also changes the state (turning a part that stays turned) spends the state the other probes still
    need: it is charged the time to restore it, times the chance it changes something, while other
    probes are still informative here; so is any other action that may change the state."""
    w, wu = belief.weights()
    pcost = np.array([cost_fn(p) for p in belief.probes])
    rho = min(0.95, model.reveal_rate() + rho_bias)
    p_change = model.change_rate()
    eig_cache: dict[frozenset, np.ndarray] = {}

    def look(st: dict, changes: dict, c: float) -> float:
        s2, rel = _free_hand(apply(st, changes))
        key = frozenset(s2.items())
        if key not in eig_cache:
            eig_cache[key] = belief.eig_probes(s2, w)
        if not len(pcost):
            return 0.0
        return float(np.max(eig_cache[key] / (c + rel * release_cost + pcost)))

    def rates(st: dict, acts: list[Action], d: int) -> list[dict]:
        eig_now = belief.eig_probes(st, w)
        total = float(eig_now.sum())
        out = []
        for a in acts:
            c = cost_fn(a)
            r_probe = 0.0
            if a in belief.pidx and st.get("body:hand", "empty") == "empty":
                e = eig_now[belief.pidx[a]]
                known = model.side_effect(a, st)
                if known is not None:
                    p_side = float(known)
                elif any(entity_of(f) == a[1] for f in st if is_scene_feature(f)):
                    p_side = model.entity_change_rate(a[1])
                else:
                    p_side = 0.0
                others = total - e > 0.02
                r_probe = e / (c + restore_cost * p_side * others)
            r_look = r_model = r_enable = 0.0
            pred, cands = None, []
            if model.untested(a, st):
                cands = candidate_changes(a, st, belief.vocab)
                if cands:
                    p_a = model.action_change_rate(a, p_change)
                    r_model = p_a * float(np.mean([look(st, ch, c) for ch in cands]))
                r_model += base_novel * wu / c
                if a[0] in PROBE_KINDS and any(st.get(f"part:{e}.open") == "closed" for e in action_entities(a)):
                    r_model += kappa_reveal * rho * wu / c
            else:
                pred, _ = model.predict(a, st)
                if pred:
                    r_look = gamma * look(st, pred, c)
            # pulling a held object back out of a part (a pick while holding it) moves it like any other step
            withdraw, k = a[0] == "pick" and st.get("body:hand", "empty") == a[1], 1.0
            if not r_probe and (a[0] != "pick" or withdraw):
                # a move away from a state whose probes are still informative spends that state too
                p_chg = (model.action_change_rate(a, p_change) if cands else 0.0) if pred is None and model.untested(a, st)                     else float(bool(pred))
                if p_chg and total > 0.02:
                    k = c / (c + restore_cost * p_chg)
                    r_look, r_model = k * r_look, k * r_model
            if d > 0 and a[0] == "pick" and pred != {}:     # (a pick known to change nothing enables nothing)
                s2 = apply(st, pred if pred is not None else {f"obj:{a[1]}.where": "held"})
                s2["body:hand"] = a[1]
                new = [b for b in model.known if hand_ok(b, s2) and not hand_ok(b, st) and b[0] != "release"]
                if new:
                    sub = max(rates(s2, new, d - 1), key=lambda r: r["rate"])
                    r_enable = k * gamma * sub["rate"] * sub["cost"] / (c + sub["cost"])
            out.append({"action": a, "rate": r_probe + r_look + r_model + r_enable, "probe": r_probe,
                        "look": r_look, "model": r_model, "enable": r_enable, "cost": c})
        return out

    return rates(state, actions, depth)
