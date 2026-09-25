"""DiscoveryAgent: find out how to reach a goal (open a door) by experiment, with curiosity only.

The agent is given generic skills (whatever actions the world offers), a self-model of what its skills
cost (time per skill, travel of the hand) and its one hand (part skills need it empty, object skills
need that object in it), and the goal as a feature condition. Nothing tells it what a knob, key or
keyhole is. Each step:

1. Exploit if a rule is probable: for the most probable hypotheses (probe, C) it plans, over the
   *learned* action model, a sequence that makes C true (A* over predicted states, restricted to the
   actions that involve C's entities). If the probability that the plan's final probe reaches the goal
   exceeds `exploit` (0.8), it executes the plan's next step. This groups equivalent hypotheses (a key
   turned vs. the plug it sits in turned), which a MAP threshold alone would not.
2. Otherwise explore: the action with the best information rate (belief.score_actions: probe EIG,
   one-step lookahead EIG, action-model and reveal value, per second of effort).
3. Update: the action model from every executed outcome, the belief from every probe, the vocabulary
   from what came into view.
4. No solution: once the action model has no open (action, context) pair left (every container opened,
   every object tried in every part both ways, ...), it sweeps: it plans to make true, and tests, the
   condition C of the most probable rule that is consistent with every outcome (h_err = 0) and whose
   literals the learned model can reach, one C at a time. It stops and says so only when no such rule
   is left: all the remaining mass (UNKNOWN included) is on rules that failed or that the learned
   model cannot make true (the planner finds no way, or its plans keep going wrong). The mass of the
   untested-but-reachable rules is never compared with UNKNOWN, which explains every failure and so
   would swamp a rule that simply was not tested yet.

Locality. Four built-in "causes act locally" assumptions, each of which a Locality switches off for
an ablation (the defaults are the ones every reported number was measured with):

- radius       the probes (the moves that may reach the goal) are the probe actions on parts within
               0.7 m of the goal part; None = a probe of any part in view (the goal part stays given)
- probe_prior  a prior penalty on probes by the probe part's distance from the goal part (0 = off)
- lit_prior    the same penalty on the literals of the conditions C (0 = off); conditions on any part
               are always in the hypothesis space, the penalty only makes far ones less probable
- explain      'near': the explanation keeps only conditions on entities near the goal part;
               'evidence': only those the agent's own log or belief supports (see pai.discovery.explain)

Locality.off() switches all four off. The switches in use are recorded in the log (log.locality), so
explain(log) applies the matching filter.

Baselines (same skills, same budget accounting): RandomAgent (uniform over the offered actions, probing
the door after each) and NoveltyAgent (prefers never-tried actions and entities, no causal belief).
compare() runs them side by side and prints a table.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from pai.discovery.belief import (OBJECT_KINDS, ActionModel, Belief, action_entities, apply, entity_of, hand_ok,
                                  is_scene_feature, score_actions)
from pai.discovery.interface import PROBE_KINDS, Action, Entity, World

# self-model of the body: seconds per skill and per metre of hand travel (not read from any world)
SKILL_TIME = {"pick": 3.0, "release": 1.5, "insert": 4.0, "turn_held": 2.0, "turn": 2.0, "pull": 3.0, "push": 3.0,
              "turn_pull": 4.0, "turn_push": 4.0, "press_push": 3.5}
HAND_SPEED = 1.2   # s per metre


@dataclass(frozen=True)
class Locality:
    """The built-in locality assumptions (see the module docstring); the defaults are the reported ones."""
    radius: float | None = 0.7     # probes on parts within this distance (m) of the goal part; None = any part
    probe_prior: float = 1.0       # weight of the distance penalty in the prior over probes (0 = off)
    lit_prior: float = 1.0         # weight of the distance penalty in the prior over literals (0 = off)
    explain: str = "near"          # explanation filter: "near" (the goal part) | "evidence" (log and belief)

    @classmethod
    def off(cls) -> "Locality":
        return cls(radius=None, probe_prior=0.0, lit_prior=0.0, explain="evidence")

    def without(self, *names: str) -> "Locality":
        """This locality with the named assumptions switched off, e.g. without("radius", "lit_prior")."""
        off = asdict(Locality.off())
        return replace(self, **{n: off[n] for n in names})


@dataclass
class EpisodeLog:
    actions: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    states: list = field(default_factory=list)        # agent state before each action
    snapshots: list = field(default_factory=list)     # belief snapshots (every few steps and at the end)
    modes: list = field(default_factory=list)         # "explore" | "exploit" | "random" | "novelty" | "probe"
    final_map: dict | None = None
    reached_goal: bool = False
    n_actions: int = 0
    time_cost: float = 0.0
    stopped_reason: str = ""
    goal: dict = field(default_factory=dict)
    entities: dict = field(default_factory=dict)      # id -> Entity (all ever seen)
    revealed_by: dict = field(default_factory=dict)   # entity id -> the action whose outcome revealed it
    no_solution_mass: float | None = None
    wall_time: float = 0.0
    locality: dict = field(default_factory=dict)     # the Locality switches the agent ran with
    consistent: list = field(default_factory=list)   # at a success: the rules with the successful probe that
    #                                                  held and got nothing wrong, [{"literals", "p"}], p
    #                                                  normalised over them (the belief's attribution)




def _pos(ents: dict, eid: str) -> np.ndarray:
    return np.asarray(ents[eid].pos, float) if eid in ents else np.zeros(3)


def target_of(a: Action) -> str:
    return a[2] if a[0] == "insert" else a[1]


def near_goal_probes(actions: list[Action], ents: dict, goal_part: str, radius: float | None) -> list[Action]:
    """Probe actions on parts within `radius` of the goal part (radius None: on any part in view)."""
    g = _pos(ents, goal_part)
    return [a for a in actions if a[0] in PROBE_KINDS and a[1] in ents
            and (radius is None or np.linalg.norm(_pos(ents, a[1]) - g) <= radius)]


class _Episode:
    """Bookkeeping shared by the agent and the baselines."""

    def __init__(self, world: World):
        self.world = world
        self.ents: dict[str, Entity] = {e.id: e for e in world.entities()}
        self.hand = "empty"
        self.state = self._observe({})
        self.hand_pos = np.array([0.6, 0.0, 1.0])
        self.log = EpisodeLog(goal=world.goal(), entities=dict(self.ents))
        (gf, _), = world.goal().items()
        self.goal_part = entity_of(gf)
        self.t0 = time.perf_counter()

    def _observe(self, state: dict, a: Action | None = None, out=None) -> dict:
        """Merge the world's features into the state; the hand ('body:hand') is inferred from my own
        actions when the world does not report it."""
        feats = self.world.features()
        # merged, so a feature a world reports only now and then is kept; but an object that went out of
        # view (a drawer pushed shut over it) takes its features with it
        gone = {e for e in (entity_of(f) for f in state if f.startswith("obj:"))
                if not any(f.startswith(f"obj:{e}.") for f in feats)}
        s = {f: v for f, v in state.items() if not (f.startswith("obj:") and entity_of(f) in gone)}
        s.update(feats)
        if "body:hand" not in feats:
            held = [entity_of(f) for f, v in feats.items() if f.endswith(".where") and v == "held"]
            if held:
                self.hand = held[0]
            elif a is not None and out is not None and out.executed and not out.stalled:
                if a[0] in ("pick", "insert", "turn_held"):
                    self.hand = a[1]
                elif a[0] == "release":
                    self.hand = "empty"
            s["body:hand"] = self.hand
        return s

    def cost(self, a: Action, frm: np.ndarray | None = None) -> float:
        frm = self.hand_pos if frm is None else frm
        return SKILL_TIME.get(a[0], 3.0) + HAND_SPEED * float(np.linalg.norm(_pos(self.ents, target_of(a)) - frm))

    def step(self, a: Action, mode: str):
        before = self.state
        out = self.world.execute(a)
        for e in out.revealed:
            self.ents[e.id] = e
            self.log.entities[e.id] = e
            self.log.revealed_by.setdefault(e.id, a)
        for e in self.world.entities():          # objects move (released elsewhere, inserted)
            self.ents[e.id] = e
            self.log.entities[e.id] = e
        self.state = self._observe(before, a, out)
        self.hand_pos = _pos(self.ents, target_of(a))
        lg = self.log
        lg.actions.append(a)
        lg.outcomes.append(out)
        lg.states.append(before)
        lg.modes.append(mode)
        lg.n_actions += 1
        lg.time_cost += out.cost
        return before, out

    def finish(self, reason: str) -> EpisodeLog:
        self.log.stopped_reason = reason
        self.log.reached_goal = bool(self.world.goal_reached())
        self.log.wall_time = time.perf_counter() - self.t0
        return self.log


# ---------------------------------------------------------------------------------------- the agent
class DiscoveryAgent:
    def __init__(self, max_lits: int = 2, eps: float = 0.005, p_unknown: float = 0.1, exploit: float = 0.8,
                 no_solution: float = 0.05, radius: float = 0.7, proximity: float = 0.5, seed: int = 0,
                 snapshot_every: int = 10, n_plan: int = 6, lit_locality: float = 1.0, sweep: int | None = None,
                 kappa_reveal: float = 0.3, base_novel: float = 0.01, init_bonus: float = 0.0,
                 extent: bool = True, locality: Locality | None = None):
        self.max_lits, self.eps, self.p_unknown = max_lits, eps, p_unknown
        # no_solution: kept for compatibility; giving up no longer thresholds a mass (see the docstring)
        self.exploit_p, self.no_solution_p = exploit, no_solution
        # radius and lit_locality are the older spelling of Locality(radius=..., lit_prior=...); a
        # `locality` given overrides them
        self.locality = loc = locality or Locality(radius=radius, lit_prior=lit_locality)
        self.radius, self.proximity = loc.radius, proximity
        self.rng = np.random.default_rng(seed)
        self.snapshot_every, self.n_plan, self.lit_locality = snapshot_every, n_plan, loc.lit_prior
        self.sweep, self.kappa_reveal, self.base_novel, self.init_bonus = sweep, kappa_reveal, base_novel, init_bonus
        self.extent = extent      # locality measured from the goal part's extent (else from its centre)
        self.belief: Belief | None = None
        self.model: ActionModel | None = None

    # ------------------------------------------------------------------ main loop
    def run(self, world: World, budget_actions: int = 300, recipes=None) -> EpisodeLog:
        ep = _Episode(world)
        ep.log.locality = asdict(self.locality)
        ents, gp = ep.ents, ep.goal_part
        probes = near_goal_probes(world.actions(), ents, gp, self.radius)
        g = _pos(ents, gp)
        g_r = float(ents[gp].attrs.get("size", 0.0)) / 2 if gp in ents and self.extent else 0.0

        def gap(e: str) -> float:
            """How far e is from the goal part itself (its extent, not its centre): everything mounted on
            a door leaf is equally local to it."""
            return max(0.0, float(np.linalg.norm(_pos(ents, e) - g)) - g_r)

        w_probe = self.locality.probe_prior
        prox = lambda ps: np.array([-w_probe * gap(p[1]) / self.proximity for p in ps])  # noqa: E731

        first = dict(ep.state)   # how things were when first seen: found ready for use, until I disturbed them

        def lit_prox(lits):
            out = []
            for f, v in lits:
                e = v[3:] if v.startswith("in:") else entity_of(f)
                out.append(-self.lit_locality * gap(e) / self.proximity
                           + self.init_bonus * (first.get(f) == v))
            return np.array(out)

        probe_prior, lit_prior, rho_bias = prox, lit_prox, 0.0
        if recipes is not None and len(recipes):
            pb, lb, rho_bias = recipes.priors(ents, gp)
            probe_prior = lambda ps: prox(ps) + pb(ps)      # noqa: E731
            lit_prior = lambda ls: lit_prox(ls) + lb(ls)    # noqa: E731
        self.belief = bel = Belief(world.goal(), probes, self.max_lits, self.eps, self.p_unknown,
                                   probe_prior, lit_prior)
        self.model = model = ActionModel()
        bel.update_vocab(ep.state, [e.id for e in ents.values() if e.kind == "part"])
        self._plans: dict = {}
        self._commit: dict | None = None
        self._failed: dict = {}          # C -> (model version when its plan last failed, failures)
        reason = "budget"
        sweeps = self.sweep if self.sweep is not None else budget_actions
        while ep.log.n_actions < budget_actions:
            if world.goal_reached():
                reason = "goal"
                break
            acts = world.actions()
            model.observe_available(acts, ep.state)
            if ep.log.n_actions % self.snapshot_every == 0:
                ep.log.snapshots.append({"n": ep.log.n_actions, **bel.snapshot()})
            a, mode = None, ""
            if model.exhausted():
                # nothing left to learn about what my actions do: test the consistent rules I can still
                # make true, most probable first; give up when none is left
                poss = model.possible(bel.lits, ep.state, bel.achieved)
                if sweeps > 0:
                    a = self._sweep(ep, acts, poss)
                    mode = "sweep" if a is not None else ""
                if a is None:
                    ep.log.no_solution_mass = bel.mass(poss)    # consistent or not, reachable literals
                    reason = "no_solution"
                    break
            if a is None:
                a, mode = self._exploit(ep, acts)
            if a is None:
                scores = score_actions(bel, model, ep.state, acts, ep.cost, kappa_reveal=self.kappa_reveal,
                                       rho_bias=rho_bias, base_novel=self.base_novel)
                best = max(s["rate"] for s in scores)
                cands = [s["action"] for s in scores if s["rate"] >= best - 1e-12]
                a, mode = cands[int(self.rng.integers(len(cands)))], "explore"
            before, out = ep.step(a, mode)
            model.record(a, before, ep.state, out.executed, out.stalled, len(out.revealed))
            sweeps -= mode == "sweep"
            if mode == "sweep" and self._commit and out.executed:
                pred = self._commit.pop("expect", None)
                if pred is not None and any(ep.state.get(f) != v for f, v in pred.items()):
                    self._fail(self._commit["C"])           # the model was wrong: try another rule
                    self._commit = None
            if out.executed and a in bel.pidx:
                bel.observe(a, before, world.goal_reached())
            for f, v in ep.state.items():
                first.setdefault(f, v)
            bel.update_vocab(ep.state, [e.id for e in ents.values() if e.kind == "part"])
        if world.goal_reached():
            reason = "goal"
        ep.log.snapshots.append({"n": ep.log.n_actions, **bel.snapshot()})
        ep.log.final_map = self._final_map(ep)
        return ep.finish(reason)

    def _final_map(self, ep: _Episode) -> dict | None:
        bel = self.belief
        if ep.world.goal_reached() and ep.log.actions and ep.log.actions[-1] in bel.pidx:
            a, s = ep.log.actions[-1], ep.log.states[-1]
            mask = (bel.h_probe == bel.pidx[a]) & bel.holds({f: v for f, v in s.items() if is_scene_feature(f)})
            if mask.any():   # the simplest rule that got nothing wrong, if there is one
                mask &= bel.h_err == bel.h_err[mask].min()
                ep.log.consistent = self._attribution(mask)
                # among equally probable ones, the rule whose conditions became true last: the probes
                # before that change failed
                w, _ = bel.weights()
                tie = mask & (w >= w[mask].max() * (1 - 1e-6))
                if tie.sum() > 1:
                    def since(i):
                        lits = [bel.lits[j] for j in bel.h_lits[i] if j < len(bel.lits)]
                        return max((max((k for k, st in enumerate(ep.log.states) if st.get(f) != v), default=-1)
                                    for f, v in lits), default=-1)
                    best = max(np.flatnonzero(tie), key=since)
                    mask = np.zeros_like(mask)
                    mask[best] = True
            return bel.map(mask)
        return bel.map()

    def _attribution(self, mask: np.ndarray, k: int = 300) -> list[dict]:
        """The k most probable rules in `mask`, with their posterior renormalised over the mask."""
        bel = self.belief
        w, _ = bel.weights()
        idx = np.flatnonzero(mask)
        z = float(w[idx].sum())
        idx = idx[np.argsort(-w[idx])][:k]
        return [{"literals": bel.describe(int(i), w)["literals"], "p": float(w[i]) / z if z > 0 else 0.0}
                for i in idx]

    def _sweep(self, ep: _Episode, acts: list[Action], poss: np.ndarray) -> Action | None:
        """Next step towards testing the most probable condition C (summed over probes) among the rules
        that are consistent with everything seen and whose literals can all be made true. The agent
        commits to one C at a time: it follows its plan (dropping C if an outcome is not what the model
        predicted), then tries the consistent probes while C holds."""
        bel = self.belief
        ok = np.r_[poss, True][bel.h_lits].all(1) & (bel.h_err == 0)
        if not ok.any():
            return None
        w, _ = bel.weights()
        idx = np.flatnonzero(ok)
        idx = idx[np.argsort(-w[idx])][:400]
        by_c: dict[tuple, list] = {}
        for i in idx:
            h = bel.describe(int(i), w)
            by_c.setdefault(tuple(sorted(h["literals"].items())), []).append(h)
        order = sorted(by_c, key=lambda c: -sum(h["p"] for h in by_c[c]))
        if self._commit is not None and self._commit["C"] not in by_c:
            self._commit = None                            # tested: every probe for it failed
        dropped: set = set()                               # given up on in this call
        for _ in range(len(order) + 1):
            if self._commit is None:
                for C in order:
                    if C in dropped:
                        continue
                    if all(ep.state.get(f) == v for f, v in C):     # already true: test it here
                        self._commit = {"C": C, "plan": []}
                        break
                    if self._unreachable(C):
                        continue
                    plan = self._plan(ep, dict(C))
                    if plan:
                        self._commit = {"C": C, "plan": list(plan)}
                        break
                    self._fail(C)
            if self._commit is None:
                return None
            C = self._commit["C"]
            a = self._sweep_step(ep, acts, by_c)
            if a is not None:
                return a
            dropped.add(C)
        return None

    def _sweep_step(self, ep: _Episode, acts: list[Action], by_c: dict) -> Action | None:
        C, plan = self._commit["C"], self._commit["plan"]
        if all(ep.state.get(f) == v for f, v in C):
            # probes that leave the state as it is first (a probe that turns a part undoes C)
            side = lambda p: bool(self.model.predict(p, ep.state)[0])   # noqa: E731
            for h in sorted(by_c[C], key=lambda h: side(h["probe"])):
                if h["probe"] in acts:
                    return h["probe"]
            rel = [a for a in acts if a[0] == "release"]
            if rel:
                return rel[0]
        elif plan and plan[0] in acts:
            a = plan.pop(0)
            self._commit["expect"] = self.model.predict(a, ep.state)[0] or {}
            return a
        self._fail(C)
        self._commit = None
        return None

    def _fail(self, C: tuple) -> None:
        n, k = self._failed.get(C, (0, 0))
        self._failed[C] = (self.model.version, k + 1)

    def _unreachable(self, C: tuple, retries: int = 3) -> bool:
        """C cannot be made true under the learned model: its plan failed and the model has learned
        nothing since, or its plans went wrong `retries` times."""
        if C not in self._failed:
            return False
        n, k = self._failed[C]
        return k >= retries or n == self.model.version

    # ------------------------------------------------------------------ exploitation
    def _exploit(self, ep: _Episode, acts: list[Action]) -> tuple[Action | None, str]:
        bel = self.belief
        w, _ = bel.weights()
        state = ep.state
        ps_now = bel.success_probs(state, w)
        best, best_plan = -1.0, None
        for i, p in enumerate(bel.probes):
            if ps_now[i] > best and p in acts:
                best, best_plan = ps_now[i], [p]
        seen = set()
        for h in bel.top(self.n_plan * 3):
            C = tuple(sorted(h["literals"].items()))
            if C in seen or all(state.get(f) == v for f, v in C):
                continue
            seen.add(C)
            if len(seen) > self.n_plan:
                break
            plan = self._plan(ep, dict(C))
            if not plan:
                continue
            s_end = state
            for a in plan:
                s_end = apply(s_end, self.model.predict(a, s_end)[0] or {})
            if not all(s_end.get(f) == v for f, v in C):
                continue
            ps = bel.success_probs(s_end, w)
            j = int(np.argmax(ps))
            if ps[j] > best:
                best, best_plan = float(ps[j]), plan + [bel.probes[j]]
        if best >= self.exploit_p and best_plan and best_plan[0] in acts:
            return best_plan[0], "exploit"
        return None, ""

    def _plan(self, ep: _Episode, C: dict, max_nodes: int = 4000) -> list[Action] | None:
        """A* over the learned model from the current state to a state satisfying C."""
        state, model = ep.state, self.model
        key0 = (tuple(sorted(C.items())), frozenset(state.items()), len(model.tried))
        if key0 in self._plans:
            return self._plans[key0]
        E = {entity_of(f) for f in C} | {v[3:] for v in C.values() if v.startswith("in:")}
        objs_in = {entity_of(f) for (f, v) in self.belief.achieved if f.endswith(".where") and v[3:] in E}
        rel = [a for a in model.known if set(action_entities(a)) & E or (a[0] in OBJECT_KINDS + ("pick",)
                                                                          and a[1] in objs_in)]
        objs = {a[1] for a in rel if a[0] in OBJECT_KINDS + ("pick",)}
        rel += [a for a in model.known if a[0] in ("pick", "release") and a[1] in objs]
        hand = state.get("body:hand", "empty")
        if hand != "empty":
            rel += [a for a in model.known if a[0] == "release" and a[1] == hand]
        rel = list(dict.fromkeys(rel))

        def hcount(s):
            return sum(s.get(f) != v for f, v in C.items())

        cnt = itertools.count()
        start = frozenset(state.items())
        frontier = [(hcount(state), next(cnt), 0.0, start, [])]
        best_g = {start: 0.0}
        result = None
        n = 0
        while frontier and n < max_nodes:
            _, _, gcost, sk, path = heapq.heappop(frontier)
            s = dict(sk)
            if hcount(s) == 0:
                result = path
                break
            n += 1
            if len(path) >= 8:
                continue
            for a in rel:
                if not hand_ok(a, s) or not model.offered_with(a, s):
                    continue
                eff, _ = model.predict(a, s)
                if not eff:
                    continue
                s2 = apply(s, eff)
                k2 = frozenset(s2.items())
                g2 = gcost + SKILL_TIME.get(a[0], 3.0)
                if g2 < best_g.get(k2, 1e18):
                    best_g[k2] = g2
                    heapq.heappush(frontier, (g2 / 3.0 + hcount(s2), next(cnt), g2, k2, path + [a]))
        self._plans[key0] = result
        return result


# ---------------------------------------------------------------------------------------- baselines
class RandomAgent:
    """Uniform over the offered actions; after each non-probe action, one random probe of the door."""

    def __init__(self, seed: int = 0, radius: float = 0.7):
        self.rng = np.random.default_rng(seed)
        self.radius = radius

    def run(self, world: World, budget_actions: int = 1000, recipes=None) -> EpisodeLog:
        ep = _Episode(world)
        probes = set(near_goal_probes(world.actions(), ep.ents, ep.goal_part, self.radius))
        while ep.log.n_actions < budget_actions and not world.goal_reached():
            acts = world.actions()
            a = acts[int(self.rng.integers(len(acts)))]
            ep.step(a, "random")
            if a in probes or world.goal_reached() or ep.log.n_actions >= budget_actions:
                continue
            avail = [p for p in world.actions() if p in probes]
            if avail:
                ep.step(avail[int(self.rng.integers(len(avail)))], "probe")
        return ep.finish("goal" if world.goal_reached() else "budget")


class NoveltyAgent:
    """Curiosity without a causal belief: least-tried action (and entity) first, a door probe after each."""

    def __init__(self, seed: int = 0, radius: float = 0.7, entity_weight: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.radius, self.entity_weight = radius, entity_weight

    def run(self, world: World, budget_actions: int = 1000, recipes=None) -> EpisodeLog:
        ep = _Episode(world)
        probes = set(near_goal_probes(world.actions(), ep.ents, ep.goal_part, self.radius))
        n_act: dict = {}
        n_ent: dict = {}

        def pick(acts):
            sc = [1.0 / (1 + n_act.get(a, 0)) + self.entity_weight * sum(1.0 / (1 + n_ent.get(e, 0))
                                                                          for e in action_entities(a))
                  + 1e-6 * self.rng.random() for a in acts]
            return acts[int(np.argmax(sc))]

        def do(a, mode):
            n_act[a] = n_act.get(a, 0) + 1
            for e in action_entities(a):
                n_ent[e] = n_ent.get(e, 0) + 1
            ep.step(a, mode)

        while ep.log.n_actions < budget_actions and not world.goal_reached():
            acts = world.actions()
            a = pick([x for x in acts if x not in probes] or acts)
            do(a, "novelty")
            if world.goal_reached() or ep.log.n_actions >= budget_actions:
                continue
            avail = [p for p in world.actions() if p in probes]
            if avail:
                do(pick(avail), "probe")
        return ep.finish("goal" if world.goal_reached() else "budget")


# ---------------------------------------------------------------------------------------- evaluation
AGENTS = {"discovery": DiscoveryAgent, "novelty": NoveltyAgent, "random": RandomAgent}


def _run_one(job: tuple) -> EpisodeLog:
    from pai.discovery.mock import MockWorld
    name, scenario, budget, seed, kwargs, recipes = job
    return AGENTS[name](seed=seed, **kwargs).run(MockWorld(scenario), budget, recipes)


def evaluate(name: str, scenarios: list, budget: int, recipes=None, processes: int = 1, **kwargs) -> list[EpisodeLog]:
    """Run agent `name` (a key of AGENTS) on each scenario (seed = its index); in parallel if asked."""
    jobs = [(name, sc, budget, i, kwargs, recipes) for i, sc in enumerate(scenarios)]
    if processes > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(processes) as pool:
            return pool.map(_run_one, jobs)
    return [_run_one(j) for j in jobs]


def summarize(logs: list[EpisodeLog], budget: int) -> dict:
    """Success rate, mean actions to success (failures counted at the budget), mean time."""
    n = len(logs)
    capped = [lg.n_actions if lg.reached_goal else budget for lg in logs]
    return {"n": n, "success": sum(lg.reached_goal for lg in logs) / max(n, 1),
            "actions": float(np.mean(capped)) if n else 0.0,
            "actions_median": float(np.median(capped)) if n else 0.0,
            "time": float(np.mean([lg.time_cost for lg in logs])) if n else 0.0,
            "wall": float(np.mean([lg.wall_time for lg in logs])) if n else 0.0}


def compare(n_seeds: int = 50, lock_state: str | None = None, split: str = "train", budget: int = 1000,
            seed: int = 0, agents: tuple = ("discovery", "novelty", "random"), processes: int = 1,
            solvable_only: bool = True, verbose: bool = True) -> dict:
    """Run the discovery agent and the baselines on the same scenarios; print and return a table."""
    from pai.discovery.mock import sample_scenario
    rng = np.random.default_rng(seed)
    scen = []
    while len(scen) < n_seeds:
        sc = sample_scenario(rng, split, lock_state=lock_state)
        if sc.solvable or not solvable_only:
            scen.append(sc)
    rows = {name: summarize(evaluate(name, scen, budget, processes=processes), budget) for name in agents}
    if verbose:
        print(format_table(rows, f"lock={lock_state or 'any'} split={split} seeds={n_seeds} budget={budget}"))
    return rows


def format_table(rows: dict, title: str = "") -> str:
    lines = [title, f"{'agent':<14}{'success':>9}{'actions':>10}{'median':>9}{'time[s]':>10}{'wall[s]':>9}"]
    for name, r in rows.items():
        lines.append(f"{name:<14}{r['success']:>9.2f}{r['actions']:>10.1f}{r['actions_median']:>9.1f}"
                     f"{r['time']:>10.1f}{r['wall']:>9.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    for st in (None, "latch", "deadbolt", "key", "both"):
        compare(n, lock_state=st, processes=8)
