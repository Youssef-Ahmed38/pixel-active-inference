"""Explaining a discovery episode: what held the door, and what released it, from the agent's own view.

The explanation is built from what the agent knows, never from the world's ground truth:

- the rule it ended on (the final MAP hypothesis: the simplest (probe, C) that got nothing wrong and
  held when the goal was reached), plus the conditions it had itself changed on the parts near the goal
  and left changed when it succeeded (a thumb-turn it turned, a key it left turned in a part). Each is
  kept only with a contrast in the log: the same probe failed while that condition was false (for a
  step outside the rule, while the rule's conditions held). A dial it happened to leave turned, with no
  failure to contrast, is not claimed. Among rules the evidence does not tell apart, the one whose
  conditions became true last is named, and the text says which other parts it cannot rule out.
- the log: which object was in the part it turned, where that object came into view (revealed by
  opening something), which other objects went into that part but would not turn (wrong keys) or did not
  go in at all, and whether a plain push/pull failed where turning while pulling worked (the latch).

explain(log) returns the text and a machine-checkable dict of claims; score(claims, truth, log) compares
them with MockWorld.truth() (explanation accuracy). truth() is used only there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pai.discovery.belief import entity_of, is_scene_feature

RELEASED = ("turn_pull", "turn_push", "press_push")
FAMILY = {"pull": "pull", "turn_pull": "pull", "push": "push", "turn_push": "push", "press_push": "push"}


@dataclass
class Explanation:
    text: str
    claims: dict


def _near(log, eid: str, radius: float) -> bool:
    ents = log.entities
    (gf, _), = log.goal.items()
    g = entity_of(gf)
    if eid not in ents or g not in ents:
        return False
    return float(np.linalg.norm(np.subtract(ents[eid].pos, ents[g].pos))) <= radius


def _shape(log, eid: str) -> str:
    e = log.entities.get(eid)
    return e.attrs.get("shape", "") if e is not None else ""


def _first_seen(log) -> dict:
    """Each feature's value when it first came into view (objects revealed later start where found)."""
    first: dict = {}
    for s in log.states:
        for f, v in s.items():
            first.setdefault(f, v)
    return first


def _objects_tried(log, part: str) -> tuple[list, list, list]:
    """Objects that went into `part` and turned it, went in but stalled when turned, or would not go in."""
    turned, stuck, no_fit = [], [], []
    for a, o, s in zip(log.actions, log.outcomes, log.states):
        if not o.executed:
            continue
        if a[0] == "insert" and a[2] == part and o.stalled and a[1] not in no_fit:
            no_fit.append(a[1])
        if a[0] == "turn_held" and s.get(f"obj:{a[1]}.where") == f"in:{part}":
            lst = stuck if o.stalled else turned
            if a[1] not in lst:
                lst.append(a[1])
    stuck = [x for x in stuck if x not in turned]
    no_fit = [x for x in no_fit if x not in turned and x not in stuck and
              not any(s.get(f"obj:{x}.where") == f"in:{part}" for s in log.states)]
    return turned, stuck, no_fit


def claims_of(log, radius: float = 0.7) -> dict:
    c = {"opened": bool(log.reached_goal), "no_solution": log.stopped_reason == "no_solution",
         "latched": None, "latch_tested": False, "probe": None, "release_part": None, "deadbolt_part": None,
         "key_part": None, "key_used": None, "key_found_in": None, "wrong_keys": [], "no_fit": [], "turned": [],
         "conditions": {}, "probably": {}, "stuck_part": None, "bolt_alternatives": []}
    if log.reached_goal and log.actions:
        p, s = log.actions[-1], log.states[-1]
        c["probe"] = p
        c["release_part"] = p[1]
        first = _first_seen(log)
        # conditions that already held at the start are not something I had to do
        # and causes act locally: a condition on something far from the door (the drawer I opened to
        # find the key) is how I got there, not what holds the door
        conds = {f: v for f, v in (log.final_map or {}).get("literals", {}).items()
                 if first.get(f) != v and _near(log, v[3:] if v.startswith("in:") else entity_of(f), radius)}
        probably = {}
        for f, v in s.items():
            if not is_scene_feature(f) or f in conds or first.get(f) == v:
                continue
            e = entity_of(f)
            where = v[3:] if v.startswith("in:") else None
            if _near(log, where or e, radius) and (f.endswith(".angle") or where) and v != "rest":
                probably[f] = v
            elif _near(log, e, radius) and f.endswith(".angle") and v == "rest":
                probably[f] = v            # turned back to rest from how it was at the start
        # keep a condition only with a contrast in the log: the same probe failed while it was false; a
        # step I took and left in place without such a contrast is not claimed
        holder_of = lambda f, v: (v[3:] if v.startswith("in:") else s.get(f"obj:{entity_of(f)}.where", "")[3:]  # noqa: E731
                                  or entity_of(f)) if f.startswith("obj:") else entity_of(f)
        fails = [st for a, o, st in zip(log.actions, log.outcomes, log.states) if a == p and o.executed]

        def contrast(f, v, given: dict):
            """A failure of the probe while f != v and the conditions `given` (on other parts) held."""
            grp = holder_of(f, v)
            return any(st.get(f) != v and all(st.get(g) == u for g, u in given.items() if holder_of(g, u) != grp)
                       for st in fails)
        conds = {f: v for f, v in conds.items() if contrast(f, v, {})}
        # a step outside the rule: only if the probe failed without it while the rule's conditions held
        probably = {f: v for f, v in probably.items() if f not in conds and contrast(f, v, conds)}
        c["conditions"], c["probably"] = conds, probably
        allc = {**probably, **conds}
        # which object sits in which part at the success
        holder = {s[f][3:]: entity_of(f) for f in s if f.endswith(".where") and str(s[f]).startswith("in:")}
        # the most recent change first: among candidates the evidence does not tell apart, the step
        # after which the probe worked is the likelier cause
        last = {f: max((i for i, st in enumerate(log.states) if st.get(f) != v), default=-1) for f, v in allc.items()}
        bolt = sorted((f for f in allc if f.startswith("part:") and f.endswith(".angle")),
                      key=lambda f: (f not in conds, -last[f]))
        for f, v in allc.items():
            e = entity_of(f)
            if f.startswith("obj:"):
                part = s.get(f"obj:{e}.where", "")[3:] or (v[3:] if v.startswith("in:") else None)
                if part:
                    c["key_part"], c["key_used"] = part, e
            elif f.endswith(".angle") and e in holder:
                c["key_part"], c["key_used"] = e, holder[e]
        bolts = [entity_of(f) for f in bolt if entity_of(f) not in holder and entity_of(f) not in (p[1], c["key_part"])]
        c["deadbolt_part"], c["bolt_alternatives"] = (bolts[0], bolts[1:]) if bolts else (None, [])
        # the latch: did a plain move of the same part fail while the conditions held?
        if p[0] in RELEASED:
            plain = (FAMILY[p[0]], p[1])
            tested = any(a == plain and o.executed and all(st.get(f) == v for f, v in allc.items())
                         for a, o, st in zip(log.actions, log.outcomes, log.states))
            c["latched"], c["latch_tested"] = True, tested
        else:
            c["latched"], c["latch_tested"] = False, True
    if c["key_part"] is None and not log.reached_goal:
        # no success: the part objects went into (if any) is the candidate lock
        parts = [s.get(f)[3:] for s in log.states for f in s if f.endswith(".where") and str(s.get(f)).startswith("in:")]
        if parts:
            c["key_part"] = max(set(parts), key=parts.count)
        # and the jointed part near the goal that none of my actions moved (every one stalled); among
        # several, the one the best remaining rules are about
        (gf, _), = log.goal.items()
        moved = {a[2] if a[0] == "insert" else a[1] for a, o in zip(log.actions, log.outcomes) if o.executed and not o.stalled}
        stuck = [e for e, x in log.entities.items() if x.kind == "part" and x.joint != "none" and e != entity_of(gf)
                 and e not in moved and _near(log, e, radius)]
        top = [entity_of(f) for h in (log.snapshots[-1]["top"] if log.snapshots else []) for f in h["literals"]]
        stuck.sort(key=lambda e: top.index(e) if e in top else len(top))
        c["stuck_part"] = stuck[0] if stuck else None
    if c["key_part"] is not None:
        turned, stuck, no_fit = _objects_tried(log, c["key_part"])
        c["turned"] = turned
        c["wrong_keys"] = [x for x in stuck if x != c["key_used"]]
        c["no_fit"] = [x for x in no_fit if x != c["key_used"]]
        k = c["key_used"]
        if k is not None and k in log.revealed_by:
            a = log.revealed_by[k]
            c["key_found_in"] = a[2] if a[0] == "insert" else a[1]
    return c


def text_of(c: dict, log) -> str:
    sh = lambda e: f"the {_shape(log, e) + ' ' if _shape(log, e) else ''}part {e}"   # noqa: E731
    if c["no_solution"]:
        out = [f"I could not open it and stopped after {log.n_actions} actions: I opened every container I found, "
               f"tried every object I found in every part, and tested every rule I could still bring about."]
        part = sh(c["key_part"]) if c["key_part"] else None
        ls = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1] if len(xs) > 1 else xs[0]   # noqa: E731
        if part and c["turned"]:
            out.append(f"{'Objects' if len(c['turned']) > 1 else 'Object'} {ls(c['turned'])} went into {part} and "
                       f"turned it, but the door still held whichever way it was turned.")
        if part and c["wrong_keys"]:
            out.append(f"{'Objects' if len(c['wrong_keys']) > 1 else 'Object'} {ls(c['wrong_keys'])} went into "
                       f"{part} but did not turn it.")
        if c["no_fit"]:
            out.append(f"{'Objects' if len(c['no_fit']) > 1 else 'Object'} {ls(c['no_fit'])} did not go "
                       f"{'into ' + part if part else 'in anywhere'}.")
        if part and not c["turned"]:
            out.append(f"{part[0].upper() + part[1:]} is probably a lock whose key is not within my reach.")
        elif c["stuck_part"]:
            out.append(f"What holds it is probably {sh(c['stuck_part'])}: none of my actions changed it, and nothing "
                       f"I found went into it.")
        else:
            out.append("What it needs is probably not within my reach.")
        return " ".join(out)
    if not c["opened"]:
        return f"I did not manage to open it within {log.n_actions} actions."
    p = c["probe"]
    move = {"pull": "pulling", "push": "pushing"}[FAMILY[p[0]]]
    out = []
    if c["latched"]:
        how = "pressing" if p[0] == "press_push" else "turning"
        tail = "" if c["latch_tested"] else " (I did not check that plain " + move + " fails there)"
        out.append(f"The door was latched: it opens when I keep {how} {sh(p[1])} while {move}{tail}.")
    else:
        out.append(f"The door opens by {move} {sh(p[1])}.")
    if c["deadbolt_part"]:
        e = c["deadbolt_part"]
        sure = "" if any(entity_of(f) == e for f in c["conditions"]) else " probably"
        alt = " or ".join(sh(x) for x in c["bolt_alternatives"])
        alt = f" (or after I turned {alt}: I could not tell which)" if alt else ""
        out.append(f"It was{sure} also bolted: it only moved after I turned {sh(e)}{alt}.")
    if c["key_part"] and c["key_used"]:
        k, part = c["key_used"], c["key_part"]
        found = f" I found inside {sh(c['key_found_in'])}" if c["key_found_in"] else ""
        out.append(f"It was also locked: it only opened after I put the object {k}{found} into {sh(part)} and "
                   f"turned it.")
        for w in c["wrong_keys"]:
            out.append(f"Object {w} went in but did not turn.")
        for w in c["no_fit"]:
            out.append(f"Object {w} did not go in.")
    return " ".join(out)


def explain(log, radius: float = 0.7) -> Explanation:
    c = claims_of(log, radius)
    return Explanation(text_of(c, log), c)


FIELDS = ("opened", "no_solution", "latched", "deadbolt_part", "key_part", "key_used", "key_found_in", "wrong_keys")


def _tried_decoys(log, part: str | None, truth: dict) -> set:
    """Objects that truly fit but do not match, and that the log shows went into `part` and were turned."""
    if part is None:
        return set()
    keys = truth["keys"]
    return {a[1] for a, o, s in zip(log.actions, log.outcomes, log.states)
            if a[0] == "turn_held" and o.executed and s.get(f"obj:{a[1]}.where") == f"in:{part}"
            and keys.get(a[1], {}).get("fits") and not keys[a[1]]["matches"]}


def score(claims: dict, truth: dict, log=None) -> dict:
    """Per-field correctness against MockWorld.truth(): 'accuracy' over every field scored, 'relevant' over
    the fields of the mechanisms that were really there (so that "no deadbolt" == "no deadbolt" does not
    pad the score), and 'false_claims', the mechanisms claimed that were not there.

    A solvable door is scored on every mechanism, whether or not the agent opened it (claiming no
    solution there leaves them unexplained: wrong). A key-locked door without a solution is scored on the
    lock the agent names. With the log, wrong_keys must be exactly the decoys it put into the true lock
    and turned; without it, only that none of them matches."""
    P = truth["parts"]
    solvable = truth["solvable"]
    cyl = P.get("cylinder")
    ok = {"opened": claims["opened"] == solvable, "no_solution": claims["no_solution"] == (not solvable)}
    relevant = ["opened", "no_solution"]
    if solvable or claims["opened"]:
        ok["latched"] = claims["latched"] == truth["latched"]
        ok["deadbolt_part"] = claims["deadbolt_part"] == (P.get("thumb") if truth["deadbolt"] else None)
        ok["key_part"] = claims["key_part"] == (cyl if truth["key_locked"] else None)
        ok["key_used"] = claims["key_used"] == (truth["matching_key"] if truth["key_locked"] else None)
        relevant += ["latched"] * truth["latched"] + ["deadbolt_part"] * truth["deadbolt"]
        if truth["key_locked"]:
            ok["key_found_in"] = claims["key_found_in"] == (P["drawer"] if truth["key_place"] == "drawer" else None)
            relevant += ["key_part", "key_used", "key_found_in"]
    elif truth["key_locked"]:
        ok["key_part"] = (claims["key_part"] or claims.get("stuck_part")) == cyl
        relevant.append("key_part")
    if log is not None:
        ok["wrong_keys"] = set(claims["wrong_keys"]) == _tried_decoys(log, cyl, truth)
    else:
        ok["wrong_keys"] = all(not truth["keys"].get(k, {}).get("matches", False) for k in claims["wrong_keys"])
    if truth["key_locked"] or claims["wrong_keys"]:
        relevant.append("wrong_keys")
    false_claims = (bool(claims["deadbolt_part"]) and not truth["deadbolt"]) +                    (bool(claims["key_used"]) and not truth["key_locked"]) + (bool(claims["latched"]) and not truth["latched"])
    return {**ok, "accuracy": float(np.mean(list(ok.values()))),
            "relevant": float(np.mean([ok[k] for k in relevant])), "false_claims": int(false_claims)}
