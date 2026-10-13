"""Bounded causal experiments with newly revealed, graspable objects.

This policy sees only the public entity, feature, action, and outcome interface.
It does not know which object is a key or which part is a lock.
"""

from __future__ import annotations

import re

from pai.discovery.interface import Action


class ObjectTrialController:
    def __init__(self, initial_state: dict | None = None):
        self.initial = dict(initial_state or {})
        self.revealed: list[str] = []
        self.attempted: set[str] = set()
        held = self.initial.get("body:hand", "empty")
        self.active: str | None = held if held != "empty" else None
        self.inserted_tried: set[tuple[str, str]] = set()
        self.turns_tried: set[tuple[str, str, int]] = set()
        self.successful_turn = False
        self.probes: list[Action] = []
        self.probe_object: str | None = None
        self.cleanup: str | None = None
        self.try_remaining = False
        self.pick_failures: dict[str, int] = {}
        self.restored_features: set[str] = set()
        self.primary_probe: Action | None = None

    def choose(self, ep, acts: list[Action], goal_probes: list[Action]) -> Action | None:
        state = ep.state
        hand = state.get("body:hand", "empty")
        if hand == "empty" and self.probes:
            while self.probes:
                probe = self.probes.pop(0)
                if probe in acts:
                    return probe
        if hand == "empty" and self.probe_object is not None:
            # Every goal probe failed after an observed turn. Retract the tested
            # object before trying the next candidate in the same fixture.
            self.cleanup = self.probe_object
            self.probe_object = None
            self.try_remaining = True
        if self.cleanup is not None:
            if hand == self.cleanup:
                release = ("release", hand)
                return release if release in acts else None
            if hand == "empty":
                where = state.get(f"obj:{self.cleanup}.where", "")
                withdraw = ("pick", self.cleanup)
                if (where.startswith("in:") and withdraw in acts
                        and self.pick_failures.get(self.cleanup, 0) < 2):
                    return withdraw
                self.cleanup = None
        if hand == "empty":
            for oid in list(self.revealed):
                if oid in self.attempted:
                    self.revealed.remove(oid)
                    continue
                pick = ("pick", oid)
                if pick in acts:
                    return pick
            if self.try_remaining:
                candidates = [a for a in acts if a[0] == "pick" and a[1] not in self.attempted]
                if candidates:
                    return min(candidates, key=lambda a: (
                        0 if a[1] in self.revealed else 1, self.pick_failures.get(a[1], 0), ep.cost(a)))
                self.try_remaining = False
            return None
        if self.active != hand:
            return None
        where = state.get(f"obj:{hand}.where", "")
        if where.startswith("in:"):
            part = where[3:]
            if self.successful_turn:
                release = ("release", hand)
                return release if release in acts else None
            for sign in (1, -1):
                turn = ("turn_held", hand, sign)
                if turn in acts and (hand, part, sign) not in self.turns_tried:
                    return turn
            withdraw = ("pick", hand)
            return withdraw if withdraw in acts else None
        if where == "held":
            candidates = [a for a in acts if a[0] == "insert" and a[1] == hand
                          and (hand, a[2]) not in self.inserted_tried]
            if candidates:
                # Small fixtures near the goal are informative insertion targets.
                # This is an appearance and geometry prior, not a part-name lookup.
                by_id = ep.ents
                goal = by_id.get(ep.goal_part)
                def rank(action):
                    part = by_id.get(action[2])
                    if part is None:
                        return (float("inf"), float("inf"))
                    size = float(part.attrs.get("size", 1.0))
                    distance = sum((x - y) ** 2 for x, y in zip(part.pos, goal.pos)) if goal else 0.0
                    return (size, distance)
                return min(candidates, key=rank)
            release = ("release", hand)
            return release if release in acts else None
        return None

    def observe(self, ep, action: Action, outcome, goal_probes: list[Action]) -> None:
        for feature, value in ep.state.items():
            self.initial.setdefault(feature, value)
        for entity in outcome.revealed:
            if entity.kind == "object" and entity.id not in self.revealed:
                self.revealed.append(entity.id)
        kind, oid = action[:2]
        hand = ep.state.get("body:hand", "empty")
        if kind == "pick" and outcome.executed and not outcome.stalled and hand == oid:
            self.active = oid
            self.attempted.add(oid)
            self.successful_turn = False
        elif kind == "pick" and hand == "empty":
            self.pick_failures[oid] = self.pick_failures.get(oid, 0) + 1
            if self.pick_failures[oid] >= 2:
                self.attempted.add(oid)
        elif kind == "insert":
            self.inserted_tried.add((oid, action[2]))
        elif kind == "turn_held" and self.active == oid:
            where = ep.state.get(f"obj:{oid}.where", "")
            if where.startswith("in:"):
                self.turns_tried.add((oid, where[3:], int(action[2])))
                if (outcome.executed and not outcome.stalled and
                        any(feature.startswith("part:") and feature.endswith(".angle")
                            for feature in outcome.changed)):
                    self.successful_turn = True
        elif kind == "release" and self.active == oid:
            if self.successful_turn:
                goal = ep.ents.get(ep.goal_part)
                def probe_rank(probe):
                    entity = ep.ents.get(probe[1])
                    distance = (sum((x - y) ** 2 for x, y in zip(entity.pos, goal.pos))
                                if entity is not None and goal is not None else float("inf"))
                    rotations = []
                    for old, outcome in zip(ep.log.actions, ep.log.outcomes):
                        if old == probe:
                            match = re.search(r"turned=([-+]?\d+(?:\.\d+)?)", outcome.notes)
                            if match:
                                rotations.append(abs(float(match.group(1))))
                    turned = max(rotations, default=0.0)
                    return (0 if turned > 0.25 else 1, distance, -turned,
                            {"turn_pull": 0, "turn_push": 1, "press_push": 2}.get(probe[0], 3),
                            ep.cost(probe))
                self.probes = sorted(goal_probes, key=probe_rank)
                self.primary_probe = self.probes[0] if self.probes else None
                self.probe_object = oid
            else:
                self.try_remaining = True
            self.active = None
            self.successful_turn = False
            if self.cleanup == oid:
                self.cleanup = None
        elif (self.probe_object is not None and action == self.primary_probe
              and outcome.stalled and not ep.world.goal_reached()):
            inserted = ep.state.get(f"obj:{self.probe_object}.where", "")
            inserted_part = inserted[3:] if inserted.startswith("in:") else None
            for feature, value in sorted(ep.state.items()):
                if (not feature.startswith("part:") or not feature.endswith(".angle")
                        or feature in self.restored_features or value not in ("pos", "neg")):
                    continue
                part = feature.split(":", 1)[1].split(".", 1)[0]
                if part in (inserted_part, action[1]):
                    continue
                self.restored_features.add(feature)
                restore = ("turn", part, -1 if value == "pos" else 1)
                self.probes = [restore, self.primary_probe] + self.probes
                break
