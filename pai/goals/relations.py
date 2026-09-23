"""Relational goals as preferences over entity states, and their decomposition into subgoals.

In active inference a goal is a prior preference: log C(outcome). Here each subgoal is a smooth
cost over (predicted) entity tokens and log C = -cost / temperature, so the planner's pragmatic
value is how preferred the imagined outcome is. Each subgoal also has a crisp `done` test on the
agent's current belief, which L2 uses to advance to the next subgoal.

The decomposition of on(obj, target) is written by hand for the vertical slice. In Phase 3 it
becomes a learned recipe (preconditions + action sketch + why).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from pai.world.entities import GRIP, POS

GRIP_OPEN = 0.07      # finger opening counted as open (m); fully open is 0.08
GRIP_CLOSED = 0.05    # opening below this with an object between the fingers counts as holding it


@dataclass
class Subgoal:
    name: str
    cost: Callable[[torch.Tensor], torch.Tensor]   # tokens (..., N, D) -> cost (...)
    done: Callable[[torch.Tensor], bool]            # current tokens (N, D) -> reached?
    grip: float | None = None                       # gripper mode chosen by L2: 1 open, -1 closed, None = plan it


class RelationalGoal:
    """on(obj, target) or in(obj, target) as an ordered list of subgoals."""

    def __init__(self, relation: str, obj: str, target: str, names: list[str], objects: dict):
        self.relation, self.obj, self.target = relation, obj, target
        self.i_g, self.i_o, self.i_t = 0, names.index(obj), names.index(target)
        o, t = objects[obj], objects[target]
        self.h_o = o.height
        self.t_top = t.height if t.kind != "bowl" else 0.012  # target surface above the table (m)
        self.subgoals = self._decompose()
        self.index = 0
        self.fell_back = False

    # ------------------------------------------------------------------ geometry helpers
    def _g(self, x):
        return x[..., self.i_g, POS]

    def _o(self, x):
        return x[..., self.i_o, POS]

    def _t(self, x):
        return x[..., self.i_t, POS]

    def _grip(self, x):
        return x[..., self.i_g, GRIP][..., 0]

    def _decompose(self) -> list[Subgoal]:
        hover, carry = 0.10, 0.12
        # Release 3.5 cm above the surface: lower and the fingertips catch a plate's rim and lift or
        # tilt it on the way out (seen in 7 of 60 evaluation episodes, mostly with heavier blocks,
        # which make the arm sag). The block drops the rest of the way.
        place_z = self.t_top + self.h_o / 2 + 0.035
        sq = lambda v: (v**2).sum(-1)
        open_cost = lambda x: 20.0 * torch.relu(GRIP_OPEN - self._grip(x)) ** 2 * 100
        closed_cost = lambda x: 20.0 * torch.relu(self._grip(x) - GRIP_CLOSED + 0.01) ** 2 * 100

        def above_obj(x):
            tgt = self._o(x) + torch.tensor([0, 0, hover], device=x.device)
            return sq(self._g(x) - tgt) * 100 + open_cost(x)

        def at_obj(x):
            tgt = self._o(x) + torch.tensor([0, 0, 0.006], device=x.device)
            return sq(self._g(x) - tgt) * 100 + open_cost(x)

        def grasped(x):
            return sq(self._g(x) - self._o(x)) * 100 + closed_cost(x)

        def lifted(x):
            z = self._o(x)[..., 2]
            return 100 * torch.relu(carry - z) ** 2 + sq(self._g(x) - self._o(x)) * 100 + closed_cost(x)

        def over_target(x):
            tgt = torch.cat([self._t(x)[..., :2], torch.full_like(self._t(x)[..., :1], place_z + carry)], -1)
            return sq(self._o(x) - tgt) * 100 + closed_cost(x)

        def lowered(x):
            tgt = torch.cat([self._t(x)[..., :2], torch.full_like(self._t(x)[..., :1], place_z)], -1)
            return sq(self._o(x) - tgt) * 100 + closed_cost(x)

        def released(x):
            tgt_xy = self._t(x)[..., :2]
            resting = (self._o(x)[..., 2] - (self.t_top + self.h_o / 2)) ** 2
            away = torch.relu(self._o(x)[..., 2] + hover - self._g(x)[..., 2]) ** 2
            return 100 * (sq(self._o(x)[..., :2] - tgt_xy) + resting + away) + open_cost(x)

        d3 = lambda x, y: float((x - y).norm())
        return [
            Subgoal("above_object", above_obj, grip=1.0, done=
                    lambda x: d3(self._g(x), self._o(x) + torch.tensor([0, 0, hover], device=x.device)) < 0.015),
            Subgoal("at_object", at_obj, grip=1.0, done=
                    lambda x: d3(self._g(x), self._o(x) + torch.tensor([0, 0, 0.006], device=x.device)) < 0.008),
            Subgoal("grasped", grasped, grip=-1.0, done=lambda x: float(self._grip(x)) < GRIP_CLOSED and d3(self._g(x), self._o(x)) < 0.02),
            Subgoal("lifted", lifted, grip=-1.0, done=lambda x: float(self._o(x)[2]) > carry - 0.02),
            Subgoal("over_target", over_target, grip=-1.0, done=
                    lambda x: d3(self._o(x)[:2], self._t(x)[:2]) < 0.015 and float(self._o(x)[2]) > place_z + carry - 0.03),
            Subgoal("lowered", lowered, grip=-1.0, done=
                    lambda x: d3(self._o(x)[:2], self._t(x)[:2]) < 0.015 and abs(float(self._o(x)[2]) - place_z) < 0.01),
            Subgoal("released", released, grip=1.0, done=
                    lambda x: float(self._grip(x)) > GRIP_OPEN and float(self._g(x)[2] - self._o(x)[2]) > hover - 0.03),
        ]

    def holding(self, x: torch.Tensor) -> bool:
        """Maintenance condition of the carrying subgoals: gripper closed around the object."""
        return float(self._grip(x)) < GRIP_CLOSED + 0.005 and float((self._g(x) - self._o(x)).norm()) < 0.03

    # ------------------------------------------------------------------ L2 interface
    @property
    def current(self) -> Subgoal | None:
        return self.subgoals[self.index] if self.index < len(self.subgoals) else None

    HOLD_FROM, HOLD_TO = 3, 6  # lifted, over_target, lowered need the object in the hand

    def update(self, tokens: torch.Tensor) -> bool:
        """Advance past satisfied subgoals; fall back to re-grasping if a carrying subgoal has lost
        its maintenance condition (the object is no longer held). Returns True when done."""
        self.fell_back = False
        if self.HOLD_FROM <= self.index < self.HOLD_TO and not self.holding(tokens):
            self.index, self.fell_back = 0, True
        while self.current is not None and self.current.done(tokens):
            self.index += 1
        return self.current is None

    def reset(self) -> None:
        self.index = 0
