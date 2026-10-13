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

from pai.world.entities import GRIP, POS, VEL

GRIP_OPEN = 0.07      # finger opening counted as open (m); fully open is 0.08
GRIP_CLOSED = 0.05    # opening below this with an object between the fingers counts as holding it
GRIP_EMPTY = 0.01     # opening below this: the fingers closed on nothing (the object escaped them)
GRIP_SETTLED = 0.0005 # change of the opening per planner step below which the fingers have stopped closing (m)
PLACE_SPEED = 0.04    # m/s; the object must be nearly still before it is released


@dataclass
class Subgoal:
    name: str
    cost: Callable[[torch.Tensor], torch.Tensor]   # tokens (..., N, D) -> cost (...)
    done: Callable[[torch.Tensor], bool]            # current tokens (N, D) -> reached?
    grip: float | None = None                       # gripper mode chosen by L2: 1 open, -1 closed, None = plan it


class RelationalGoal:
    """on(obj, target) or in(obj, target) as an ordered list of subgoals."""

    def __init__(self, relation: str, obj: str, target: str, names: list[str], objects: dict,
                 block_clearance: float = 0.008, contact_repair: bool = True):
        self.relation, self.obj, self.target = relation, obj, target
        self.i_g, self.i_o, self.i_t = 0, names.index(obj), names.index(target)
        o, t = objects[obj], objects[target]
        self.h_o = o.height
        self.target_is_block = t.kind == "block"
        self.block_clearance = block_clearance
        self.contact_repair = contact_repair
        self.t_top = (t.height / 2 if t.kind == "block" else
                      t.height if t.kind != "bowl" else 0.012)
        # Carry and grasp geometry (m). carry_low() changes them for an object that slips in the
        # fingers; the defaults are the original hand-written decomposition.
        self.lift_z: float | None = 0.12  # object height to lift to; None: carry_above over the release point
        self.carry_above = 0.12           # carry height above the release point
        self.grasp_raise = 0.006          # grasp this far above the object centre (fingertips clear the table)
        # Widest opening at which the fingers can be closed on the object: its diagonal, when they
        # close on a block that sits at an angle to them (the agent does not control the wrist yaw).
        self.grip_hold = max(GRIP_CLOSED + 0.005, min(GRIP_OPEN - 0.005, o.height * 2 ** 0.5 + 0.003))
        self._last_grip: float | None = None
        self._grip_settled = False
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
        # Release 2.5 cm above a plate, but only 8 mm above a supporting block so
        # the second impact is less likely to knock the base out of place. (Diagnosis in
        # results/slice_v3: failed placements came from the hand moving sideways while the fingers
        # opened, throwing the block into the plate, not from the release height.)
        def place_z(x):
            # A supporting block can itself be on another block. Its observed centre,
            # rather than a fixed table height, determines the release height.
            if self.target_is_block and self.contact_repair:
                return self._t(x)[..., 2] + self.t_top + self.h_o / 2 + self.block_clearance
            return self.t_top + self.h_o / 2 + 0.025

        def target_at(x, height):
            z = place_z(x) + height
            if not isinstance(z, torch.Tensor):
                z = torch.full_like(self._t(x)[..., :1], z)
            else:
                z = z.unsqueeze(-1)
            return torch.cat([self._t(x)[..., :2], z], -1)
        sq = lambda v: (v**2).sum(-1)
        open_cost = lambda x: 20.0 * torch.relu(GRIP_OPEN - self._grip(x)) ** 2 * 100
        closed_cost = lambda x: 20.0 * torch.relu(self._grip(x) - GRIP_CLOSED + 0.01) ** 2 * 100

        def above_obj(x):
            tgt = self._o(x) + torch.tensor([0, 0, hover], device=x.device)
            return sq(self._g(x) - tgt) * 100 + open_cost(x)

        def at_obj(x):
            tgt = self._o(x) + torch.tensor([0, 0, self.grasp_raise], device=x.device)
            return sq(self._g(x) - tgt) * 100 + open_cost(x)

        def grasped(x):
            return sq(self._g(x) - self._o(x)) * 100 + closed_cost(x)

        def lift_goal(x):
            return self.lift_z if self.lift_z is not None else place_z(x) + self.carry_above

        def lifted(x):
            z = self._o(x)[..., 2]
            return 100 * torch.relu(lift_goal(x) - z) ** 2 + sq(self._g(x) - self._o(x)) * 100 + closed_cost(x)

        def over_target(x):
            tgt = target_at(x, self.carry_above)
            return sq(self._o(x) - tgt) * 100 + closed_cost(x)

        def lowered(x):
            # Place gently: releasing while the object still moves makes it hit a light plate at speed,
            # which knocks the plate aside or tips it over the object (8 of 9 failures in evaluation v2).
            tgt = target_at(x, 0.0)
            speed = sq(x[..., self.i_o, VEL])
            support_speed = sq(x[..., self.i_t, VEL]) if self.target_is_block and self.contact_repair else 0.0
            return sq(self._o(x) - tgt) * 100 + 20.0 * (speed + support_speed) + closed_cost(x)

        def opened(x):
            # Hold still while the fingers open: moving the hand at release throws the object.
            still = sq(x[..., self.i_g, VEL])
            support = sq(x[..., self.i_t, VEL]) if self.target_is_block and self.contact_repair else 0.0
            overlap = sq(self._o(x)[..., :2] - self._t(x)[..., :2]) if self.target_is_block and self.contact_repair else 0.0
            return 200.0 * still + 20.0 * support + 100.0 * overlap + open_cost(x)

        def released(x):
            # Retreat straight up: stay above the object, rise to the hover height.
            over = sq(self._g(x)[..., :2] - self._o(x)[..., :2])
            away = torch.relu(self._o(x)[..., 2] + hover - self._g(x)[..., 2]) ** 2
            return 100 * (over + away) + open_cost(x)

        d3 = lambda x, y: float((x - y).norm())
        return [
            Subgoal("above_object", above_obj, grip=1.0, done=
                    lambda x: d3(self._g(x), self._o(x) + torch.tensor([0, 0, hover], device=x.device)) < 0.015),
            Subgoal("at_object", at_obj, grip=1.0, done=
                    lambda x: d3(self._g(x), self._o(x) + torch.tensor([0, 0, self.grasp_raise], device=x.device)) < 0.008),
            # Closed on the object: the fingers are nearly shut on it, or have stopped closing at an
            # opening the object can fill (a block held at an angle keeps them wider apart).
            Subgoal("grasped", grasped, grip=-1.0, done=lambda x: (
                    float(self._grip(x)) < GRIP_CLOSED
                    or (self._grip_settled and GRIP_EMPTY < float(self._grip(x)) < self.grip_hold))
                    and d3(self._g(x), self._o(x)) < 0.02),
            Subgoal("lifted", lifted, grip=-1.0, done=lambda x: float(self._o(x)[2]) > float(lift_goal(x)) - 0.02),
            Subgoal("over_target", over_target, grip=-1.0, done=
                    lambda x: d3(self._o(x)[:2], self._t(x)[:2]) < 0.015 and float(self._o(x)[2]) > float(place_z(x)) + self.carry_above - 0.03),
            Subgoal("lowered", lowered, grip=-1.0, done=
                    lambda x: d3(self._o(x)[:2], self._t(x)[:2]) < 0.015 and abs(float(self._o(x)[2]) - float(place_z(x))) < 0.01
                    and float(x[self.i_o, VEL].norm()) < PLACE_SPEED
                    and (not self.target_is_block or not self.contact_repair
                         or float(x[self.i_t, VEL].norm()) < PLACE_SPEED)),
            Subgoal("opened", opened, grip=1.0, done=
                    lambda x: float(self._grip(x)) > GRIP_OPEN and float(x[self.i_o, VEL].norm()) < PLACE_SPEED
                    and (not self.target_is_block or not self.contact_repair or
                         (float(x[self.i_t, VEL].norm()) < PLACE_SPEED
                          and d3(self._o(x)[:2], self._t(x)[:2]) < 0.02))),
            Subgoal("released", released, grip=1.0, done=
                    lambda x: float(self._grip(x)) > GRIP_OPEN and float(self._g(x)[2] - self._o(x)[2]) > hover - 0.03),
        ]

    def holding(self, x: torch.Tensor) -> bool:
        """Maintenance condition of the carrying subgoals: gripper closed around the object."""
        return float(self._grip(x)) < self.grip_hold and float((self._g(x) - self._o(x)).norm()) < 0.03

    def carry_low(self, carry_above: float = 0.01, grasp_raise: float = 0.0) -> None:
        """Carry an object that slips in the fingers: it creeps out of the grip the longer it is held
        (in the simulator at 5% friction it is lost within about a second of lifting), so lift it only
        just clear of the scene and the target, carry it there directly, and grasp it at its centre,
        which leaves the most finger length for it to slide along before it escapes. The scripted
        oracle with this carry places such a block in 20 of 20 development episodes, with the
        default carry in 0 of 20 (pai.envs.scripted.PickPlace)."""
        self.lift_z, self.carry_above, self.grasp_raise = None, carry_above, grasp_raise

    # ------------------------------------------------------------------ L2 interface
    @property
    def current(self) -> Subgoal | None:
        return self.subgoals[self.index] if self.index < len(self.subgoals) else None

    HOLD_FROM, HOLD_TO = 3, 6  # lifted, over_target, lowered need the object in the hand

    def update(self, tokens: torch.Tensor) -> bool:
        """Advance past satisfied subgoals; fall back to re-grasping if a carrying subgoal has lost
        its maintenance condition (the object is no longer held). Returns True when done."""
        self.fell_back = False
        grip = float(self._grip(tokens))
        self._grip_settled = self._last_grip is not None and abs(grip - self._last_grip) < GRIP_SETTLED
        self._last_grip = grip
        if self.current is not None and self.current.name == "grasped" and self._grip_settled and grip < GRIP_EMPTY:
            # The fingers closed on nothing: the object was squeezed out as they closed. Closed
            # fingers cannot pick it up; open and approach it again.
            self.index, self.fell_back = 0, True
            return False
        if self.index == 0 and self.holding(tokens):
            # A retried relation may begin with the object already in hand.
            # Opening the gripper for "above_object" would drop it.
            self.index = self.HOLD_FROM
        if self.HOLD_FROM <= self.index < self.HOLD_TO and not self.holding(tokens):
            self.index, self.fell_back = 0, True
        while self.current is not None and self.current.done(tokens):
            self.index += 1
        return self.current is None

    def reset(self) -> None:
        self.index = 0
