"""Skill registry: every skill behind one signature, so an exploring agent can enumerate its options.

`SKILLS[name].fn(ctx, target, **params) -> SkillResult`. A `SkillDef` also says which kinds of target
the skill applies to ("part": on a fixture, "object": movable; none = the skill needs no target),
whether it needs a held object, and a few discrete `variants` (turn +/-, pull / push). `options()`
crosses them with the targets in view: the "skill x target" pairs over which the discovery agent keeps
its belief about what changes what (docs/DISCOVERY.md). Nothing here ranks the options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pai.skills import primitives as P
from pai.skills.core import SkillContext, SkillResult, Target


@dataclass(frozen=True)
class SkillDef:
    name: str
    fn: Callable[..., SkillResult]
    targets: tuple[str, ...]              # target kinds it applies to; () = no target
    variants: tuple[dict, ...] = ({},)    # discrete parameter settings to enumerate
    needs_held: bool = False
    optional_target: bool = False         # may also run without a target (grasp/turn/push_pull in place)
    doc: str = ""


SKILLS: dict[str, SkillDef] = {
    "reach": SkillDef("reach", P.reach, ("part", "object"), doc="palm to the target, grasp orientation"),
    "grasp": SkillDef("grasp", P.grasp, ("part", "object"), optional_target=True, doc="close the hand in place"),
    "release": SkillDef("release", P.release, (), doc="open the hand in place"),
    "turn": SkillDef("turn", P.turn, ("part", "object"), ({"angle": 0.6}, {"angle": -0.6}), optional_target=True,
                     doc="rotate the palm about the target axis through the grasped point"),
    "push_pull": SkillDef("push_pull", P.push_pull, ("part", "object"),
                          ({"direction": "pull"}, {"direction": "push"}), optional_target=True,
                          doc="move along the target's approach, with a force limit"),
    "insert": SkillDef("insert", P.insert, ("part",), needs_held=True,
                       doc="bring the held object's tip to the target and push it in along the axis"),
    "retreat": SkillDef("retreat", P.retreat, (), doc="open and back away"),
}


@dataclass(frozen=True)
class Option:
    """One enumerable intervention: skill x target x parameter variant."""
    skill: str
    target: Target | None
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in self.params.items())
        t = self.target.name if self.target is not None else "-"
        return f"{self.skill}({t}{', ' + p if p else ''})"

    def run(self, ctx: SkillContext, **overrides) -> SkillResult:
        return run_skill(ctx, self.skill, self.target, **{**self.params, **overrides})


def run_skill(ctx: SkillContext, name: str, target: Target | None = None, **params) -> SkillResult:
    sd = SKILLS[name]
    if target is None and sd.targets and not sd.optional_target:
        raise ValueError(f"skill {name!r} needs a target")
    return sd.fn(ctx, target, **params)


def options(targets: list[Target], holding: bool = False) -> list[Option]:
    """Every skill x applicable target x variant. `holding`: whether something is held (insert needs it)."""
    out: list[Option] = []
    for sd in SKILLS.values():
        if sd.needs_held and not holding:
            continue
        cands = [t for t in targets if t.kind in sd.targets] if sd.targets else [None]
        for t in cands:
            for v in sd.variants:
                out.append(Option(sd.name, t, dict(v)))
    return out
