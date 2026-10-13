"""Shared upper-level vocabulary for task domains with different bodies and sensors.

Coordinates and feature vectors remain local to a domain. Names, relations, actions,
prediction errors and outcomes are the portable layer. A domain adapter must not
put simulator truth in ``StateBelief``; truth belongs only in evaluation records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class EntityState:
    id: str
    kind: str
    features: dict[str, float | str | bool]
    confidence: float = 1.0


@dataclass(frozen=True)
class GoalRelation:
    predicate: str
    subject: str
    object: str | None = None
    preference: float = 1.0


@dataclass(frozen=True)
class StateBelief:
    domain: str
    step: int
    entities: tuple[EntityState, ...]
    relations: frozenset[tuple[str, str, str | None]]
    uncertainty: float = 0.0

    def satisfies(self, goal: GoalRelation) -> bool:
        return (goal.predicate, goal.subject, goal.object) in self.relations


@dataclass(frozen=True)
class SkillCommand:
    name: str
    target: str | None = None
    parameters: tuple[float, ...] = ()
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillOutcome:
    command: SkillCommand
    executed: bool
    stalled: bool
    duration: float
    changed: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SurpriseEvidence:
    step: int
    channels: tuple[str, ...]
    residual: tuple[float, ...]
    sigma: tuple[float, ...]

    @property
    def surprise(self) -> float:
        if len(self.channels) != len(self.residual) or len(self.channels) != len(self.sigma):
            raise ValueError("evidence channels and values must have the same length")
        return 0.5 * sum((r / max(s, 1e-9)) ** 2 for r, s in zip(self.residual, self.sigma))


@dataclass
class CognitiveEpisode:
    domain: str
    episode: int
    goal: GoalRelation
    success: bool
    steps: int
    beliefs: list[StateBelief] = field(default_factory=list)
    commands: list[SkillCommand] = field(default_factory=list)
    outcomes: list[SkillOutcome] = field(default_factory=list)
    evidence: list[SurpriseEvidence] = field(default_factory=list)
    cause_posterior: dict[str, float] = field(default_factory=dict)
    adaptations: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class DomainAdapter(Protocol):
    """Only L0 and perception know the particular body and simulator."""

    domain: str

    def observe(self) -> StateBelief: ...
    def available_skills(self) -> tuple[SkillCommand, ...]: ...
    def execute(self, command: SkillCommand) -> SkillOutcome: ...
