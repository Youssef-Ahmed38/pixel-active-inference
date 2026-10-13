"""One episode store for arm and maze records, indexed by goal and entity kinds."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pai.cognition.contracts import CognitiveEpisode, GoalRelation


class CognitiveMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.episodes: list[CognitiveEpisode] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, episode: CognitiveEpisode) -> None:
        self.episodes.append(episode)
        if self.path:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(episode), default=_json_default) + "\n")

    def recall(self, goal: GoalRelation, *, domain: str | None = None,
               successes_only: bool = True) -> list[CognitiveEpisode]:
        return [episode for episode in self.episodes
                if episode.goal.predicate == goal.predicate
                and episode.goal.subject == goal.subject
                and episode.goal.object == goal.object
                and (domain is None or episode.domain == domain)
                and (not successes_only or episode.success)]

    def recall_cause(self, cause: str, *, domain: str | None = None) -> list[CognitiveEpisode]:
        return [episode for episode in self.episodes
                if cause in episode.cause_posterior
                and episode.cause_posterior[cause] == max(episode.cause_posterior.values())
                and (domain is None or episode.domain == domain)]


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")
