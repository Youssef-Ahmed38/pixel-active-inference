"""Episodic memory: every attempt is stored, so later levels can recall, replay and explain.

A record holds what the agent wanted, what it did, what it expected, what surprised it, what it
concluded about the cause, and how it ended, plus the simulator's ground truth for scoring.
Records are appended to a JSONL file. Retrieval by similarity and replay come in Phase 3.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EpisodeRecord:
    episode: int
    goal: str
    success: bool
    steps: int
    subgoal_times: dict[str, int]
    surprise: list[float]
    inferred_cause: str | None
    cause_posterior: dict[str, float] | None
    cause_params: dict | None
    true_disturbances: list[dict]
    fallbacks: int = 0              # times a carrying subgoal lost the object and fell back
    adaptations: list[dict] = field(default_factory=list)  # what the agent changed after an explanation
    surprise_threshold: float = 6.0  # calibrated spike threshold used in this episode
    events: list[dict] = field(default_factory=list)
    # Trajectory for replay in imagination: entity tokens (T+1, N, D), actions (T, A), and the
    # planner step at which each subgoal was active: name -> (first, last).
    tokens: np.ndarray | None = field(default=None, repr=False)
    actions: np.ndarray | None = field(default=None, repr=False)
    phases: dict[str, tuple[int, int]] = field(default_factory=dict)


class EpisodicMemory:
    def __init__(self, path: str | Path | None = None):
        self.records: list[EpisodeRecord] = []
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: EpisodeRecord) -> None:
        self.records.append(record)
        if self.path:
            row = asdict(record)
            tokens, actions = row.pop("tokens"), row.pop("actions")
            if tokens is not None:
                traj = self.path.parent / f"{self.path.stem}_ep{record.episode:04d}.npz"
                np.savez_compressed(traj, tokens=tokens, actions=actions)
                row["trajectory"] = traj.name
            with self.path.open("a") as f:
                f.write(json.dumps(row, default=_json_default) + "\n")

    def successes(self) -> list[EpisodeRecord]:
        return [r for r in self.records if r.success and r.tokens is not None]

    def __len__(self) -> int:
        return len(self.records)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
