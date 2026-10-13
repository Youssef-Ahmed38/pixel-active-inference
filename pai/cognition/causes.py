"""Common posterior and adaptation threshold; likelihood models remain domain specific."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CausePosterior:
    probabilities: dict[str, float]

    @classmethod
    def from_log_evidence(cls, evidence: dict[str, float], prior: dict[str, float] | None = None):
        if not evidence:
            raise ValueError("at least one cause hypothesis is required")
        prior = prior or {name: 1.0 for name in evidence}
        names = tuple(evidence)
        scores = np.array([evidence[name] + np.log(max(prior.get(name, 0.0), 1e-12))
                           for name in names], dtype=float)
        scores -= scores.max()
        p = np.exp(scores)
        p /= p.sum()
        return cls(dict(zip(names, map(float, p))))

    @property
    def best(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.best]

    def should_adapt(self, threshold: float = 0.9) -> bool:
        return self.best != "none" and self.confidence > threshold


def select_arm_cause(report, thinker_report: dict | None, threshold: float = 0.9):
    """A calibrated diagnosis survives a thinker's false negative.

    A strong matching thinker diagnosis can act earlier when the Bayesian
    posterior has not yet crossed the adaptation threshold.
    """
    if report is None:
        return None, False
    posterior = CausePosterior(report.posterior)
    confident = posterior.should_adapt(threshold)
    if (thinker_report is not None and thinker_report.get("actionable")
            and thinker_report.get("cause") == report.best):
        confident = True
    return report.best, confident
