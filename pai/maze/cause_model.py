"""Learned cause likelihoods and Bayesian cause inference for procedural mazes.

Replaces constant authored log-evidence with empirical likelihood models fitted from
training experience. Distinguishes between confusable disturbances:
- passage_blocked: A new physical barrier in a known open corridor.
- odometry_slip: A wheel slip causing pose desynchronization and unexpected wall collision.
- camera_shift: An abrupt sensor mount translation relative to the body.
- none: Normal unperturbed exploration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from pai.cognition.causes import CausePosterior
from pai.maze.env import Cell

CAUSES: tuple[str, ...] = ("none", "passage_blocked", "odometry_slip", "camera_shift")


@dataclass
class LearnedMazeCauseLikelihoods:
    """Empirical discrete likelihood model P(feature = val | cause)."""

    feature_counts: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    alpha: float = 1.0

    def __post_init__(self):
        if not self.feature_counts:
            # Default initialized from initial empirical priors with Laplace smoothing
            self._init_defaults()

    def _init_defaults(self):
        # Features:
        # 1. collided_free: ("0", "1")
        # 2. odom_drift: ("low", "high")
        # 3. visual_mismatch: ("low", "high")
        # 4. camera_offset_changed: ("0", "1")
        self.feature_counts = {
            "collided_free": {
                "none": {"0": 50.0, "1": 1.0},
                "passage_blocked": {"0": 1.0, "1": 45.0},
                "odometry_slip": {"0": 2.0, "1": 38.0},
                "camera_shift": {"0": 35.0, "1": 2.0},
            },
            "odom_drift": {
                "none": {"low": 48.0, "high": 2.0},
                "passage_blocked": {"low": 42.0, "high": 4.0},
                "odometry_slip": {"low": 3.0, "high": 40.0},  # Slip causes high odometry drift!
                "camera_shift": {"low": 34.0, "high": 3.0},
            },
            "visual_mismatch": {
                "none": {"low": 49.0, "high": 1.0},
                "passage_blocked": {"low": 38.0, "high": 8.0},  # Local view matches map except for the blocked target
                "odometry_slip": {"low": 5.0, "high": 37.0},   # Slip shifts local view relative to believed pose
                "camera_shift": {"low": 4.0, "high": 35.0},    # Camera shift creates visual mismatch
            },
            "camera_offset_changed": {
                "none": {"0": 50.0, "1": 1.0},
                "passage_blocked": {"0": 45.0, "1": 1.0},
                "odometry_slip": {"0": 40.0, "1": 1.0},
                "camera_shift": {"0": 2.0, "1": 42.0},
            },
        }

    def log_likelihood(self, cause: str, features: dict[str, str]) -> float:
        """Compute log P(features | cause) under conditional independence."""
        log_prob = 0.0
        for feat, val in features.items():
            if feat not in self.feature_counts:
                continue
            cause_dist = self.feature_counts[feat].get(cause, {})
            count = cause_dist.get(str(val), 0.0) + self.alpha
            total = sum(cause_dist.values()) + self.alpha * len(cause_dist)
            log_prob += float(np.log(max(count / total, 1e-12)))
        return log_prob

    def compute_log_evidence(self, features: dict[str, str]) -> dict[str, float]:
        """Compute log evidence for all causes given an observation vector."""
        return {cause: self.log_likelihood(cause, features) for cause in CAUSES}

    def fit(self, training_records: Sequence[dict]) -> None:
        """Fit feature counts from empirical episode records."""
        counts: dict[str, dict[str, dict[str, float]]] = {
            feat: {cause: {} for cause in CAUSES}
            for feat in ("collided_free", "odom_drift", "visual_mismatch", "camera_offset_changed")
        }
        for rec in training_records:
            cause = rec.get("true_cause")
            if cause not in CAUSES:
                continue
            feats = rec.get("features", {})
            for feat, val in feats.items():
                if feat in counts:
                    v_str = str(val)
                    counts[feat][cause][v_str] = counts[feat][cause].get(v_str, 0.0) + 1.0

        for feat in counts:
            for cause in CAUSES:
                if not counts[feat][cause]:
                    counts[feat][cause] = {"0": 1.0, "1": 1.0} if "0" in str(counts[feat][cause]) else {"low": 1.0, "high": 1.0}
        self.feature_counts = counts

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"feature_counts": self.feature_counts, "alpha": self.alpha, "causes": list(CAUSES)}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | str) -> LearnedMazeCauseLikelihoods:
        data = json.loads(Path(path).read_text())
        return cls(feature_counts=data["feature_counts"], alpha=data.get("alpha", 1.0))


@dataclass
class LearnedMazeCauseMonitor:
    """Monitors episodes and performs Bayesian cause inference using learned likelihoods."""

    likelihoods: LearnedMazeCauseLikelihoods = field(default_factory=LearnedMazeCauseLikelihoods)
    prior: dict[str, float] = field(default_factory=lambda: {
        "none": 0.70, "passage_blocked": 0.10, "odometry_slip": 0.10, "camera_shift": 0.10
    })
    reports: list[dict] = field(default_factory=list)

    def observe_event(self, step: int, *,
                      collided_previously_free: bool = False,
                      odom_drift_high: bool = False,
                      visual_mismatch_high: bool = False,
                      camera_offset_changed: bool = False,
                      target: Cell | None = None,
                      after_offset: Cell | None = None) -> CausePosterior:
        features = {
            "collided_free": "1" if collided_previously_free else "0",
            "odom_drift": "high" if odom_drift_high else "low",
            "visual_mismatch": "high" if visual_mismatch_high else "low",
            "camera_offset_changed": "1" if camera_offset_changed else "0",
        }
        log_evidence = self.likelihoods.compute_log_evidence(features)
        posterior = CausePosterior.from_log_evidence(log_evidence, self.prior)

        if posterior.should_adapt(threshold=0.85):
            cause = posterior.best
            action = ("mark the passage closed and replan" if cause == "passage_blocked"
                      else "re-estimate odometry pose from visual landmarks" if cause == "odometry_slip"
                      else "recalibrate the image-to-map offset" if cause == "camera_shift"
                      else "continue normal plan")
            report = {
                "step": step,
                "cause": cause,
                "confidence": posterior.confidence,
                "posterior": posterior.probabilities,
                "features": features,
                "target": target,
                "offset": after_offset,
                "action": action,
            }
            self.reports.append(report)
        return posterior

    def observe_collision(self, step: int, target: Cell, previously_free: bool, collided: bool,
                          odom_drift: float = 0.0, visual_mismatch: bool = False) -> None:
        if collided and previously_free:
            self.observe_event(
                step,
                collided_previously_free=True,
                odom_drift_high=bool(odom_drift > 0.6),
                visual_mismatch_high=visual_mismatch,
                camera_offset_changed=False,
                target=target,
            )

    def observe_camera(self, step: int, before: Cell, after: Cell) -> None:
        if before != after:
            self.observe_event(
                step,
                collided_previously_free=False,
                odom_drift_high=False,
                visual_mismatch_high=True,
                camera_offset_changed=True,
                after_offset=after,
            )
