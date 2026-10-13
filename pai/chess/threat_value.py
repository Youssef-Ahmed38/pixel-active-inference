"""Learn a small contextual residual from the already learned attack field.

No movement geometry, tactical rule or material weight is authored here. Generic
category counts aggregate attacks predicted by the taught relational rule model.
"""
import json
from pathlib import Path

import numpy as np

from pai.chess.evaluation import LearnedPositionValue
from pai.chess.rules import BETWEEN


def context_features(observation, attack_table):
    board = np.asarray(observation.pieces)
    blockers = np.tensordot(BETWEEN, board != 0, axes=([2], [0])) > 0
    attack = attack_table[board[:, None], np.arange(64)[:, None], np.arange(64)[None, :], blockers.astype(int)]
    white = ((board > 0) & (board <= 6)).astype(int) @ attack
    black = (board > 6).astype(int) @ attack
    x = []
    for category in range(1, 7):
        own, other = board == category, board == category + 6
        x.extend([np.sum(own & (black > 0)) - np.sum(other & (white > 0)),
                  np.sum(own & (black > 0) & (white == 0)) - np.sum(other & (white > 0) & (black == 0)),
                  np.sum(white[own]) - np.sum(black[other]),
                  np.sum(attack[own]) - np.sum(attack[other])])
    return np.asarray(x, dtype=float)


class ContextPositionValue:
    def __init__(self, baseline, rules, weights, summary=None):
        self.baseline, self.rules = baseline, rules
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (24,) or not np.all(np.isfinite(self.weights)):
            raise ValueError("24 finite context weights required")
        self.summary = summary or {}

    def predict(self, observation):
        return float(self.baseline.predict(observation) + context_features(observation, self.rules.attack_table) @ self.weights)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 2, "type": "learned_context_value", "weights": self.weights.tolist(),
            "baseline_weights": self.baseline.weights.tolist(), "baseline_summary": self.baseline.summary,
            "attack_table": self.rules.attack_table.tolist(), "summary": self.summary}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 2 or data["type"] != "learned_context_value":
            raise ValueError("unsupported contextual value checkpoint")
        # Only learned attack data is needed by this value head.
        from types import SimpleNamespace
        rules = SimpleNamespace(attack_table=np.asarray(data["attack_table"], dtype=np.int8))
        return cls(LearnedPositionValue(data["baseline_weights"], data["baseline_summary"]), rules,
                   data["weights"], data["summary"])
