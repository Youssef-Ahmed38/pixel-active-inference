"""Chess value residual learned by the project's existing recurrent HRM.

The baseline and HRM read public board tokens only. Learned rule models still
determine legality and endings; the neural head never overrides either.
"""
from pathlib import Path
import json

import numpy as np
import torch

from pai.chess.evaluation import LearnedPositionValue
from pai.chess.representation import BoardObservation
from pai.thinker.hrm import HRM


def mirror_observation(obs):
    pieces = np.asarray(obs.pieces).reshape(8, 8)[::-1].copy().reshape(-1)
    pieces = np.where(pieces == 0, 0, np.where(pieces <= 6, pieces + 6, pieces - 6))
    return BoardObservation(tuple(map(int, pieces)), 1 - obs.turn,
                            (*obs.castling[2:], *obs.castling[:2]),
                            64 if obs.ep_square == 64 else obs.ep_square ^ 56)


def board_tensors(observations):
    pieces = np.asarray([o.pieces for o in observations], dtype=int)
    x = np.zeros((len(observations), 64, 16), np.float32)
    x[:, :, :13] = np.eye(13, dtype=np.float32)[pieces]
    for index, obs in enumerate(observations):
        if obs.ep_square != 64:
            x[index, obs.ep_square, 13] = 1
    x[:, :, 14] = np.tile(np.arange(8), 8) / 7
    x[:, :, 15] = np.repeat(np.arange(8), 8) / 7
    ctx = np.asarray([[2 * o.turn - 1, *o.castling] for o in observations], dtype=np.float32)
    return torch.from_numpy(x), torch.ones((len(x), 64)), torch.from_numpy(ctx)


class HRMPositionValue:
    def __init__(self, model, baseline, config, summary=None, residual_scale=500):
        self.model = model.cpu().eval()
        self.baseline = baseline
        self.config = config
        self.summary = summary or {}
        self.residual_scale = residual_scale

    @torch.no_grad()
    def predict_many(self, observations, segments=None):
        if segments is not None and segments < 1:
            raise ValueError("positive thinking segment count required")
        observations = list(observations)
        if not observations:
            return np.empty(0, dtype=float)
        paired = observations + [mirror_observation(o) for o in observations]
        x, mask, ctx = board_tensors(paired)
        # Fixed computation: the inherited cause Q-head has not been trained for chess halting.
        result = self.model.think(x, mask, ctx, max_segments=segments, halt=False)
        raw = result.logits["value"].squeeze(-1).numpy()
        residual = (raw[:len(observations)] - raw[len(observations):]) / 2
        baseline = np.array([self.baseline.predict(o) for o in observations])
        return baseline + self.residual_scale * residual

    def predict(self, observation):
        return float(self.predict_many([observation])[0])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": 1, "type": "chess_hrm_value", "config": self.config,
                    "state_dict": self.model.state_dict(), "baseline_weights": self.baseline.weights.tolist(),
                    "baseline_summary": self.baseline.summary, "summary": self.summary,
                    "residual_scale": self.residual_scale}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["schema"] != 1 or data["type"] != "chess_hrm_value":
            raise ValueError("unsupported chess HRM checkpoint")
        model = HRM(**data["config"])
        model.load_state_dict(data["state_dict"])
        return cls(model, LearnedPositionValue(data["baseline_weights"], data["baseline_summary"]),
                   data["config"], data["summary"], data["residual_scale"])


def load_value(path):
    if Path(path).suffix == ".pt":
        return HRMPositionValue.load(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("type") == "learned_context_value":
        from pai.chess.threat_value import ContextPositionValue
        return ContextPositionValue.load(path)
    return LearnedPositionValue.load(path)
