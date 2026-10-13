"""Inference wrapper for the cross-domain HRM cause thinker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pai.cognition.thinker_data import CAUSES, FEATURES, WINDOW
from pai.thinker.hrm import HRM


class SharedCauseThinker:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        if tuple(payload["causes"]) != CAUSES or tuple(payload["features"]) != FEATURES:
            raise ValueError("checkpoint evidence schema differs from the shared cause schema")
        self.device = torch.device(device)
        self.model = HRM(len(FEATURES), 0, WINDOW, {"cause": len(CAUSES)}, d_model=32, n_heads=4,
                         h_layers=1, l_layers=1, h_cycles=1, l_steps=1, max_segments=2)
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def infer(self, window: np.ndarray) -> dict[str, float]:
        if window.shape != (WINDOW, len(FEATURES)):
            raise ValueError(f"expected evidence window {(WINDOW, len(FEATURES))}")
        x = torch.as_tensor(window[None], dtype=torch.float32, device=self.device)
        logits = self.model.forward_segments(x, torch.ones(x.shape[:2], device=self.device),
                                             n_segments=1)[-1].logits["cause"]
        return dict(zip(CAUSES, map(float, logits.softmax(-1)[0].cpu().tolist())))


class OnlineMazeCauseThinker:
    """Infer during exploration from raw stall and visual-frame evidence."""

    def __init__(self, checkpoint: str | Path, threshold: float = 0.9,
                 device: str = "cpu"):
        self.thinker = SharedCauseThinker(checkpoint, device)
        self.threshold = threshold
        self.history: list[np.ndarray] = []
        self.reports: list[dict] = []

    def observe(self, step: int, *, unexpected_collision: bool = False,
                frame_shift: bool = False) -> dict | None:
        while len(self.history) <= step:
            row = np.zeros(len(FEATURES), np.float32)
            row[-1] = 1.0
            self.history.append(row)
        row = self.history[step]
        if unexpected_collision:
            row[0] = row[1] = 3.0
        if frame_shift:
            row[3] = 3.0
        if not unexpected_collision and not frame_shift:
            return None
        window = np.zeros((WINDOW, len(FEATURES)), np.float32)
        lo = max(0, step - WINDOW + 5)
        selected = self.history[lo:lo + WINDOW]
        window[:len(selected)] = selected
        posterior = self.thinker.infer(window)
        best = max(posterior, key=posterior.get)
        report = {"step": step, "cause": best, "confidence": posterior[best],
                  "posterior": posterior, "actionable": best != "none" and posterior[best] >= self.threshold}
        self.reports.append(report)
        return report


class OnlineArmCauseThinker:
    """Stream the arm's calibrated residuals into the same cause network."""

    def __init__(self, checkpoint: str | Path, threshold: float = 0.9,
                 device: str = "cpu"):
        self.thinker = SharedCauseThinker(checkpoint, device)
        self.threshold = threshold
        self.history: list[np.ndarray] = []
        self.reports: list[dict] = []

    def observe(self, step: int, z: np.ndarray, holding: bool,
                trigger: bool = False) -> dict | None:
        z = np.asarray(z, np.float32)
        if z.shape != (12,):
            raise ValueError("arm evidence must have 12 calibrated residual channels")
        row = np.asarray([np.linalg.norm(z[:3]), np.linalg.norm(z[3:6]),
                          np.linalg.norm(z[6:9]), np.linalg.norm(z[9:12]),
                          z[2], z[5], z[8], float(holding), 0.0], np.float32)
        row[:7] = np.arcsinh(row[:7])
        self.history.append(row)
        if not trigger:
            return None
        window = np.zeros((WINDOW, len(FEATURES)), np.float32)
        lo = max(0, len(self.history) - WINDOW + 4)
        selected = self.history[lo:lo + WINDOW]
        window[:len(selected)] = selected
        posterior = self.thinker.infer(window)
        best = max(posterior, key=posterior.get)
        report = {"step": step, "cause": best, "confidence": posterior[best],
                  "posterior": posterior,
                  "actionable": best != "none" and posterior[best] >= self.threshold}
        self.reports.append(report)
        return report
