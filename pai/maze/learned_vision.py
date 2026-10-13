"""Small supervised RGB tile model trained from local simulator frames.

The simulator supplies labels only during training and scoring. At inference the
model receives RGB pixels; the maze policy still sees only decoded observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pai.maze.env import MazeWorld, VISUAL_COLORS

LABELS = tuple(VISUAL_COLORS)


@dataclass
class TileColorModel:
    means: np.ndarray
    variances: np.ndarray

    @classmethod
    def fit(cls, pixels: np.ndarray, labels: np.ndarray) -> "TileColorModel":
        pixels = np.asarray(pixels, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if pixels.ndim != 2 or pixels.shape[1] != 3 or len(pixels) != len(labels):
            raise ValueError("expected aligned RGB samples and class labels")
        if set(labels.tolist()) != set(range(len(LABELS))):
            raise ValueError("training examples must include every tile class")
        means = np.stack([pixels[labels == i].mean(0) for i in range(len(LABELS))])
        variances = np.stack([pixels[labels == i].var(0) + 100.0 for i in range(len(LABELS))])
        return cls(means, variances)

    def predict_rgb(self, pixels: np.ndarray) -> np.ndarray:
        values = np.asarray(pixels, dtype=np.float64)
        error = values[..., None, :] - self.means
        score = (error**2 / self.variances + np.log(self.variances)).sum(-1)
        return score.argmin(-1).astype(np.uint8)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, means=self.means, variances=self.variances)

    @classmethod
    def load(cls, path: str | Path) -> "TileColorModel":
        with np.load(path) as data:
            return cls(data["means"].copy(), data["variances"].copy())


def simulator_samples(seeds, *, appearance_jitter: float = 12.0,
                      appearance_bias=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Collect labelled pixels from independently generated rendered mazes."""
    pixels, labels = [], []
    for seed in seeds:
        world = MazeWorld(seed, size=11, sensor_range=3,
                          appearance_jitter=appearance_jitter,
                          appearance_bias=appearance_bias)
        pixels.append(np.asarray([world.visual_colors[k] for k in LABELS], dtype=float))
        labels.append(np.arange(len(LABELS)))
        for _ in range(12):
            frame = world.visual_observe(odom_noise=0.0).image
            n = frame.shape[0] // 6
            rgb = frame.reshape(n, 6, n, 6, 3).mean((1, 3)).reshape(-1, 3)
            palette = np.asarray([world.visual_colors[k] for k in LABELS], dtype=float)
            y = ((rgb[:, None, :] - palette) ** 2).sum(-1).argmin(-1)
            pixels.append(rgb)
            labels.append(y)
            valid = [d for d in ((-1, 0), (0, 1), (1, 0), (0, -1))
                     if not world.grid[world.position[0] + d[0], world.position[1] + d[1]]]
            world.step(valid[int(world.rng.integers(len(valid)))])
    return np.concatenate(pixels), np.concatenate(labels)
