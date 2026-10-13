"""A calibrated RGB orientation estimator, independent of runtime bike state.

This estimates three frame angles from a fixed camera, not full joint motion or
inverse-dynamics controls. Simulator pose labels belong only to the teacher.
"""
import json
from pathlib import Path
import numpy as np


def image_features(image, grid_size=8):
    image = np.asarray(image)
    if not isinstance(grid_size, int) or not 2 <= grid_size <= 32:
        raise ValueError("integer grid size between two and 32 required")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] % grid_size or image.shape[1] % grid_size:
        raise ValueError("RGB image dimensions must be positive multiples of the grid size")
    if min(image.shape[:2]) < grid_size or image.dtype != np.uint8:
        raise ValueError("nonempty uint8 RGB image required")
    h, w = image.shape[:2]
    pooled = image.reshape(grid_size, h // grid_size, grid_size, w // grid_size, 3).mean(axis=(1, 3)) / 255.0
    return np.append(pooled.ravel(), 1.0)


class ImageOrientation:
    def __init__(self, weights, summary=None, grid_size=8):
        if not isinstance(grid_size, int) or not 2 <= grid_size <= 32:
            raise ValueError("integer grid size between two and 32 required")
        self.grid_size = grid_size
        self.weights = np.asarray(weights, dtype=float)
        if self.weights.shape != (3 * grid_size ** 2 + 1, 3) or not np.all(np.isfinite(self.weights)):
            raise ValueError("finite image weights matching the grid required")
        self.summary = summary or {}

    def predict(self, image):
        return image_features(image, self.grid_size) @ self.weights

    @classmethod
    def fit(cls, images, angles, ridge=1.0, grid_size=8):
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("positive finite ridge required")
        x = np.asarray([image_features(image, grid_size) for image in images])
        y = np.asarray(angles, dtype=float)
        if len(x) == 0 or y.shape != (len(x), 3) or not np.all(np.isfinite(y)):
            raise ValueError("matching nonempty finite angle labels required")
        return cls(np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y),
                   {"training_frames": len(x), "ridge": ridge, "grid_size": grid_size,
                    "channels": ["roll", "pitch", "yaw"]}, grid_size)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "weights": self.weights.tolist(), "summary": self.summary,
                                    "grid_size": self.grid_size}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported image orientation checkpoint")
        return cls(data["weights"], data["summary"], data.get("grid_size", 8))
