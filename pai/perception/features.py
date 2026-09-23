"""Frozen DINOv2 patch features, and segmentation masks pooled to the same patch grid.

A 224 x 224 frame becomes a 16 x 16 grid of patch features (ViT-S/14: 384-d each). Object slots
(week 2) are learned on these features rather than on pixels, following DINOSAUR / SlotContrast
and DINO-WM (see docs/LITERATURE.md). Features are extracted once and cached, because the encoder
is frozen: slot and world-model training then never touch images again.

The weights come from torch.hub (facebookresearch/dinov2) on first use.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

PATCH = 14
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DinoFeatures:
    def __init__(self, variant: str = "dinov2_vits14", device: str | torch.device = "cpu", half: bool = True):
        self.device = torch.device(device)
        self.model = torch.hub.load("facebookresearch/dinov2", variant).to(self.device).eval()
        self.half = half and self.device.type == "cuda"
        if self.half:
            self.model = self.model.half()

    @torch.no_grad()
    def __call__(self, images_uint8: np.ndarray | torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) uint8 with H, W multiples of 14 -> (B, H/14, W/14, C) float16/32 patch features."""
        x = torch.as_tensor(images_uint8, device=self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        if self.half:
            x = x.half()
        tokens = self.model.forward_features(x)["x_norm_patchtokens"]  # (B, h*w, C)
        h, w = x.shape[-2] // PATCH, x.shape[-1] // PATCH
        return tokens.reshape(x.shape[0], h, w, -1)


def masks_to_patches(masks: np.ndarray, n_labels: int, patch: int = PATCH) -> np.ndarray:
    """(B, H, W) integer masks -> (B, H/patch, W/patch, n_labels) fraction of each label per patch.

    Soft fractions, not a majority vote: small objects often cover only part of a patch, and the
    slot loss should know that.
    """
    m = torch.as_tensor(masks.astype(np.int64))
    one_hot = F.one_hot(m, n_labels).permute(0, 3, 1, 2).float()  # (B, L, H, W)
    pooled = F.avg_pool2d(one_hot, patch)                          # (B, L, h, w)
    return pooled.permute(0, 2, 3, 1).numpy().astype(np.float16)
