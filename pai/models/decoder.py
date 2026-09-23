"""PixelAI visual generative model g(mu): joint angles -> predicted camera image.

A deconvolutional decoder as in PixelAI (Sancaktar et al., 2020), modernised with
upsample+conv blocks (no checkerboard artefacts) and GroupNorm. It must be differentiable
w.r.t. its input: active inference needs dg/dmu for both perception and action.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.norm1(self.conv1(x)))
        return F.silu(self.norm2(self.conv2(x)))


class JointDecoder(nn.Module):
    """g: R^n_joints -> [0, 1]^(3 x S x S).

    Inputs are raw joint angles (rad); they are normalised to [-1, 1] with the joint
    ranges stored as buffers, so the checkpoint is self-contained.
    """

    def __init__(self, joint_low, joint_high, image_size: int = 128, base_channels: int = 32, max_channels: int = 512):
        super().__init__()
        n_up = int(math.log2(image_size // 4))
        if 4 * 2**n_up != image_size:
            raise ValueError(f"image_size must be 4 * 2^k, got {image_size}")
        self.image_size = image_size
        self._config = {
            "joint_low": [float(v) for v in joint_low],
            "joint_high": [float(v) for v in joint_high],
            "image_size": image_size,
            "base_channels": base_channels,
            "max_channels": max_channels,
        }
        self.register_buffer("q_low", torch.as_tensor(joint_low, dtype=torch.float32))
        self.register_buffer("q_high", torch.as_tensor(joint_high, dtype=torch.float32))
        n_in = self.q_low.numel()

        chans = [min(max_channels, base_channels * 2 ** (n_up - i)) for i in range(n_up + 1)]
        self.fc = nn.Sequential(
            nn.Linear(n_in, 256), nn.SiLU(),
            nn.Linear(256, 512), nn.SiLU(),
            nn.Linear(512, chans[0] * 16),
        )
        self.blocks = nn.Sequential(*[UpBlock(chans[i], chans[i + 1]) for i in range(n_up)])
        self.out = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.c0 = chans[0]

    def normalize(self, q: torch.Tensor) -> torch.Tensor:
        return 2 * (q - self.q_low) / (self.q_high - self.q_low) - 1

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        x = self.fc(self.normalize(q)).view(-1, self.c0, 4, 4)
        return torch.sigmoid(self.out(self.blocks(x)))

    def config(self) -> dict:
        return dict(self._config)


def build_decoder(joint_low, joint_high, cfg) -> JointDecoder:
    return JointDecoder(joint_low, joint_high, cfg.image_size, cfg.base_channels, cfg.max_channels)


def load_decoder(path, device="cpu") -> JointDecoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dec = JointDecoder(**ckpt["decoder_config"])
    dec.load_state_dict(ckpt["model"])
    return dec.to(device).eval()


def to_tensor_image(img_uint8, size: int, device) -> torch.Tensor:
    """(H, W, 3) or (B, H, W, 3) uint8 -> (B, 3, size, size) float in [0, 1]."""
    x = torch.as_tensor(img_uint8, device=device)
    if x.dim() == 3:
        x = x[None]
    x = x.permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="area")
    return x
