"""Object slots on frozen DINOv2 patch features (week 2).

Slot attention (Locatello et al., 2020) groups the 16 x 16 patch features into K slots that
compete for patches. A DINOSAUR-style decoder reconstructs each patch feature as an alpha-weighted
mixture of per-slot predictions. The alphas are the slots' soft segmentation.

Training signal:
- feature reconstruction (self-supervised; the part that will remain when supervision is removed)
- mask supervision (rule 2: privileged first): slots are matched to ground-truth objects with the
  Hungarian algorithm, and their alpha masks are pulled towards the matched object's patch fractions
- token readout: from each matched slot, predict that object's entity token (position, velocity,
  yaw, kind, colour), so slots plug into the world model and planner already built on tokens

For video, slots of frame t initialise the slot attention at frame t+1. This keeps object identity
over time, the concern SlotContrast addresses (see docs/LITERATURE.md).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from pai.world.entities import TOKEN_DIM


class SlotAttention(nn.Module):
    def __init__(self, n_slots: int, dim: int, iters: int = 3, hidden: int = 256):
        super().__init__()
        self.n_slots, self.iters, self.scale = n_slots, iters, dim**-0.5
        self.mu = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm_in, self.norm_slots, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, inputs: torch.Tensor, init: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs (B, P, D) -> slots (B, K, D), attention (B, K, P) (softmax over slots)."""
        B, _, D = inputs.shape
        x = self.norm_in(inputs)
        k, v = self.k(x), self.v(x)
        if init is None:
            slots = self.mu + self.log_sigma.exp() * torch.randn(B, self.n_slots, D, device=inputs.device)
        else:
            slots = init
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.norm_slots(slots))
            attn = torch.softmax(torch.einsum("bkd,bpd->bkp", q, k) * self.scale, dim=1)  # slots compete
            weights = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bkp,bpd->bkd", weights, v)
            slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D)).reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


class SlotModel(nn.Module):
    def __init__(self, feat_dim: int = 384, n_patches: int = 256, n_slots: int = 8, dim: int = 128, iters: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)
        self.slot_attention = SlotAttention(n_slots, dim, iters)
        # DINOSAUR-style MLP decoder: each slot + position -> (patch feature, alpha logit)
        self.dec_pos = nn.Parameter(torch.randn(1, 1, n_patches, dim) * 0.02)
        self.decoder = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                     nn.Linear(512, feat_dim + 1))
        self.readout = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, TOKEN_DIM))
        self._config = {"feat_dim": feat_dim, "n_patches": n_patches, "n_slots": n_slots, "dim": dim, "iters": iters}

    def config(self) -> dict:
        return dict(self._config)

    def forward(self, feats: torch.Tensor, init: torch.Tensor | None = None) -> dict:
        """feats (B, P, feat_dim) -> dict(slots, alpha (B, K, P), recon (B, P, feat_dim), tokens (B, K, TOKEN_DIM))."""
        x = self.encoder(feats) + self.pos
        slots, attn = self.slot_attention(x, init)
        out = self.decoder(slots[:, :, None, :] + self.dec_pos)            # (B, K, P, feat_dim + 1)
        alpha = torch.softmax(out[..., -1], dim=1)                          # (B, K, P)
        recon = (alpha[..., None] * out[..., :-1]).sum(1)                   # (B, P, feat_dim)
        return {"slots": slots, "alpha": alpha, "attn": attn, "recon": recon, "tokens": self.readout(slots)}


def match_slots(alpha: torch.Tensor, target: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hungarian matching of slots to labels by soft-mask IoU.

    alpha (B, K, P) slot masks; target (B, P, L) label fractions -> per batch item (slot_idx, label_idx).
    """
    a = alpha.detach().float()
    t = target.detach().float().transpose(1, 2)                           # (B, L, P)
    inter = torch.einsum("bkp,blp->bkl", a, t)
    union = a.sum(-1)[:, :, None] + t.sum(-1)[:, None, :] - inter
    iou = (inter / (union + 1e-8)).cpu().numpy()
    return [linear_sum_assignment(-m) for m in iou]


def slot_losses(out: dict, feats: torch.Tensor, target_masks: torch.Tensor, target_tokens: torch.Tensor | None,
                object_labels: slice, mask_weight: float = 1.0, token_weight: float = 1.0) -> dict:
    """Reconstruction + (optional) mask and token supervision on matched slots.

    target_masks (B, P, L) label fractions, labels 0 = background, 1 = robot, 2.. = objects.
    target_tokens (B, n_objects + 1, TOKEN_DIM) entity tokens (gripper first), for the object labels.
    """
    losses = {"recon": F.mse_loss(out["recon"], feats)}
    if mask_weight > 0:
        pairs = match_slots(out["alpha"], target_masks)
        m_loss, t_loss, n = 0.0, 0.0, 0
        for b, (ks, ls) in enumerate(pairs):
            m_loss = m_loss + F.binary_cross_entropy(out["alpha"][b, ks].clamp(1e-6, 1 - 1e-6),
                                                     target_masks[b, :, ls].T.float())
            if target_tokens is not None:
                obj = [(k, l) for k, l in zip(ks, ls) if object_labels.start <= l < object_labels.stop]
                if obj:
                    kk = torch.as_tensor([k for k, _ in obj], device=feats.device)
                    tt = torch.as_tensor([l - object_labels.start + 1 for _, l in obj], device=feats.device)
                    t_loss = t_loss + F.mse_loss(out["tokens"][b, kk], target_tokens[b, tt])
            n += 1
        losses["mask"] = mask_weight * m_loss / n
        if target_tokens is not None:
            losses["token"] = token_weight * t_loss / n
    losses["total"] = sum(losses.values())
    return losses
