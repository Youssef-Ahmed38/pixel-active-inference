"""Object slots on frozen DINOv2 patch features (week 2).

Slot attention (Locatello et al., 2020) groups the patch features (32 x 32 at 448 px) into K slots
that compete for patches. A DINOSAUR-style decoder reconstructs each patch feature as an
alpha-weighted mixture of per-slot predictions. The alphas are the slots' soft segmentation.

Training signal:
- feature reconstruction (self-supervised; the part that will remain when supervision is removed)
- mask supervision (rule 2: privileged first): slots are assigned to ground-truth labels (a fixed
  slot per label, or the Hungarian matching), then every patch's distribution over slots (the
  decoder alphas *and* the last slot-attention map) is pulled towards the patch's label fractions
  (cross-entropy), plus a Dice term per label so that a 4-patch block counts as much as the
  800-patch table
- token readout: from each matched slot, predict that object's entity token (position, velocity,
  yaw, kind, colour), so slots plug into the world model and planner already built on tokens

Why it is built this way (a first version learned nothing about objects: object IoU 0.06):
- Objects cover ~4 of 1024 patches. With random slot initialisation and IoU matching, every slot
  was a ~120-patch piece of table, the best slot for a block overlapped it by 1-3 %, the matching
  was arbitrary, and the Dice term (1 - 2*4/(120+4)) had almost no gradient. Here each slot has its
  own learned initial query (as in BO-QSA) and is tied to one label, so it keeps specialising on
  the same object (see `assign_slots`); the Hungarian cost, when used, is the loss itself (as in
  DETR); the attention map is supervised directly.
- With ~35 training episodes the model overfitted (training object IoU 0.68, held-out 0.40): the
  mask logit now includes the slot's attention logit on the patch's own features, and positions
  enter as smooth Fourier features instead of a free embedding per patch.
- A 3 x 3 convolution over the patch grid gives each patch its neighbours, which sharpens the
  boundaries of objects smaller than a patch.
- Position is read out geometrically: the slot's mask centroid on the patch grid goes through a
  learned image -> table mapping (a homography per object kind, see `TableMapping`). Read out
  from the slot vector, positions were memorised (held-out error 41 mm, median 18 mm).

Object identity over time (the concern SlotContrast addresses, see docs/LITERATURE.md) comes from
the fixed slot per object. `forward(init=...)` can start from the previous frame's slots, but these
models are trained from the learned queries only, and without such training the slots drift.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from pai.world.entities import KIND, KINDS, POS, TOKEN_DIM

N_GEOM = 7  # mask centroid (2), spread (2), log area (1), centroid of the mask's peak (2)


class SlotAttention(nn.Module):
    def __init__(self, n_slots: int, dim: int, iters: int = 3, hidden: int = 256):
        super().__init__()
        self.n_slots, self.iters, self.scale = n_slots, iters, dim**-0.5
        self.mu = nn.Parameter(torch.randn(1, n_slots, dim) * 0.5)  # one learned query per slot
        self.norm_in, self.norm_slots, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def forward(self, inputs: torch.Tensor, init: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs (B, P, D) -> slots (B, K, D), log attention (B, K, P) (log-softmax over slots)."""
        B, _, D = inputs.shape
        x = self.norm_in(inputs)
        k, v = self.k(x), self.v(x)
        slots = self.mu.expand(B, -1, -1) if init is None else init
        for _ in range(self.iters):
            prev = slots
            q = self.q(self.norm_slots(slots))
            log_attn = torch.log_softmax(torch.einsum("bkd,bpd->bkp", q, k) * self.scale, dim=1)  # slots compete
            attn = log_attn.exp()
            weights = attn / (attn.sum(-1, keepdim=True) + 1e-8)
            updates = torch.einsum("bkp,bpd->bkd", weights, v)
            slots = self.gru(updates.reshape(-1, D), prev.reshape(-1, D)).reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, log_attn


def patch_grid(n_patches: int, device=None) -> torch.Tensor:
    """(P, 2) row, column of every patch, scaled to [0, 1]."""
    side = int(round(math.sqrt(n_patches)))
    r, c = torch.meshgrid(torch.arange(side, device=device), torch.arange(side, device=device), indexing="ij")
    return torch.stack([r, c], -1).reshape(-1, 2).float() / max(1, side - 1)


def mask_geometry(alpha: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """alpha (B, K, P) -> (B, K, N_GEOM) per-slot geometric features on the patch grid.

    The centroid weights are alpha^2: a small object's slot also leaks a little alpha onto
    hundreds of table patches, which would drag a plain alpha-weighted mean to the image centre.
    alpha^8 keeps only the mask's core (the visible top of a block, not its fringe).
    Always computed in float32: under fp16 autocast (T4) alpha^8 underflows for weak slots."""
    with torch.autocast(device_type=alpha.device.type, enabled=False):
        return _mask_geometry(alpha.float(), grid.float())


def _mask_geometry(alpha: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    w = alpha.square()
    area = w.sum(-1, keepdim=True) + 1e-6
    mean = torch.einsum("bkp,pd->bkd", w, grid) / area
    var = torch.einsum("bkp,pd->bkd", w, grid.square()) / area - mean.square()
    w8 = w.pow(4)
    peak = torch.einsum("bkp,pd->bkd", w8, grid) / (w8.sum(-1, keepdim=True) + 1e-12)
    return torch.cat([mean, 10 * var.clamp(min=1e-6).sqrt(), torch.log(alpha.sum(-1, keepdim=True) + 1e-3), peak], -1)


def fourier_features(grid: torch.Tensor, n_freqs: int = 8) -> torch.Tensor:
    """(P, 2) coordinates in [0, 1] -> (P, 4 n_freqs) sin/cos features, periods from 2 down to 1/64 of the grid."""
    freqs = math.pi * 2.0 ** torch.arange(n_freqs, device=grid.device, dtype=grid.dtype)
    x = grid[:, :, None] * freqs                                           # (P, 2, n_freqs)
    return torch.cat([x.sin(), x.cos()], -1).flatten(1)


class TableMapping(nn.Module):
    """Learned image -> table mapping: patch-grid centroid (u, v) -> position (normalised units).

    A camera sees a plane through a homography, so x, y are a projective function of (u, v), and
    the height of a resting object depends only on its kind. One such map per kind (the visible top
    of a block and the rim of a bowl are at different heights), mixed by the predicted kind. A small
    MLP (initialised to zero) adds what the plane cannot explain (tilted or occluded objects).
    Compared with an MLP alone on the same inputs, on 40 unseen episodes: median error 5.3 vs 6.1 mm,
    objects on the table 9.7 vs 10.8 mm; lifted objects are worse (142 vs 104 mm): a block in the
    gripper is off the plane, and its height cannot be told from its image position alone."""

    def __init__(self, n_geom: int = N_GEOM, n_kinds: int = len(KINDS), hidden: int = 128):
        super().__init__()
        self.affine = nn.Parameter(torch.zeros(n_kinds, 3, 3))   # rows: x, y, z from (u, v, 1)
        self.persp = nn.Parameter(torch.zeros(n_kinds, 2))       # projective denominator 1 + g . (u, v)
        self.gate = nn.Linear(n_kinds, n_kinds)                  # predicted kind -> mixing weights
        self.residual = nn.Sequential(nn.Linear(n_geom + n_kinds, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, geom: torch.Tensor, kind: torch.Tensor) -> torch.Tensor:
        """geom (B, K, N_GEOM) from `mask_geometry`, kind (B, K, n_kinds) readout -> (B, K, 3)."""
        uv = geom[..., :2]
        uv1 = torch.cat([uv, torch.ones_like(uv[..., :1])], -1)                   # (B, K, 3)
        lin = torch.einsum("nij,bkj->bkni", self.affine, uv1)                     # (B, K, n_kinds, 3)
        den = 1 + torch.einsum("nj,bkj->bkn", self.persp, uv)[..., None]
        per_kind = torch.cat([lin[..., :2] / den.clamp_min(0.1), lin[..., 2:]], -1)
        w = torch.softmax(self.gate(kind), -1)[..., None]
        return (w * per_kind).sum(-2) + self.residual(torch.cat([geom, kind], -1))


class SlotModel(nn.Module):
    def __init__(self, feat_dim: int = 384, n_patches: int = 256, n_slots: int = 8, dim: int = 128, iters: int = 3):
        super().__init__()
        self.side = int(round(math.sqrt(n_patches)))
        assert self.side**2 == n_patches, "patch features must form a square grid"
        self.encoder = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.context = nn.Conv2d(dim, dim, 3, padding=1)  # neighbouring patches, for sub-patch objects
        # Positions enter as smooth Fourier features, not a free embedding per patch: with a free
        # embedding the model memorised where objects are in the training scenes.
        self.register_buffer("fourier", fourier_features(patch_grid(n_patches)), persistent=False)
        self.pos = nn.Linear(self.fourier.shape[-1], dim)
        self.slot_attention = SlotAttention(n_slots, dim, iters)
        # DINOSAUR-style MLP decoder: each slot + position -> (patch feature, alpha logit)
        self.dec_pos = nn.Linear(self.fourier.shape[-1], dim)
        self.decoder = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                     nn.Linear(512, feat_dim + 1))
        self.readout = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, TOKEN_DIM))
        # position from mask geometry and kind only. Not from the slot vector: from it the head
        # memorises the training scenes (held-out error 41 mm).
        self.pos_head = TableMapping()
        self.register_buffer("grid", patch_grid(n_patches), persistent=False)
        self._config = {"feat_dim": feat_dim, "n_patches": n_patches, "n_slots": n_slots, "dim": dim, "iters": iters}

    def config(self) -> dict:
        return dict(self._config)

    def forward(self, feats: torch.Tensor, init: torch.Tensor | None = None) -> dict:
        """feats (B, P, feat_dim) -> dict(slots, alpha (B, K, P), log_alpha, log_attn, recon (B, P, feat_dim),
        tokens (B, K, TOKEN_DIM) in normalised token units)."""
        B, P, _ = feats.shape
        x = self.encoder(feats)
        grid_x = x.transpose(1, 2).reshape(B, -1, self.side, self.side)
        x = x + self.context(grid_x).flatten(2).transpose(1, 2) + self.pos(self.fourier)
        slots, log_attn = self.slot_attention(x, init)
        out = self.decoder(slots[:, :, None, :] + self.dec_pos(self.fourier))            # (B, K, P, feat_dim + 1)
        # The mask logit adds the slot's attention logit on the patch's own features to the decoder's:
        # the decoder alone sees only slot + position and must memorise shapes (it overfits: held-out
        # object IoU 0.40 vs 0.53 for the attention map), the features know where the object is.
        log_alpha = torch.log_softmax(out[..., -1].float() + log_attn.float(), dim=1)   # (B, K, P)
        alpha = log_alpha.exp()
        recon = (alpha[..., None] * out[..., :-1]).sum(1)                   # (B, P, feat_dim)
        tokens = self.readout(slots).float()
        pos = self.pos_head(mask_geometry(alpha, self.grid), tokens[..., KIND].detach()).float()
        tokens = torch.cat([pos, tokens[..., POS.stop:]], -1)
        return {"slots": slots, "alpha": alpha, "log_alpha": log_alpha, "log_attn": log_attn.float(),
                "recon": recon, "tokens": tokens}


def _match_cost(log_p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """log_p (B, K, P) log masks, target (B, P, L) label fractions -> (B, K, L) cost of slot k for label l:
    the per-label cross-entropy (mean over the label's patches) + Dice loss."""
    p = log_p.exp()
    area = target.sum(1)[:, None, :]                                        # (B, 1, L)
    ce = -torch.einsum("bkp,bpl->bkl", log_p, target) / (area + 1.0)
    inter = torch.einsum("bkp,bpl->bkl", p, target)
    dice = 1 - (2 * inter + 1) / (p.sum(-1)[:, :, None] + area + 1)
    return ce + dice


def match_slots(alpha: torch.Tensor, target: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hungarian matching of slots to labels.

    alpha (B, K, P) slot masks; target (B, P, L) label fractions -> per batch item (slot_idx, label_idx)."""
    cost = _match_cost(alpha.detach().float().clamp_min(1e-6).log(), target.detach().float()).cpu().numpy()
    return [linear_sum_assignment(c) for c in cost]


def assign_slots(alpha: torch.Tensor, target: torch.Tensor, mode: str = "hungarian") -> torch.Tensor:
    """(B, L) index of the slot assigned to each label.

    "hungarian": the cheapest one-to-one matching per frame (slots are interchangeable).
    "fixed": slot l is label l in every frame. Each object then has a slot of its own that learns to
    look for that object's appearance. With interchangeable slots the four blocks shared four
    generic "block" slots, and two touching blocks were often taken by one slot while the other
    slot stayed empty (61 % of the held-out misses were one block taken by another block's slot)."""
    B, L = target.shape[0], target.shape[-1]
    if mode == "fixed":
        assert alpha.shape[1] >= L, "fixed assignment needs a slot per label"
        return torch.arange(L, device=alpha.device).expand(B, L)
    perm = torch.zeros(B, L, dtype=torch.long, device=alpha.device)
    for b, (ks, ls) in enumerate(match_slots(alpha, target)):
        perm[b, torch.as_tensor(ls, device=alpha.device)] = torch.as_tensor(ks, device=alpha.device)
    return perm


def _seg_loss(log_p: torch.Tensor, target: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Per-patch cross-entropy over slots + Dice per label, for matched slots.
    log_p (B, K, P); target (B, P, L); perm (B, L) slot matched to each label."""
    lp = torch.gather(log_p, 1, perm[:, :, None].expand(-1, -1, log_p.shape[-1]))   # (B, L, P)
    t = target.transpose(1, 2)                                                        # (B, L, P)
    ce = -(t * lp).sum(1).mean()
    p = lp.exp()
    dice = 1 - (2 * (p * t).sum(-1) + 1) / (p.sum(-1) + t.sum(-1) + 1)
    return ce + dice.mean()


def slot_losses(out: dict, feats: torch.Tensor, target_masks: torch.Tensor, target_tokens: torch.Tensor | None,
                object_labels: slice, mask_weight: float = 1.0, token_weight: float = 1.0,
                recon_weight: float = 1.0, attn_weight: float = 1.0, pos_weight: float = 1.0,
                assignment: str = "hungarian") -> dict:
    """Reconstruction + (optional) mask and token supervision on matched slots.

    target_masks (B, P, L) label fractions, labels 0 = background, 1 = robot, 2.. = objects.
    target_tokens (B, n_objects + 1, TOKEN_DIM) normalised entity tokens (gripper first), for the object labels.
    Features are compared after scaling by their overall std, so the loss does not depend on the PCA scale.
    """
    scale = feats.detach().std() + 1e-6
    losses = {"recon": recon_weight * F.mse_loss(out["recon"] / scale, feats / scale)}
    if mask_weight > 0:
        perm = assign_slots(out["alpha"], target_masks, assignment)
        losses["mask"] = mask_weight * _seg_loss(out["log_alpha"], target_masks, perm)
        if attn_weight > 0:
            losses["attn"] = attn_weight * mask_weight * _seg_loss(out["log_attn"], target_masks, perm)
        if target_tokens is not None:
            obj = perm[:, object_labels]                                                 # (B, n_obj)
            pred = torch.gather(out["tokens"], 1, obj[:, :, None].expand(-1, -1, TOKEN_DIM))
            tgt = target_tokens[:, 1 : 1 + obj.shape[1]]
            # only objects that are visible (a hidden object's position cannot be seen)
            vis = (target_masks[:, :, object_labels].sum(1) > 0.5).float()[..., None]
            n = vis.sum().clamp_min(1.0)
            losses["token"] = token_weight * (((pred[..., POS.stop:] - tgt[..., POS.stop:]) ** 2) * vis).sum() / (
                n * (TOKEN_DIM - POS.stop))
            # position: L1 in normalised units, so centimetre errors still have a gradient
            losses["pos"] = pos_weight * ((pred[..., POS] - tgt[..., POS]).abs() * vis).sum() / (n * 3)
    losses["total"] = sum(losses.values())
    return losses
