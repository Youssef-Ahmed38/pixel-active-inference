"""Object slots: shapes, slot assignment, geometric position readout, and that supervision is learnable."""

import numpy as np
import torch

from pai.perception.slots import SlotModel, assign_slots, mask_geometry, match_slots, patch_grid, slot_losses
from pai.world.entities import TOKEN_DIM


def _scene(B=4, side=8, L=5, C=16, seed=0):
    """Features that name each patch's label (plus noise), one-patch objects at random places."""
    g = torch.Generator().manual_seed(seed)
    P = side * side
    labels = torch.zeros(B, P, dtype=torch.long)
    for b in range(B):
        where = torch.randperm(P, generator=g)[: L - 1]
        labels[b, where] = torch.arange(1, L)
    masks = torch.nn.functional.one_hot(labels, L).float()
    code = torch.randn(L, C, generator=torch.Generator().manual_seed(123))  # the same looks in every scene
    feats = code[labels] + 0.1 * torch.randn(B, P, C, generator=g)
    return feats, masks


def test_slot_model_shapes_and_masks():
    for side in (8, 16):
        m = SlotModel(feat_dim=16, n_patches=side * side, n_slots=6, dim=32, iters=2)
        out = m(torch.randn(3, side * side, 16))
        assert out["alpha"].shape == (3, 6, side * side) and out["tokens"].shape == (3, 6, TOKEN_DIM)
        assert torch.allclose(out["alpha"].sum(1), torch.ones(3, side * side), atol=1e-5)
        assert torch.allclose(out["log_attn"].exp().sum(1), torch.ones(3, side * side), atol=1e-5)
        out2 = m(torch.randn(3, side * side, 16), init=out["slots"])  # temporal initialisation
        assert out2["slots"].shape == out["slots"].shape


def test_mask_geometry_centroid_ignores_leakage():
    grid = patch_grid(32 * 32)
    alpha = torch.full((1, 1, 1024), 0.005)  # a little alpha leaked onto every patch (5 patches in total)
    target = 5 * 32 + 20
    alpha[0, 0, target] = 0.95
    geom = mask_geometry(alpha, grid)
    assert torch.allclose(geom[0, 0, :2], grid[target], atol=0.05)   # alpha^2 centroid
    assert torch.allclose(geom[0, 0, 5:7], grid[target], atol=1e-3)  # peak centroid


def test_assignment_fixed_and_hungarian():
    _, masks = _scene()
    alpha = torch.cat([masks.transpose(1, 2)[:, [3, 0, 4, 1, 2]], torch.zeros(4, 1, 64)], 1) * 0.98 + 0.002
    perm = assign_slots(alpha, masks, "hungarian")
    assert perm.tolist() == [[1, 3, 4, 0, 2]] * 4
    assert assign_slots(alpha, masks, "fixed").tolist() == [[0, 1, 2, 3, 4]] * 4
    ks, ls = match_slots(alpha, masks)[0]
    assert dict(zip(ls.tolist(), ks.tolist())) == {0: 1, 1: 3, 2: 4, 3: 0, 4: 2}


def test_losses_are_finite_and_reach_the_position_head():
    feats, masks = _scene()
    m = SlotModel(feat_dim=16, n_patches=64, n_slots=6, dim=32, iters=2)
    tokens = torch.randn(4, 4, TOKEN_DIM)  # gripper + 3 objects (labels 2..4)
    for mode in ("fixed", "hungarian"):
        m.zero_grad()
        losses = slot_losses(m(feats), feats, masks, tokens, slice(2, 5), assignment=mode)
        assert all(torch.isfinite(v) for v in losses.values()) and {"mask", "attn", "pos", "token"} <= set(losses)
        losses["total"].backward()
        assert m.pos_head.affine.grad.abs().sum() > 0 and m.slot_attention.mu.grad.abs().sum() > 0


def test_mask_supervision_is_learned():
    """A few hundred steps on a toy scene: each object's slot finds its one-patch object."""
    torch.manual_seed(0)
    m = SlotModel(feat_dim=16, n_patches=64, n_slots=5, dim=32, iters=2)
    opt = torch.optim.Adam(m.parameters(), 2e-3)
    for step in range(300):
        feats, masks = _scene(B=8, seed=step)
        losses = slot_losses(m(feats), feats, masks, None, slice(2, 5), recon_weight=0.1, assignment="fixed")
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
    feats, masks = _scene(B=8, seed=10_000)  # unseen scenes
    with torch.no_grad():
        a = m(feats)["alpha"]
    t = masks.transpose(1, 2)[:, 2:5]
    inter = (a[:, 2:5] * t).sum(-1)
    iou = inter / (a[:, 2:5].sum(-1) + t.sum(-1) - inter)
    assert float(iou.mean()) > 0.5, float(iou.mean())
