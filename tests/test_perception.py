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


# ---------------------------------------------------------------------------------------------
# Pixel geometry (pai.perception.slot_tokens.PixelGeometry): objects placed from their own pixels
# ---------------------------------------------------------------------------------------------

import mujoco  # noqa: E402
import pytest  # noqa: E402

from pai.config import REPO_ROOT, load_config  # noqa: E402
from pai.perception.slot_tokens import PixelGeometry, SlotPerception  # noqa: E402


@pytest.fixture(scope="module")
def work_env():
    from pai.envs import TabletopEnv
    cfg = load_config(REPO_ROOT / "configs" / "tabletop.yaml", ["env.image_size=448", "env.render_images=false"])
    env = TabletopEnv(cfg.env, disturbances=[])
    yield env
    env.close()


def _truth(env, obs):
    return np.stack([obs["state"][n]["pos"] for n in env.objects])


def test_projection_matches_the_rendered_objects(work_env):
    env = work_env
    obs = env.reset(seed=30001)
    g = PixelGeometry(env, "work", 448)
    seg = env.render_segmentation("work")
    for o, x in enumerate(_truth(env, obs)[:4]):          # the blocks
        v, u = np.nonzero(seg == 2 + o)
        assert np.abs(g.project(x) - [u.mean() + 0.5, v.mean() + 0.5]).max() < 1.5
        back = g.on_plane(u.mean() + 0.5, v.mean() + 0.5, x[2])
        assert np.linalg.norm(back - x) < 0.002


def test_blocks_on_the_table_are_placed_to_a_few_mm(work_env):
    env = work_env
    g = PixelGeometry(env, "work", 448)
    for seed in (30002, 30003):
        obs = env.reset(seed=seed)
        g.label(env.render("work"))
        for o, x in enumerate(_truth(env, obs)[:4]):
            b = g.blob(o, g.project(x + [0.03, -0.02, 0]), 70)   # found near a rough prior
            c = g.on_plane(b[0].mean(), b[1].mean(), g.rest_z[o])
            assert np.linalg.norm((c - x)[:2]) < 0.003, (seed, o)
        # the plate (white, as parts of the robot) and the bowl are fitted by silhouette
        for o in (4, 5):
            x = _truth(env, obs)[o]
            p = g.fit(o, np.r_[x[:2] + [0.01, 0.01], g.rest_z[o]], 0.0, "xy",
                      steps=(0.008, 0.004, 0.002, 0.001, 0.0005))
            assert np.linalg.norm((p - x)[:2]) < 0.01, (seed, o)


def test_a_partly_hidden_block_is_fitted_not_dragged(work_env):
    """Half of the block hidden (grey, like the hand): the centroid shifts, the silhouette fit does not."""
    env = work_env
    obs = env.reset(seed=30004)
    g = PixelGeometry(env, "work", 448)
    img = env.render("work").copy()
    x = _truth(env, obs)[0]
    yaw = float(np.arctan2(*_yaw_sc(obs["state"]["red"]["quat"])))
    u0, v0 = g.project(x)
    img[: int(v0), int(u0) - 40: int(u0) + 40] = 150          # a grey occluder over the top half
    g.label(img)
    b = g.blob(0, g.project(x), 70)
    c = g.on_plane(b[0].mean(), b[1].mean(), g.rest_z[0])
    p = g.fit(0, c, yaw, "xy")
    assert np.linalg.norm((c - x)[:2]) > 0.004                # the partial view drags the centroid
    assert np.linalg.norm((p - x)[:2]) < 0.002


def _yaw_sc(quat):
    w, x, y, z = quat
    return 2 * (w * z + x * y), 1 - 2 * (y * y + z * z)


def test_height_of_a_lifted_block_is_measured_along_the_vertical(work_env):
    env = work_env
    obs = env.reset(seed=30005)
    g = PixelGeometry(env, "work", 448)
    adr = env.obj_qpos["red"]
    for z in (0.06, 0.12):
        env.data.qpos[adr + 2] = z
        mujoco.mj_forward(env.model, env.data)
        x = env.data.qpos[adr: adr + 3].copy()
        yaw = float(np.arctan2(*_yaw_sc(env.data.qpos[adr + 3: adr + 7])))
        g.label(env.render("work"))
        p = g.fit(0, np.r_[x[:2], z + 0.03], yaw, "z", steps=(0.004, 0.002, 0.001, 0.0005))
        assert abs(p[2] - z) < 0.003


def test_camera_bump_reaches_the_perception_camera(work_env):
    from pai.envs import build_disturbances
    env = work_env
    sp = object.__new__(SlotPerception)       # only the camera logic, no slot checkpoint needed
    sp.env, sp.camera, sp.bump_follows = env, "work", True
    env.disturbances = build_disturbances([{"type": "camera_shift", "start": 0, "duration": -1,
                                            "offset": [0.04, -0.02, 0.0]}])
    env.reset(seed=30006)
    env.step(np.zeros(5))
    sp._follow_bump()
    c = env.model.camera("work").id
    assert np.allclose(env.model.cam_pos[c] - env._nominal["cam_pos"][c], [0.04, -0.02, 0.0])
    env.disturbances = build_disturbances([])
    env.reset(seed=30006)                      # a reset restores every camera
    assert np.allclose(env.model.cam_pos[c], env._nominal["cam_pos"][c])
