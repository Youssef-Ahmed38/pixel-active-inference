import itertools
import math

import numpy as np
import torch

from pai.thinker import CAUSES, HRM, CauseTask, FixedDepthTransformer, MLPThinker, TaskConfig
from pai.thinker.multicause import CAUSE_LIB, N_LIB, MultiCauseConfig, MultiCauseTask, _PAIR_PARTNER
from pai.thinker.synthetic import _template
from pai.thinker.train import build_model, multicause_loss, thinker_loss, train_thinker

TINY = dict(d_model=32, n_heads=2, h_layers=1, l_layers=1, h_cycles=2, l_steps=2)


def _task(**kw) -> CauseTask:
    return CauseTask(TaskConfig(horizon=16, min_length=4, onset_range=(2, 8), **kw))


def _hrm(task: CauseTask, max_segments: int = 4) -> HRM:
    return HRM(task.n_channels, task.ctx_dim, task.cfg.horizon, {"cause": task.n_causes, "plan": 3},
               max_segments=max_segments, **TINY)


def test_sample_shapes_and_teacher_normalised():
    task = _task()
    b = task.sample(64, torch.Generator().manual_seed(0), heldout_prob=0.2)
    assert b["x"].shape == (64, 16, task.n_channels) and b["mask"].shape == (64, 16)
    assert b["teacher"].shape == (64, len(CAUSES))
    assert torch.allclose(b["teacher"].sum(-1), torch.ones(64), atol=1e-5)
    assert (b["x"][b["mask"] == 0] == 0).all()  # nothing observed beyond the evidence length
    assert (b["y"] == len(CAUSES)).any()  # held-out cause is generated with label K


def test_teacher_exact_on_noise_free_example():
    task = _task(overlap=0.5)
    for k in range(len(CAUSES)):
        x = task.bank[k, 7][None]  # onset 5 < horizon, so every non-"none" cause is visible
        post = task.posterior(x, torch.tensor([16]), torch.tensor([0.05]))
        assert post[0, k] > 0.999, (CAUSES[k], post)


def test_teacher_matches_brute_force():
    task = _task()
    b = task.sample(8, torch.Generator().manual_seed(1))
    mask = b["mask"][:, None, None, :, None]
    sq = ((b["x"][:, None, None] - task.bank[None] * mask) ** 2).sum((-1, -2))  # [B, K, G]
    ll = -0.5 * sq / b["sigma"][:, None, None] ** 2
    ref = torch.logsumexp(ll, -1).softmax(-1)
    assert torch.allclose(ref, b["teacher"], atol=1e-4)


def test_hrm_shapes_and_halting_respects_max():
    task = _task()
    model = _hrm(task, max_segments=3)
    b = task.sample(32, torch.Generator().manual_seed(0))
    outs = model.forward_segments(b["x"], b["mask"], b["ctx"])
    assert len(outs) == 3 and outs[0].logits["cause"].shape == (32, 6) and outs[0].logits["plan"].shape == (32, 3)
    assert outs[0].q.shape == (32, 2)
    with torch.no_grad():  # force every example to want to halt immediately / never
        for bias, expect in [((5.0, -5.0), 1), ((-5.0, 5.0), 2)]:
            model.q_head.bias.copy_(torch.tensor(bias))
            res = model.think(b["x"], b["mask"], b["ctx"], max_segments=2)
            assert (res.segments == expect).all()
    res = model.think(b["x"], b["mask"], b["ctx"], halt=False)
    assert (res.segments == 3).all() and res.logits["cause"].shape == (32, 6)
    # With halting off, think() must reproduce the training pass's last segment.
    assert torch.allclose(res.logits["cause"], outs[-1].logits["cause"], atol=1e-4)


def test_one_step_gradient_reaches_all_parts():
    task = _task()
    model = _hrm(task)
    b = task.sample(16, torch.Generator().manual_seed(0))
    loss, _ = thinker_loss(model.forward_segments(b["x"], b["mask"], b["ctx"]), b)
    loss.backward()
    for name in ["encoder.proj.weight", "L.blocks.0.mlp.down.weight", "H.blocks.0.attn.qkv.weight", "q_head.bias"]:
        assert model.get_parameter(name).grad.abs().sum() > 0, name


def test_baselines_interface():
    task = _task()
    b = task.sample(8, torch.Generator().manual_seed(0))
    for model in [FixedDepthTransformer(5, 2, 16, {"cause": 6}, d_model=32, n_heads=2, n_layers=2),
                  MLPThinker(5, 2, 16, {"cause": 6}, hidden=32, n_layers=2)]:
        res = model.think(b["x"], b["mask"], b["ctx"])
        assert res.probs().shape == (8, 6) and (res.segments == 1).all()


def test_tiny_training_reduces_loss():
    torch.manual_seed(0)
    task = _task()
    model = _hrm(task, max_segments=2)
    fixed = task.sample(256, torch.Generator().manual_seed(99))

    def eval_loss() -> float:
        with torch.no_grad():
            return thinker_loss(model.forward_segments(fixed["x"], fixed["mask"], fixed["ctx"]), fixed)[0].item()

    before = eval_loss()
    tcfg = dict(batch_size=128, steps=60, lr=3e-3, warmup_steps=5, weight_decay=0.0, log_every=1000, amp="off")
    train_thinker(model, task, tcfg, torch.device("cpu"))
    assert eval_loss() < 0.8 * before


# --------------------------------------------------------------------------------------------
# Multi-cause task (0-3 simultaneous causes, exact subset teacher, explaining away)
# --------------------------------------------------------------------------------------------

def _mc_task(**kw) -> MultiCauseTask:
    cfg = dict(horizon=10, min_length=4, max_active=2, onset_grid=(2, 6), amplitude_grid=(1.0,), overlap=0.4)
    cfg.update(kw)
    return MultiCauseTask(MultiCauseConfig(**cfg))


def test_multicause_sample_shapes_and_teacher_normalised():
    task = _mc_task()
    b = task.sample(32, torch.Generator().manual_seed(0))
    assert b["x"].shape == (32, 10, task.n_channels) and b["mask"].shape == (32, 10)
    assert b["teacher_subset"].shape == (32, 64) and b["teacher_marginal"].shape == (32, N_LIB)
    assert torch.allclose(b["teacher_subset"].sum(-1), torch.ones(32), atol=1e-4)
    assert (b["n_active"] <= 2).all() and (b["n_active"] >= 0).all()
    assert (b["y_multihot"].sum(-1) == b["n_active"].float()).all()
    assert (b["x"][b["mask"] == 0] == 0).all()


def test_multicause_teacher_exact_brute_force():
    """Independently recompute log p(subset | x) by brute force (its own loop over masks and the
    nuisance grid, not the class's vectorised/scatter implementation) and compare."""
    task = _mc_task()
    cfg = task.cfg
    b = task.sample(6, torch.Generator().manual_seed(1))
    for n in range(6):
        x = b["x"][n].numpy()
        length, sigma = int(b["length"][n]), float(b["sigma"][n])
        log_joint = {}
        for mask in range(64):
            active = [i for i in range(N_LIB) if mask & (1 << i)]
            s = len(active)
            if s > cfg.max_active:
                continue
            lls = []
            for obj in (0, 1):
                for amp in cfg.amplitude_grid:
                    for taus in itertools.product(cfg.onset_grid, repeat=s):
                        m = np.zeros((cfg.horizon, task.n_channels))
                        for ci, tau in zip(active, taus):
                            cause = CAUSE_LIB[ci]
                            t1 = _template(cause, tau, amp, obj, cfg.horizon)
                            partner = _PAIR_PARTNER.get(cause)
                            if partner:
                                t2 = _template(partner, tau, amp, obj, cfg.horizon)
                                t1 = t1 + 0.5 * cfg.overlap * (t2 - t1)
                            m += t1
                        xm, mm = x[:length], m[:length]
                        lls.append((np.sum(xm * mm) - 0.5 * np.sum(mm ** 2)) / sigma ** 2)
            mx = max(lls)
            avg_ll = mx + math.log(sum(math.exp(v - mx) for v in lls)) - math.log(len(lls))
            prior = -math.log(cfg.max_active + 1) - math.log(math.comb(N_LIB, s))
            log_joint[mask] = avg_ll + prior
        mx = max(log_joint.values())
        denom = sum(math.exp(v - mx) for v in log_joint.values())
        for mask, v in log_joint.items():
            expect = math.exp(v - mx) / denom
            assert abs(b["teacher_subset"][n, mask].item() - expect) < 1e-3, (n, mask)


def test_multicause_marginal_consistent_with_subset_posterior():
    task = _mc_task()
    b = task.sample(16, torch.Generator().manual_seed(2))
    subset = b["teacher_subset"]
    for i in range(N_LIB):
        bits = torch.tensor([1.0 if (mask & (1 << i)) else 0.0 for mask in range(64)])
        expect = (subset * bits).sum(-1)
        assert torch.allclose(expect, b["teacher_marginal"][:, i], atol=1e-5)


def test_multicause_model_interface_and_training_reduces_loss():
    torch.manual_seed(0)
    task = _mc_task()
    model = build_model("hrm", task, {"hrm": dict(**TINY, max_segments=2)})
    fixed = task.sample(128, torch.Generator().manual_seed(9))

    def eval_loss() -> float:
        with torch.no_grad():
            return multicause_loss(model.forward_segments(fixed["x"], fixed["mask"], fixed["ctx"]), fixed)[0].item()

    before = eval_loss()
    tcfg = dict(batch_size=64, steps=60, lr=3e-3, warmup_steps=5, weight_decay=0.0, log_every=1000, amp="off")
    train_thinker(model, task, tcfg, torch.device("cpu"), loss_fn=multicause_loss)
    assert eval_loss() < 0.8 * before

    for kind in ["transformer", "mlp"]:
        m = build_model(kind, task, {"transformer": dict(d_model=32, n_heads=2, n_layers=2),
                                     "mlp": dict(hidden=32, n_layers=2)})
        res = m.think(fixed["x"], fixed["mask"], fixed["ctx"])
        assert res.logits["subset"].shape == (128, 64) and res.logits["cause_marginal"].shape == (128, N_LIB)
