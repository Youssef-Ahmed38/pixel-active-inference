import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pai.causes.inference import CAUSES, CHANNELS
from pai.thinker.real import RealConfig, RealEvidence, load_evidence, write_fake_evidence
from pai.thinker.train import build_model, thinker_loss

ROOT = Path(__file__).resolve().parent.parent
TINY = {"hrm": dict(d_model=32, n_heads=2, h_layers=1, l_layers=1, h_cycles=2, l_steps=2, max_segments=2),
        "transformer": dict(d_model=32, n_heads=2, n_layers=2)}


@pytest.fixture(scope="module")
def fake(tmp_path_factory) -> Path:
    return write_fake_evidence(tmp_path_factory.mktemp("ev") / "fake_evidence.npz", n_episodes=24, seed=0)


def test_fake_file_matches_contract(fake):
    d = load_evidence(fake)
    s, e = d["z"].shape[0], len(d["labels"])
    assert d["z"].dtype == np.float32 and d["z"].shape[1] == len(d["channels"])
    assert tuple(d["channels"]) == CHANNELS and tuple(d["causes"]) == CAUSES and set(d["labels"]) == set(CAUSES)
    assert d["holding"].dtype == bool and d["holding"].shape == d["phase"].shape == d["episode"].shape == (s,)
    assert d["teacher"].shape == (e, len(d["causes"])) and d["onset"].shape == (e,)
    assert np.allclose(d["teacher"].sum(-1), 1, atol=1e-5)
    assert (np.diff(d["episode"]) >= 0).all() and set(d["episode"].tolist()) == set(range(e))


def test_dataset_shapes(fake):
    cfg = RealConfig(window=20, n_phase=10)
    data = RealEvidence(fake, cfg)
    assert data.n_channels == len(CHANNELS) + 1 + 10 and data.n_causes == len(CAUSES) and data.n_episodes == 24
    b = data.sample(16, torch.Generator().manual_seed(0))
    assert b["x"].shape == (16, 20, data.n_channels) and b["mask"].shape == (16, 20) and b["ctx"].shape == (16, 2)
    assert b["teacher"].shape == (16, data.n_causes) and b["y"].shape == (16,)
    assert (b["x"][b["mask"] == 0] == 0).all()
    assert set(b["episode"].tolist()) <= set(data.splits["train"].tolist())
    t = data.split("test")
    assert t["x"].shape == (len(data.splits["test"]), 20, data.n_channels)
    # Canonical window = the teacher's: peak - window + after, clipped at 0.
    assert torch.equal(data.start, (data.peak - 15).clamp_min(0))


def test_peak_in_jittered_window(fake):
    data = RealEvidence(fake, RealConfig(jitter=5))
    gen = torch.Generator().manual_seed(1)
    for _ in range(5):
        b = data.sample(64, gen)
        ep = b["episode"]
        # the peak's features sit at position (peak - start); recover start by matching the row
        peak_feat = data.feats[ep, data.peak[ep]]
        hit = (b["x"] == peak_feat[:, None]).all(-1).any(-1)
        assert hit.all()


def test_split_has_no_episode_overlap_and_is_seeded(fake):
    data = RealEvidence(fake, RealConfig(), seed=3)
    sp = {k: set(v.tolist()) for k, v in data.splits.items()}
    assert not (sp["train"] & sp["val"]) and not (sp["train"] & sp["test"]) and not (sp["val"] & sp["test"])
    assert sp["train"] | sp["val"] | sp["test"] == set(range(data.n_episodes))
    assert all(len(v) > 0 for v in sp.values())
    other = RealEvidence(fake, RealConfig(), seed=3)
    assert all(torch.equal(data.splits[k], other.splits[k]) for k in data.splits)
    data.set_split(4)
    assert any(set(data.splits[k].tolist()) != sp[k] for k in sp)


def test_model_on_real_batch(fake):
    data = RealEvidence(fake, RealConfig())
    b = data.sample(8, torch.Generator().manual_seed(0))
    for kind in ["hrm", "transformer"]:
        model = build_model(kind, data, TINY)
        loss, _ = thinker_loss(model.forward_segments(b["x"], b["mask"], b["ctx"]), b)
        assert torch.isfinite(loss)


def test_script_end_to_end(fake, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("train_thinker_real", ROOT / "scripts" / "train_thinker_real.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    overrides = [f"out_dir={tmp_path.as_posix()}", "seeds=2", "train.steps=6", "train.log_every=3",
                 "train.warmup_steps=2", "train.batch_size=16",
                 "model.hrm={d_model: 32, n_heads: 2, h_layers: 1, l_layers: 1, h_cycles: 2, l_steps: 2, max_segments: 2, patch: 1}",
                 "model.transformer={d_model: 32, n_heads: 2, n_layers: 2, patch: 1}"]
    monkeypatch.setattr(sys, "argv", ["train_thinker_real.py", str(fake), "--tag", "t", *overrides])
    mod.main()
    res = json.loads((tmp_path / "real_t.json").read_text())
    assert set(res["models"]) == {"hrm", "transformer"} and res["n_seeds"] == 2
    for m in res["models"].values():
        assert 0 <= m["test"]["teacher_agree"]["mean"] <= 1 and m["params"] > 0
        assert len(m["runs"]) == 2 and sum(sum(r.values()) for r in m["confusion"].values()) > 0
    assert "gate_pass" in res and (tmp_path / "real_t.md").exists()
    assert (tmp_path / "real_t_hrm_s0.jsonl").read_text().count("\n") == 2
