"""scripts/collect_evidence.py: episode plan, shard merge, and a tiny real run (1 process) that
resumes, merges and loads with the thinker's loader."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pai.causes.inference import CAUSES, CHANNELS
from pai.thinker.real import RealConfig, RealEvidence, load_evidence

ROOT = Path(__file__).resolve().parent.parent
WORLD_MODEL = ROOT / "runs" / "slice_world_model" / "world_model.pt"

spec = importlib.util.spec_from_file_location("collect_evidence", ROOT / "scripts" / "collect_evidence.py")
ce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ce)


def test_plan_is_interleaved_and_chunks_cover_every_episode_once():
    plan = ce.episode_plan(["none", "push"], 3)
    assert [g for g, _ in plan] == list(range(6)) and [c for _, c in plan[:2]] == ["none", "push"]
    chunks = [ce.chunks_for_rank(103, 10, r, 2) for r in range(2)]
    assert sorted(chunks[0] + chunks[1]) == list(range(11)) and not set(chunks[0]) & set(chunks[1])


def _fake_episode(g: int, label: str, rng) -> dict:
    t = int(rng.integers(5, 12))
    return {"z": rng.standard_normal((t, len(CHANNELS))), "holding": rng.random(t) > 0.5,
            "phase": rng.integers(-1, 4, t), "label": label, "teacher": np.full(len(CAUSES), 1 / len(CAUSES)),
            "onset": -1, "episode_id": g, "seed": 10000 + g, "success": True, "inferred": "none"}


def test_merge_orders_by_global_id_and_renumbers(tmp_path):
    rng = np.random.default_rng(0)
    eps = [_fake_episode(g, CAUSES[g % 3], rng) for g in range(6)]
    ce.save_atomic(ce.shard_path(tmp_path, 1), ce.pack(eps[3:]))  # written out of order
    ce.save_atomic(ce.shard_path(tmp_path, 0), ce.pack(eps[:3]))
    d = load_evidence(ce.merge(tmp_path, tmp_path / "m_evidence.npz"))
    assert d["episode_id"].tolist() == list(range(6)) and set(d["episode"].tolist()) == set(range(6))
    assert (np.diff(d["episode"]) >= 0).all() and len(d["z"]) == sum(len(e["z"]) for e in eps)
    assert np.allclose(d["z"][d["episode"] == 4], eps[4]["z"])
    ce.save_atomic(ce.shard_path(tmp_path, 2), ce.pack(eps[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        ce.merge(tmp_path)


@pytest.mark.skipif(not WORLD_MODEL.exists(), reason="needs runs/slice_world_model/world_model.pt")
def test_tiny_real_run_resumes_merges_and_loads(tmp_path):
    out = tmp_path / "ev"
    cmd = [sys.executable, str(ROOT / "scripts" / "collect_evidence.py"), "--config", "configs/tabletop.yaml",
           "--conditions", "none,push", "--episodes", "1", "--calibration-episodes", "1", "--chunk", "1",
           "--out", str(out), "slice.max_plan_steps=40", "slice.samples=64", f"slice.world_model={WORLD_MODEL}"]
    first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert first.returncode == 0, first.stderr[-2000:]
    assert "ETA" in first.stdout and sorted(p.name for p in out.glob("shard_*.npz")) == ["shard_00000.npz",
                                                                                         "shard_00001.npz"]
    again = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)  # resume: nothing to run
    assert again.returncode == 0 and "0 episodes to run" in again.stdout, again.stderr[-2000:]
    merge = subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_evidence.py"), "--merge", str(out),
                            "--tag", "tiny"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert merge.returncode == 0, merge.stderr[-2000:]
    d = load_evidence(out / "tiny_evidence.npz")
    assert d["labels"].tolist() == ["none", "push"] and d["seed"].tolist() == [10000, 10001]
    assert tuple(d["causes"]) == CAUSES and tuple(d["channels"]) == CHANNELS and d["z"].dtype == np.float32
    data = RealEvidence(out / "tiny_evidence.npz", RealConfig(val_frac=0.0, test_frac=0.0))
    assert data.n_episodes == 2 and data.feats.shape[-1] == len(CHANNELS) + 1 + 10
