import json

import numpy as np

from pai.config import Config
from pai.perception.train_slots import train_slots
from pai.world.model import load_world_model
from pai.world.train import train_world_model


def _fake_episodes(d, n=4, T=12, E=7):
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        tok = rng.normal(size=(T + 1, E, 19)).astype(np.float32)
        np.savez_compressed(d / f"ep_{i:05d}.npz", tokens=tok, actions=rng.normal(size=(T, 4)).astype(np.float32))


def test_world_model_training_resumes(tmp_path):
    _fake_episodes(tmp_path / "data")
    wm = {"data_dir": str(tmp_path / "data"), "out_dir": str(tmp_path / "run"), "n_members": 2, "d_model": 16,
          "n_layers": 1, "n_heads": 2, "lr": 1e-3, "batch_size": 8, "steps": 4, "log_every": 2}
    cfg = Config({"seed": 0, "world_model": wm})
    train_world_model(cfg, device="cpu")
    cfg.world_model.steps = 6
    path = train_world_model(cfg, device="cpu")
    rows = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if r.get("event") == "start"] == [0, 4]
    assert load_world_model(path).predict is not None


def test_slot_training_runs_on_tiny_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    N, L = 40, 8
    np.save(cache / "feats.npy", rng.normal(size=(N, 256, 384)).astype(np.float16))
    m = rng.random((N, 256, L)).astype(np.float32)
    np.save(cache / "masks.npy", (m / m.sum(-1, keepdims=True)).astype(np.float16))
    np.save(cache / "tokens.npy", rng.normal(size=(N, 7, 19)).astype(np.float32))
    np.save(cache / "episode.npy", np.repeat(np.arange(20), 2).astype(np.int32))
    sc = {"cache_dir": str(cache), "out_dir": str(tmp_path / "slots"), "n_slots": 8, "dim": 32, "iters": 2,
          "lr": 1e-3, "batch_size": 4, "steps": 3, "mask_weight": 1.0, "token_weight": 1.0, "log_every": 3}
    path = train_slots(Config({"seed": 0, "slots": sc}), device="cpu")
    last = json.loads((tmp_path / "slots" / "log.jsonl").read_text().splitlines()[-1])
    assert path.exists() and 0.0 <= last["val_object_miou"] <= 1.0 and np.isfinite(last["total"])
