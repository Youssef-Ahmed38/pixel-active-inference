import json

import numpy as np
import torch

from pai.config import load_config, REPO_ROOT
from pai.data import JointImageDataset
from pai.data.collect import collect
from pai.models import JointDecoder, load_decoder
from pai.train.decoder import train


def test_decoder_shapes_and_range():
    for size in (16, 64, 256):
        dec = JointDecoder([-1] * 4, [1] * 4, size, base_channels=8, max_channels=32)
        out = dec(torch.zeros(2, 4))
        assert out.shape == (2, 3, size, size)
        assert out.min() >= 0 and out.max() <= 1


def test_collect_train_resume_roundtrip(tmp_path):
    overrides = [
        "env.image_size=64", "decoder.image_size=32", "decoder.base_channels=8", "decoder.max_channels=32",
        f"data.dir={tmp_path / 'data'}", "data.n_samples=64", "data.shard_size=32", "data.workers=1",
        f"train.out_dir={tmp_path / 'run'}", "train.batch_size=8", "train.max_steps=6", "train.warmup_steps=2",
        "train.log_every=2", "train.eval_every=4", "train.ckpt_every=3", "train.num_workers=0",
    ]
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml", overrides)
    collect(cfg)
    ds = JointImageDataset(cfg.data.dir)
    assert len(ds) == 64
    q, img = ds[5]
    assert img.shape == (3, 32, 32) and img.dtype == torch.uint8 and q.shape == (4,)

    path = train(cfg)
    dec = load_decoder(path)
    assert dec(torch.zeros(1, 4)).shape == (1, 3, 32, 32)

    # Extending max_steps resumes from the saved step instead of starting over.
    cfg.train.max_steps = 9
    train(cfg)
    rows = [json.loads(l) for l in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    starts = [r["step"] for r in rows if r.get("event") == "start"]
    assert starts == [0, 6]
    assert np.isfinite([r["val_mse"] for r in rows if "val_mse" in r]).all()
