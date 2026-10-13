import builtins
import numpy as np
import pytest
import torch

from pai.chess.representation import BoardObservation
from pai.chess.evaluation import LearnedPositionValue
from pai.chess.hrm_value import HRMPositionValue, board_tensors, mirror_observation
from pai.thinker.hrm import HRM


def sample():
    pieces = [0] * 64
    pieces[4], pieces[60], pieces[12], pieces[51] = 6, 12, 1, 7
    return BoardObservation(tuple(pieces), 1, (1, 0, 0, 1), 20)


def model():
    config = {"in_dim": 16, "ctx_dim": 5, "max_len": 64, "heads": {"value": 1},
              "d_model": 16, "n_heads": 4, "h_layers": 1, "l_layers": 1,
              "h_cycles": 1, "l_steps": 1, "max_segments": 2}
    torch.manual_seed(1)
    baseline = LearnedPositionValue(np.arange(391) / 10, {"training_positions": []})
    return HRMPositionValue(HRM(**config), baseline, config)


def test_public_encoding_and_mirror_involution():
    observation = sample()
    assert mirror_observation(mirror_observation(observation)) == observation
    x, mask, ctx = board_tensors([observation])
    assert x.shape == (1, 64, 16) and ctx.shape == (1, 5)
    assert x[0, 20, 13] == 1 and x[0, :, :13].sum() == 64
    assert mask.sum() == 64
    with pytest.raises(ValueError):
        model().predict_many([observation], segments=0)


def test_batch_predictions_symmetry_and_checkpoint(tmp_path):
    value = model()
    observation = sample()
    predictions = value.predict_many([observation, mirror_observation(observation)])
    assert predictions[0] == pytest.approx(-predictions[1], abs=1e-3)
    assert value.predict(observation) == pytest.approx(predictions[0], abs=1e-3)
    path = tmp_path / "value.pt"
    value.save(path)
    restored = HRMPositionValue.load(path)
    assert restored.predict(observation) == pytest.approx(value.predict(observation), abs=1e-4)


def test_zero_residual_preserves_baseline_and_teacher_free_inference(monkeypatch):
    value = model()
    torch.nn.init.zeros_(value.model.readout.heads["value"].weight)
    torch.nn.init.zeros_(value.model.readout.heads["value"].bias)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "chess" or name.startswith("chess.") or name == "pai.chess.teacher":
            raise AssertionError("teacher imported during HRM inference")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert value.predict(sample()) == pytest.approx(value.baseline.predict(sample()), abs=1e-4)
