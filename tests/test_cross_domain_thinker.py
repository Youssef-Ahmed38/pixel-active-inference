"""The few-shot learning curve must continue one optimizer trajectory."""

import numpy as np

from pai.cognition.thinker_data import CAUSES, FEATURES, WINDOW, EvidenceSet
from scripts.cross_domain_thinker import _model, _train


def test_few_shot_training_keeps_optimizer_state_between_checkpoints():
    data = EvidenceSet(np.zeros((3, WINDOW, len(FEATURES)), np.float32),
                       np.array([0, 4, 5]), np.eye(len(CAUSES), dtype=np.float32)[[0, 4, 5]],
                       np.repeat("maze", 3), np.arange(3))
    model = _model(0)
    opt, rng = _train(model, data, np.arange(3), 1, seed=0)
    opt2, rng2 = _train(model, data, np.arange(3), 1, seed=0, optimizer=opt, rng=rng)
    assert opt2 is opt and rng2 is rng
    assert max(int(state["step"]) for state in opt.state.values()) == 2
