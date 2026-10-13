import numpy as np
import pytest

from pai.chess.evaluation import LearnedPositionValue
from pai.chess.hrm_value import mirror_observation, load_value
from pai.chess.threat_value import ContextPositionValue, context_features
from pai.chess.representation import BoardObservation


def observation():
    pieces = [0] * 64
    pieces[4], pieces[60], pieces[12], pieces[51] = 6, 12, 1, 7
    return BoardObservation(tuple(pieces), 1, (1, 0, 0, 1), 64)


def test_context_reads_learned_attack_field_and_changes_sign():
    board = observation()
    table = np.ones((13, 64, 64, 2), dtype=np.int8)
    # Empty token categories have no field in this synthetic checkpoint.
    table[0] = 0
    x = context_features(board, table)
    assert x.shape == (24,)
    np.testing.assert_array_equal(context_features(mirror_observation(board), table), -x)
    changed = table.copy()
    changed[1] = 0
    assert not np.array_equal(context_features(board, changed), x)


def test_context_checkpoint_is_self_contained_and_baseline_persists(tmp_path):
    from types import SimpleNamespace
    baseline = LearnedPositionValue(np.ones(391), {"training_positions": []})
    value = ContextPositionValue(baseline, SimpleNamespace(attack_table=np.zeros((13, 64, 64, 2), dtype=np.int8)), np.ones(24))
    path = tmp_path / "value.json"
    value.save(path)
    restored = load_value(path)
    assert restored.predict(observation()) == baseline.predict(observation())
    assert restored.summary == value.summary


def test_invalid_context_weights_rejected():
    with pytest.raises(ValueError):
        ContextPositionValue(None, None, np.zeros(3))


def test_search_evaluation_excludes_ordering_model_teaching(monkeypatch):
    import chess
    from types import SimpleNamespace
    import pai.chess.search_evaluation as module
    from pai.chess.teacher import observation as observe, position_key
    board = chess.Board()
    key = position_key(observe(board))
    monkeypatch.setattr(module, "tactical_positions", lambda *args: [(board, "poisoned_rook")])
    planner = SimpleNamespace(state_model=SimpleNamespace(training_positions=[],
        rules=SimpleNamespace(training_positions=[])), value=SimpleNamespace(summary={}),
        ordering_value=SimpleNamespace(summary={"training_positions": [key]}))
    report = module.evaluate_search(planner, budgets=[32])
    assert report["excluded_teaching_boards"] == 1
    assert report["budgets"][0]["positions"] == 0
    assert not report["budgets"][0]["correctness_gate_passed"]
