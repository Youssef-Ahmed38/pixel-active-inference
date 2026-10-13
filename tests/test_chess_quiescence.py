import numpy as np
import pytest

chess = pytest.importorskip("chess")
pytest.importorskip("sklearn")
from test_chess_rules import learned, state_model, evaluator
from pai.chess.evaluation import LearnedPositionValue
from pai.chess.search import LearnedChessSearch
from pai.chess.state_training import game_state
from pai.chess.teacher import observation


def value():
    weights = np.zeros(391)
    weights[:6] = [100, 320, 330, 500, 900, 0]
    return LearnedPositionValue(weights, {})


def teacher_quiet(board, learned_value, remaining, ply):
    outcome = board.outcome()
    if outcome is not None:
        return 0 if outcome.winner is None else (100000 - ply) * (1 if outcome.winner == board.turn else -1)
    score = learned_value.predict(observation(board)) * (1 if board.turn else -1)
    if remaining == 0:
        return score
    checked = board.is_check()
    best = -float("inf") if checked else score
    for move in list(board.legal_moves):
        if not checked and not board.is_capture(move):
            continue
        child = board.copy(stack=True)
        child.push(move)
        best = max(best, -teacher_quiet(child, learned_value, remaining - 1, ply + 1))
    return best


@pytest.mark.parametrize("mirror", [False, True])
def test_quiescence_matches_independent_teacher_and_avoids_recapture(evaluator, mirror):
    board = chess.Board("3q2k1/3r4/8/8/8/8/8/3Q2K1 w - - 0 1")
    if mirror:
        board = board.mirror()
    learned_value = value()
    planner = LearnedChessSearch(evaluator.state_model, evaluator, learned_value)
    expected = -float("inf")
    for move in board.legal_moves:
        child = board.copy(stack=True)
        child.push(move)
        expected = max(expected, -teacher_quiet(child, learned_value, 2, 1))
    result = planner.search(game_state(board), max_depth=1, max_nodes=4000, max_quiescence=2)
    assert result.completed_depth == 1
    assert result.score_cp == pytest.approx(expected)
    assert board.piece_at(chess.Move.from_uci(result.action.uci()).to_square) is None
    replay = board.copy(stack=True)
    for action in result.principal_variation:
        move = chess.Move.from_uci(action.uci())
        assert replay.is_legal(move)
        replay.push(move)


def test_quiescence_respects_budget_and_rejects_negative_depth(evaluator):
    board = chess.Board()
    planner = LearnedChessSearch(evaluator.state_model, evaluator, value())
    result = planner.search(game_state(board), max_nodes=1, max_quiescence=3)
    assert result.nodes == 1 and result.budget_exhausted
    assert board.is_legal(chess.Move.from_uci(result.action.uci()))
    with pytest.raises(ValueError):
        planner.search(game_state(board), max_quiescence=-1)
