import pytest

chess = pytest.importorskip("chess")
pytest.importorskip("sklearn")
from test_chess_rules import learned, state_model, evaluator
from test_chess_quiescence import value
from pai.chess.search import LearnedChessSearch
from pai.chess.state_training import game_state
from pai.chess.teacher import observation


class ReverseOrdering:
    def __init__(self, leaf):
        self.leaf = leaf
        self.calls = 0

    def predict_many(self, observations):
        self.calls += 1
        return [-self.leaf.predict(obs) for obs in observations]


def minimax(board, leaf, depth, ply=0):
    outcome = board.outcome()
    if outcome is not None:
        return 0 if outcome.winner is None else (100000 - ply) * (1 if outcome.winner == board.turn else -1)
    if depth == 0:
        return leaf.predict(observation(board)) * (1 if board.turn else -1)
    scores = []
    for move in board.legal_moves:
        child = board.copy(stack=True)
        child.push(move)
        scores.append(-minimax(child, leaf, depth - 1, ply + 1))
    return max(scores)


@pytest.mark.parametrize("mirror", [False, True])
def test_ordering_and_pv_reuse_preserve_full_depth_minimax(evaluator, mirror):
    board = chess.Board("3q2k1/3r4/8/8/8/8/8/3Q2K1 w - - 0 1")
    if mirror:
        board = board.mirror()
    leaf = value()
    ordering = ReverseOrdering(leaf)
    planner = LearnedChessSearch(evaluator.state_model, evaluator, leaf, ordering)
    expected = minimax(board, leaf, 2)
    for reuse in (False, True):
        result = planner.search(game_state(board), max_depth=2, max_nodes=4000, reuse_pv=reuse)
        assert result.completed_depth == 2
        assert result.score_cp == pytest.approx(expected)
        replay = board.copy(stack=True)
        for action in result.principal_variation:
            move = chess.Move.from_uci(action.uci())
            assert replay.is_legal(move)
            replay.push(move)
    assert ordering.calls > 0
    result = planner.search(game_state(board), max_nodes=1, reuse_pv=True)
    assert result.nodes == 1 and result.completed_depth == 0 and result.budget_exhausted
    assert board.is_legal(chess.Move.from_uci(result.action.uci()))


@pytest.mark.parametrize("kind", ["alpha-beta", "mcts"])
def test_short_deadlines_work_when_monotonic_has_coarse_ticks(evaluator, monkeypatch, kind):
    import itertools
    import time
    from pai.chess.mcts import LearnedChessMCTS
    ticks = itertools.count()
    monkeypatch.setattr(time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks) * .001)
    cls = LearnedChessSearch if kind == "alpha-beta" else LearnedChessMCTS
    board = chess.Board()
    result = cls(evaluator.state_model, evaluator, value()).search(game_state(board), time_limit=1e-9)
    assert result.nodes == 0 and result.completed_depth == 0 and result.budget_exhausted
    assert board.is_legal(chess.Move.from_uci(result.action.uci()))
