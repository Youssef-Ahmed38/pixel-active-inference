import pytest
import numpy as np

chess = pytest.importorskip("chess")
pytest.importorskip("sklearn")

from pai.chess import LearnedPositionValue
from pai.chess.evaluation import value_features
from pai.chess.mcts import LearnedChessMCTS
from pai.chess.teacher import ChessTeacher, LESSONS
from pai.chess.state import GameState
from test_chess_rules import learned, state_model, evaluator


@pytest.fixture(scope="module")
def planner(evaluator):
    board = GameState.from_belief(ChessTeacher().observe()).board
    value = LearnedPositionValue(np.zeros(len(value_features(board))), {})
    return LearnedChessMCTS(evaluator.state_model, evaluator, value)


@pytest.mark.parametrize("budget", [1, 2, 32, 128])
def test_budget_legality_and_no_mutation(planner, budget):
    teacher = ChessTeacher()
    state = GameState.from_belief(teacher.observe())
    result = planner.search(state, max_nodes=budget)
    assert result.nodes <= budget and result.completed_depth == 0
    assert result.deepest_ply <= 3
    assert result.command() is not None
    assert teacher.board.is_legal(chess.Move.from_uci(result.action.uci()))
    assert GameState.from_belief(teacher.observe()) == state


def test_terminal_and_external_claim_abstain(planner):
    teacher = ChessTeacher(LESSONS["checkmate"])
    assert planner.command(teacher.observe()) is None
    result = planner.search(GameState.from_belief(teacher.observe()))
    assert result.action is None and not result.claim_draw


def test_optional_draw_with_one_node_has_executable_witness(planner):
    teacher = ChessTeacher("3q2k1/3r4/8/8/8/8/8/3Q2K1 w - - 99 1")
    state = GameState.from_belief(teacher.observe())
    result = planner.search(state, max_nodes=1)
    assert result.claim_draw
    assert teacher.execute(result.command()).executed
    assert teacher.game_outcome().winner is None
    assert GameState.from_belief(teacher.observe()) == state


def test_winning_move_remains_available_with_draw_claim(planner):
    teacher = ChessTeacher(LESSONS["mate_in_one"].replace("0 1", "100 1"))
    result = planner.search(GameState.from_belief(teacher.observe()), max_nodes=256)
    assert not result.claim_draw
    assert teacher.execute(result.command()).executed
    assert teacher.board.is_checkmate()


def test_invalid_budget(planner):
    state = GameState.from_belief(ChessTeacher().observe())
    with pytest.raises(ValueError):
        planner.search(state, max_nodes=0)


def test_mcts_runtime_does_not_import_teacher(planner, monkeypatch):
    import builtins
    state = GameState.from_belief(ChessTeacher().observe())
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "chess" or name.startswith("chess.") or name == "pai.chess.teacher" or name.endswith("_training"):
            raise AssertionError(f"oracle import during inference: {name}")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert planner.search(state, max_nodes=32).action is not None
