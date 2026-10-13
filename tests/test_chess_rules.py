import builtins

import numpy as np
import pytest

chess = pytest.importorskip("chess")
pytest.importorskip("sklearn")

from pai.chess.representation import BoardAction
from pai.chess.rule_evaluation import evaluate_rules
from pai.chess.rule_training import teach_attacks, teaching_positions, train_rules
from pai.chess.rules import ACTION_ARRAY, BETWEEN, LearnedChessRules, spatial_features
from pai.chess.teacher import ChessTeacher, LESSONS, observation
from pai.chess.state import GameState, LearnedChessStateModel
from pai.chess.state_training import game_state, train_state_model
from pai.chess.evaluation import LearnedChessEvaluator, LearnedPositionValue, value_features
from pai.chess.evaluation_training import train_evaluator
from pai.chess.search import LearnedChessSearch
from pai.cognition.contracts import SkillCommand


@pytest.fixture(scope="module")
def learned():
    positions = []
    for topic, fen in LESSONS.items():
        positions.extend(((chess.Board(fen), topic), (chess.Board(fen).mirror(), topic)))
    for fen in ("k7/8/8/4KPpr/8/8/8/8 w - g6 0 1",
                "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1"):
        positions.extend(((chess.Board(fen), "safety"), (chess.Board(fen).mirror(), "safety")))
    positions += teaching_positions(seed=72, games=2, plies=20, sparse=12, special=30, capture_lessons=True)
    return train_rules(positions, seed=72, corrections=12, progress=lambda *a, **kw: None)


def test_taught_attacks_generalize_to_cluttered_boards(learned):
    for board, _ in teaching_positions(seed=917, games=2, plies=20, sparse=8, special=8):
        obs = observation(board)
        pieces = np.asarray(obs.pieces)
        blocked = np.tensordot(BETWEEN, pieces != 0, axes=([2], [0])) > 0
        predicted = learned.attack_table[pieces[:, None], np.arange(64)[:, None], np.arange(64)[None, :], blocked.astype(int)]
        for source in range(64):
            assert set(np.flatnonzero(predicted[source])) == set(board.attacks(source))


@pytest.mark.parametrize("topic", ["castling", "en_passant", "promotion", "checkmate", "stalemate"])
def test_full_candidate_set_and_special_effects(learned, topic):
    board = chess.Board(LESSONS[topic])
    obs = observation(board)
    assert {a.uci() for a in learned.legal_actions(obs)} == {m.uci() for m in board.legal_moves}
    for move in board.legal_moves:
        after = board.copy()
        after.push(move)
        assert learned.predict_piece_board(obs, BoardAction.from_uci(move.uci())) == observation(after).pieces


def test_ep_discovered_check_and_castling_through_attack(learned):
    board = chess.Board("k7/8/8/4KPpr/8/8/8/8 w - g6 0 1")
    assert board.is_valid() and board.is_pseudo_legal(chess.Move.from_uci("f5g6"))
    assert not learned.classify(observation(board), [(37, 46, 0)])[0]
    board = chess.Board("r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1")
    assert "e1g1" not in {a.uci() for a in learned.legal_actions(observation(board))}


def test_json_reload_and_inference_without_rules_or_training_libraries(learned, tmp_path, monkeypatch):
    path = tmp_path / "learned.json"
    learned.save(path)
    obs = observation(chess.Board(LESSONS["en_passant"]))
    expected = learned.classify(obs)
    real_import = builtins.__import__
    def reject_oracle(name, *args, **kwargs):
        if name == "chess" or name.startswith(("chess.", "sklearn")) or name.endswith(("teacher", "rule_training")):
            raise AssertionError("inference attempted to access a teacher or trainer")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_oracle)
    restored = LearnedChessRules.load(path)
    assert np.array_equal(restored.classify(obs), expected)
    assert restored.choose(obs).uci() in {"e1d1", "e1f1", "e1d2", "e1e2", "e1f2", "e5e6", "e5d6"}


def test_evaluation_excludes_training_and_scores_all_actions(learned):
    positions = teaching_positions(seed=123, games=1, plies=8, sparse=2, special=4)
    report = evaluate_rules(learned, positions, progress=lambda *a, **kw: None)
    assert report["excluded_training_positions"] > 0
    assert report["position_overlap"] == 0
    assert sum(report["confusion"].values()) == report["positions"] * len(ACTION_ARRAY)
    assert report["actions_scored"] == report["positions"] * 20160


def test_independent_generator_validity_and_reproducibility():
    a = teaching_positions(seed=901, games=1, plies=8, sparse=5, special=12)
    b = teaching_positions(seed=901, games=1, plies=8, sparse=5, special=12)
    assert [(board.fen(), topic) for board, topic in a] == [(board.fen(), topic) for board, topic in b]
    assert all(board.is_valid() for board, _ in a)


def test_learned_command_executes_through_cognitive_adapter(learned):
    teacher = ChessTeacher(LESSONS["en_passant"])
    command = learned.command(teacher.observe())
    assert teacher.execute(command).executed
    assert teacher.record.commands[-1] == command
    assert learned.command(ChessTeacher(LESSONS["checkmate"]).observe()) is None


def test_rejected_move_predicts_unchanged_board(learned):
    obs = observation(chess.Board())
    illegal = BoardAction.from_uci("e2e5")
    assert not learned.classify(obs, [(illegal.source, illegal.destination, illegal.promotion)])[0]
    assert learned.predict_piece_board(obs, illegal) == obs.pieces


def test_initial_rank_pawn_capture_regression(learned):
    # Extra pieces and permissions differ from the isolated teaching examples.
    board = chess.Board("r3k2r/5p2/1b4R1/8/8/3r4/8/R3K2R b KQq - 0 1")
    action = BoardAction.from_uci("f7g6")
    assert board.is_legal(chess.Move.from_uci(action.uci()))
    assert learned.classify(observation(board), [(action.source, action.destination, 0)])[0]


def test_quiet_pawn_jump_cannot_capture_destination(learned):
    board = chess.Board("7R/k4pN1/3Kn2n/5N2/5NB1/4b3/3BB1R1/8 b - - 0 1")
    action = BoardAction.from_uci("f7f5")
    assert board.is_valid() and not board.is_legal(chess.Move.from_uci(action.uci()))
    assert not learned.classify(observation(board), [(action.source, action.destination, 0)])[0]


@pytest.fixture(scope="module")
def state_model(learned):
    positions = teaching_positions(seed=73, games=3, plies=30, sparse=30, special=160, capture_lessons=True)
    return train_state_model(learned, positions, seed=73)


@pytest.mark.parametrize("fen,moves", [
    (chess.STARTING_FEN, "e2e4 a7a6 e4e5 d7d5 e5d6"),
    (chess.STARTING_FEN, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"),
    (LESSONS["castling"], "e1g1 e8c8"),
    (LESSONS["castling"], "e1f1 e8f8 f1e1 f8e8"),
    (LESSONS["castling"], "a1a8"),
    (LESSONS["promotion"], "a7a8n"),
    (LESSONS["promotion"], "a7a8q"),
])
def test_full_state_sequence_and_history(state_model, fen, moves):
    board = chess.Board(fen)
    board.halfmove_clock = 137
    board.fullmove_number = 423
    predicted = game_state(board)
    for text in moves.split():
        action = BoardAction.from_uci(text)
        move = chess.Move.from_uci(text)
        assert board.is_legal(move)
        predicted = state_model.predict(predicted, action)
        board.push(move)
        assert predicted == game_state(board)
    assert GameState.from_belief(predicted.to_belief(step=19)) == predicted


def test_complete_state_rejected_actions_and_terminal(state_model):
    state = game_state(chess.Board())
    assert state_model.predict(state, BoardAction.from_uci("e2e5")) is state
    terminal = game_state(chess.Board(LESSONS["checkmate"]))
    assert state_model.successors(terminal) == ()
    imported = GameState(state.board, 73, 91)
    after = state_model.predict(imported, BoardAction.from_uci("g1f3"))
    assert after.history == (state.board, after.board)
    with pytest.raises(ValueError):
        GameState(state.board, -1)
    with pytest.raises(ValueError):
        GameState(after.board, history=(state.board,))


def test_full_state_checkpoint_inference_without_teacher(state_model, tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    state_model.save(path)
    before = game_state(chess.Board(LESSONS["en_passant"]))
    action = BoardAction.from_uci("e5d6")
    expected = state_model.predict(before, action)
    real_import = builtins.__import__
    def reject_oracle(name, *args, **kwargs):
        if name == "chess" or name.startswith(("chess.", "sklearn")) or name.endswith(("teacher", "rule_training", "state_training")):
            raise AssertionError("state inference attempted to access a teacher or trainer")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_oracle)
    restored = LearnedChessStateModel.load(path, state_model.rules)
    assert restored.predict(before, action) == expected


def test_teacher_exposes_full_state_through_shared_beliefs(state_model):
    teacher = ChessTeacher()
    before = GameState.from_belief(teacher.observe())
    action = BoardAction.from_uci("e2e4")
    predicted = state_model.predict(before, action)
    assert teacher.execute(action.command()).executed
    assert GameState.from_belief(teacher.observe()) == predicted
    assert not teacher.execute(BoardAction.from_uci("e7e4").command()).executed
    assert GameState.from_belief(teacher.observe()) == predicted


@pytest.fixture(scope="module")
def evaluator(state_model):
    return train_evaluator(state_model, seed=74, material_samples=800)


@pytest.mark.parametrize("topic", ["checkmate", "stalemate", "opening", "king", "castling"])
@pytest.mark.parametrize("clock", [0, 98, 99, 100, 149, 150, 170])
def test_learned_terminal_precedence_and_clock_claims(evaluator, topic, clock):
    board = chess.Board(LESSONS[topic])
    board.halfmove_clock = clock
    for claim in (False, True):
        answer = board.outcome(claim_draw=claim)
        status = evaluator.status(game_state(board), claim_draw=claim)
        assert status.termination == (None if answer is None else answer.termination.name)
        assert status.winner == (None if answer is None or answer.winner is None else int(answer.winner))


def test_learned_repetition_claim_before_move_and_automatic_draw(evaluator):
    board = chess.Board()
    sequence = "g1f3 g8f6 f3g1 f6g8".split()
    for i in range(17):
        state = game_state(board)
        for claim in (False, True):
            expected = board.outcome(claim_draw=claim)
            actual = evaluator.status(state, claim_draw=claim)
            assert actual.termination == (None if expected is None else expected.termination.name)
        if i < 16:
            board.push_uci(sequence[i % 4])


def test_repetition_identity_uses_only_legal_en_passant(evaluator):
    from dataclasses import replace
    for fen in (LESSONS["en_passant"], "k7/8/8/4KPpr/8/8/8/8 w - g6 0 1",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"):
        board = chess.Board(fen)
        obs = observation(board)
        same = evaluator.repetition_key(obs) == evaluator.repetition_key(replace(obs, ep_square=64))
        assert same == (not board.has_legal_en_passant())


@pytest.mark.parametrize("fen", [
    "8/8/2n5/8/8/6k1/5q2/7K w - - 0 1",
    "8/6k1/3n4/5K2/8/P7/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/2N5/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/2NN4/4K3 w - - 0 1",
])
def test_material_uses_both_colors_and_mixed_tokens(evaluator, fen):
    from pai.chess.evaluation import material_features
    board = chess.Board(fen)
    assert board.is_valid()
    x = material_features(observation(board))[None]
    assert bool(evaluator.trees["material"].predict(x)[0]) == board.is_insufficient_material()


def test_automatic_repetition_takes_precedence_over_claim(evaluator):
    board = chess.Board()
    board.halfmove_clock = 83
    for uci in "g1f3 g8f6 f3g1 f6g8".split() * 4:
        board.push_uci(uci)
    assert board.halfmove_clock == 99
    state = game_state(board)
    for claim in (False, True):
        answer = board.copy(stack=True).outcome(claim_draw=claim)
        assert answer.termination.name == "FIVEFOLD_REPETITION"
        assert evaluator.status(state, claim_draw=claim).termination == "FIVEFOLD_REPETITION"
    assert game_state(board) == state


def test_endings_checkpoint_and_value_inference_without_oracle(evaluator, tmp_path, monkeypatch):
    path = tmp_path / "endings.json"
    evaluator.save(path)
    state = game_state(chess.Board(LESSONS["checkmate"]))
    real_import = builtins.__import__
    def reject_oracle(name, *args, **kwargs):
        if name == "chess" or name.startswith(("chess.", "sklearn")) or name.endswith(("teacher", "rule_training", "state_training", "evaluation_training")):
            raise AssertionError("evaluation inference attempted to use an oracle")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_oracle)
    restored = LearnedChessEvaluator.load(path, evaluator.state_model)
    assert restored.status(state).termination == "CHECKMATE"
    value = LearnedPositionValue(np.zeros(391))
    path = tmp_path / "value.json"
    value.save(path)
    assert LearnedPositionValue.load(path).predict(state.board) == 0
    assert value.choose(evaluator.state_model, evaluator, state) is None


def test_value_features_color_symmetry_and_shape():
    board = chess.Board(LESSONS["promotion"])
    x = value_features(observation(board))
    assert x.shape == (391,)
    assert np.array_equal(x, -value_features(observation(board.mirror())))
    with pytest.raises(ValueError):
        LearnedPositionValue(np.zeros(7))


@pytest.mark.parametrize("mirror", [False, True])
def test_value_policy_prioritizes_immediate_win_through_cognitive_adapter(evaluator, mirror):
    board = chess.Board(LESSONS["mate_in_one"])
    if mirror:
        board = board.mirror()
    state = game_state(board)
    value = LearnedPositionValue(np.zeros(391))
    command = value.command(evaluator.state_model, evaluator, state.to_belief())
    move = chess.Move.from_uci(command.arguments[0])
    assert board.is_legal(move)
    board.push(move)
    assert board.is_checkmate()


def reference_minimax(board, value, depth, ply=0):
    result = board.copy(stack=True).outcome()
    if result:
        return 0.0 if result.winner is None else (100000 - ply) * (1 if result.winner == board.turn else -1)
    claim = board.copy(stack=True).can_claim_draw()
    if depth == 0:
        score = value.predict(observation(board)) * (1 if board.turn else -1)
        return max(0.0, score) if claim else score
    best = 0.0 if claim else -float("inf")
    for move in list(board.legal_moves):
        child = board.copy(stack=True)
        child.push(move)
        best = max(best, -reference_minimax(child, value, depth - 1, ply + 1))
    return best


@pytest.mark.parametrize("fen", [LESSONS["mate_in_one"], LESSONS["en_passant"],
                                "8/8/8/8/8/2k5/4r3/4K3 w - - 0 1"])
def test_search_matches_teacher_minimax_over_same_value(evaluator, fen):
    board = chess.Board(fen)
    value = LearnedPositionValue(np.random.default_rng(82).normal(size=391))
    expected = reference_minimax(board, value, 2)
    planner = LearnedChessSearch(evaluator.state_model, evaluator, value)
    state = game_state(board)
    result = planner.search(state, max_depth=2, max_nodes=2000)
    assert result.completed_depth == 2 and not result.budget_exhausted
    assert result.score_cp == pytest.approx(expected)
    assert result.action is not None and board.is_legal(chess.Move.from_uci(result.action.uci()))
    assert game_state(board) == state


def test_search_budget_keeps_last_complete_iteration(evaluator):
    value = LearnedPositionValue(np.zeros(391))
    planner = LearnedChessSearch(evaluator.state_model, evaluator, value)
    state = game_state(chess.Board())
    tiny = planner.search(state, max_depth=3, max_nodes=1)
    assert tiny.nodes == 1 and tiny.completed_depth == 0 and tiny.budget_exhausted
    assert tiny.score_cp is None and tiny.command() is not None
    shallow = planner.search(state, max_depth=1, max_nodes=100)
    interrupted = planner.search(state, max_depth=3, max_nodes=shallow.nodes + 1)
    assert interrupted.completed_depth == 1 and interrupted.budget_exhausted
    assert interrupted.action == shallow.action and interrupted.score_cp == shallow.score_cp
    timed = planner.search(state, time_limit=1e-9)
    assert timed.completed_depth == 0 and timed.budget_exhausted
    for kw in ({"max_nodes": 0}, {"max_depth": 0}, {"time_limit": 0}):
        with pytest.raises(ValueError):
            planner.search(state, **kw)


def test_search_claims_losing_position_with_intended_move(evaluator):
    # The halfmove clock permits a claim only if a qualifying move is intended.
    teacher = ChessTeacher("3q2k1/3r4/8/8/8/8/8/3Q2K1 w - - 99 1")
    value = LearnedPositionValue(np.zeros(391))
    planner = LearnedChessSearch(evaluator.state_model, evaluator, value)
    before = GameState.from_belief(teacher.observe())
    result = planner.search(before, max_depth=1, max_nodes=100)
    assert result.claim_draw and result.action is not None
    command = result.command()
    assert command.name == "claim_draw" and len(command.arguments) == 1
    assert teacher.execute(command).executed
    assert GameState.from_belief(teacher.observe()) == before
    assert teacher.game_outcome().termination.name == "FIFTY_MOVES"
    assert planner.command(teacher.observe()) is None
    assert not teacher.execute(BoardAction.from_uci("d1d7").command()).executed


@pytest.mark.parametrize("clock", [99, 100])
def test_draw_option_does_not_hide_winning_move(evaluator, clock):
    teacher = ChessTeacher(LESSONS["mate_in_one"].replace("0 1", f"{clock} 1"))
    planner = LearnedChessSearch(evaluator.state_model, evaluator, LearnedPositionValue(np.zeros(391)))
    result = planner.search(GameState.from_belief(teacher.observe()), max_depth=1, max_nodes=100)
    assert not result.claim_draw
    assert teacher.execute(result.command()).executed
    assert teacher.board.is_checkmate()


def test_teacher_rejects_bad_claims_and_freezes_automatic_draws():
    teacher = ChessTeacher()
    state = GameState.from_belief(teacher.observe())
    assert not teacher.execute(SkillCommand("claim_draw")).executed
    assert not teacher.execute(SkillCommand("claim_draw", arguments=("e2e5",))).executed
    assert GameState.from_belief(teacher.observe()) == state
    terminal = ChessTeacher(LESSONS["castling"].replace("0 1", "150 1"))
    assert terminal.game_outcome().termination.name == "SEVENTYFIVE_MOVES"
    assert not terminal.execute(BoardAction.from_uci("a1a2").command()).executed
    assert not terminal.demonstrate()


def test_search_claims_repetition_before_move(evaluator):
    teacher = ChessTeacher()
    for text in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1".split():
        assert teacher.execute(BoardAction.from_uci(text).command()).executed
    planner = LearnedChessSearch(evaluator.state_model, evaluator, LearnedPositionValue(np.zeros(391)))
    before = GameState.from_belief(teacher.observe())
    result = planner.search(before, max_depth=1, max_nodes=100)
    assert result.claim_draw and result.action.uci() == "f6g8"
    assert teacher.execute(result.command()).executed
    assert teacher.game_outcome().termination.name == "THREEFOLD_REPETITION"
    assert GameState.from_belief(teacher.observe()) == before


def test_search_runs_without_teacher_imports(evaluator, monkeypatch):
    state = game_state(chess.Board(LESSONS["mate_in_one"]))
    planner = LearnedChessSearch(evaluator.state_model, evaluator, LearnedPositionValue(np.zeros(391)))
    real_import = builtins.__import__
    def reject_oracle(name, *args, **kwargs):
        if name == "chess" or name.startswith(("chess.", "sklearn")) or name.endswith(("teacher", "rule_training", "state_training", "evaluation_training")):
            raise AssertionError("search inference attempted to use an oracle")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_oracle)
    result = planner.search(state, max_depth=2, max_nodes=100)
    assert result.action is not None and result.score_cp > 90000
