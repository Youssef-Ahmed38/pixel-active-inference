"""Batched learned-rule features, distilled value symmetry and ladder statistics."""
import random
from pathlib import Path

import chess
import numpy as np
import pytest

from pai.chess.distill import DistilledValue, FEATURES, canonical, features, prob_to_cp, cp_to_prob
from pai.chess.ladder import anchored_elo, candidate_score, elo_from_score, pair_statistics, play_game
from pai.chess.rules import LearnedChessRules, safety_features, safety_features_batch
from pai.chess.rules import ACTION_ARRAY, spatial_features, spatial_features_reference
from pai.chess.teacher import observation

RULES = Path("runs/chess_rules/rules.json")
needs_rules = pytest.mark.skipif(not RULES.exists(), reason="learned rule checkpoint not present")


def random_board(rng, plies):
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


@needs_rules
def test_batched_safety_features_equal_per_board_features():
    rules = LearnedChessRules.load(RULES)
    rng = random.Random(3)
    for _ in range(25):
        obs = observation(random_board(rng, rng.randrange(0, 90)))
        boards = np.array([obs.pieces] * 4)
        for row in boards[1:]:
            row[rng.randrange(64)] = 0
        expected = np.array([safety_features(b, obs.turn, rules.attack_table) for b in boards])
        assert (safety_features_batch(boards, obs.turn, rules.attack_table) == expected).all()


@needs_rules
def test_batched_rules_still_match_teacher_on_random_positions():
    rules = LearnedChessRules.load(RULES)
    rng = random.Random(11)
    for _ in range(12):
        board = random_board(rng, rng.randrange(0, 70))
        learned = {a.uci() for a in rules.legal_actions(observation(board))}
        assert learned == {m.uci() for m in board.legal_moves}


def test_canonical_view_swaps_colour_and_ranks():
    board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1")
    obs = observation(board)
    out, rights = canonical([obs.pieces], [obs.turn], [obs.castling])
    flipped = observation(board.mirror())
    assert tuple(out[0]) == flipped.pieces


@needs_rules
def test_distilled_value_is_colour_symmetric():
    rules = LearnedChessRules.load(RULES)
    rng = np.random.default_rng(0)
    layers = [(rng.normal(0, .05, (FEATURES, 16)), rng.normal(0, .05, 16)), (rng.normal(0, .3, (16, 1)), np.zeros(1))]
    value = DistilledValue(layers, rules.attack_table)
    r = random.Random(5)
    for _ in range(10):
        board = random_board(r, r.randrange(1, 40))
        a, b = value.predict(observation(board)), value.predict(observation(board.mirror()))
        assert a == pytest.approx(-b, abs=1e-3)


def test_probability_cp_round_trip():
    cps = np.array([-900., -100., 0., 250., 1200.])
    assert np.allclose(prob_to_cp(cp_to_prob(cps)), cps, atol=1e-2)


def test_distilled_value_save_load(tmp_path):
    table = np.zeros((13, 64, 64, 2), dtype=np.int8)
    layers = [(np.ones((FEATURES, 4)) * .01, np.zeros(4)), (np.ones((4, 1)), np.zeros(1))]
    value = DistilledValue(layers, table, {"a": 1})
    value.save(tmp_path / "v.npz")
    loaded = DistilledValue.load(tmp_path / "v.npz")
    obs = observation(chess.Board())
    assert loaded.summary == {"a": 1}
    assert loaded.predict(obs) == pytest.approx(value.predict(obs))


def game(pair, colour, result, status="complete"):
    return {"pair": pair, "candidate_color": colour, "result": result, "status": status}


def test_unresolved_games_are_not_scored_as_draws():
    assert candidate_score(game(0, "white", "*", "truncated")) is None
    stats = pair_statistics([game(0, "white", "1-0"), game(0, "black", "*", "truncated")])
    assert stats["games"] == 1 and stats["unresolved_games"] == 1


def test_pair_statistics_and_elo_mapping():
    games = [game(i, c, r) for i in range(10) for c, r in (("white", "1-0"), ("black", "1/2-1/2"))]
    stats = pair_statistics(games)
    assert stats["score"] == pytest.approx(0.75)
    assert stats["elo_difference"] == pytest.approx(elo_from_score(.75))
    assert stats["score_ci95_pair_bootstrap"][0] == pytest.approx(.75)  # every pair identical
    assert stats["lower_bound_above_half"]
    assert elo_from_score(0) is None and elo_from_score(1) is None


def test_zero_score_has_no_finite_anchored_estimate():
    losses = [game(i, c, "0-1" if c == "white" else "1-0") for i in range(4) for c in ("white", "black")]
    assert anchored_elo([(1320, losses)])["estimate"] is None


def test_anchored_estimate_recovers_even_score():
    games = [game(i, c, "1-0") for i in range(5) for c in ("white",)] + \
            [game(i, c, "1-0") for i in range(5) for c in ("black",)]
    # Candidate wins as White and loses as Black: 50% against a 1500 anchor.
    estimate = anchored_elo([(1500, games)], bootstrap=200)
    assert estimate["estimate"] == pytest.approx(1500, abs=1)


class Scripted:
    def __init__(self, moves):
        self.moves = list(moves)

    def command(self, teacher, rng):
        from pai.chess.representation import BoardAction
        return BoardAction.from_uci(self.moves.pop(0)).command(), None


def test_teacher_declares_result_of_scripted_game():
    white, black = Scripted(["f2f3", "g2g4"]), Scripted(["e7e5", "d8h4"])
    result = play_game(white, black, chess.STARTING_FEN, max_plies=10)
    assert result["status"] == "complete" and result["result"] == "0-1"
    assert result["termination"] == "CHECKMATE"


def test_illegal_command_is_an_execution_failure_not_a_result():
    white, black = Scripted(["e2e5"]), Scripted([])
    result = play_game(white, black, chess.STARTING_FEN, max_plies=4)
    assert result["status"] == "execution_failure" and result["result"] == "*"

@needs_rules
def test_compiled_features_equal_frozen_reference_for_all_actions():
    rules = LearnedChessRules.load(RULES)
    for fen in (chess.STARTING_FEN, "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1"):
        obs = observation(chess.Board(fen))
        np.testing.assert_array_equal(spatial_features(obs, ACTION_ARRAY, rules.attack_table),
                                      spatial_features_reference(obs, ACTION_ARRAY, rules.attack_table))

@needs_rules
def test_agent_decision_path_cannot_access_chess_library_or_engine(monkeypatch):
    import builtins
    from pai.chess.ladder import AgentPlayer
    from pai.chess.teacher import ChessTeacher
    teacher = ChessTeacher()
    belief = teacher.observe()
    player = AgentPlayer({"value": "runs/chess_rules/value.json", "time": .1})
    player.prepare(belief)
    original = builtins.__import__
    def reject_import(name, *args, **kwargs):
        if name == "chess" or name.startswith("chess."):
            raise AssertionError("agent consulted chess oracle")
        return original(name, *args, **kwargs)
    def reject(*args, **kwargs):
        raise AssertionError("agent consulted chess oracle")
    monkeypatch.setattr(builtins, "__import__", reject_import)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", reject)
    monkeypatch.setattr(chess.Board, "legal_moves", property(reject))
    command, stats = player.command_from_belief(belief)
    assert command.name == "move_piece" and stats["nodes"] > 0

def test_incomplete_pairs_and_duplicates_cannot_promote():
    partial = [game(i, "white", "1-0") for i in range(10)]
    assert not pair_statistics(partial)["promotion_gate_passed"]
    with pytest.raises(ValueError): pair_statistics(partial + [partial[0]])
    all_wins = [game(i, c, "1-0" if c == "white" else "0-1") for i in range(10) for c in ("white", "black")]
    assert pair_statistics(all_wins)["paired_sign_test_p_one_sided"] == pytest.approx(1 / 1024)
    assert pair_statistics(all_wins)["promotion_gate_passed"]

def test_invalid_completed_results_and_layers_rejected():
    with pytest.raises(ValueError): candidate_score(game(0, "white", "*"))
    with pytest.raises(ValueError): DistilledValue([], np.zeros((13, 64, 64, 2)))

def test_extreme_bootstrap_rating_endpoints_are_json_safe():
    import json
    rows = [game(0, "white", "1-0"), game(0, "black", "0-1"),
            game(1, "white", "0-1"), game(1, "black", "1-0")]
    estimate = anchored_elo([(1500, rows)], bootstrap=200)
    assert estimate["estimate"] == pytest.approx(1500)
    assert estimate["ci95"] == [None, None]
    json.dumps(estimate, allow_nan=False)

def test_scalar_and_batched_linear_values_are_identical():
    from pai.chess.evaluation import LearnedPositionValue
    rng = np.random.default_rng(23)
    value = LearnedPositionValue(rng.normal(size=391))
    boards = [observation(random_board(random.Random(i), 30)) for i in range(12)]
    np.testing.assert_allclose(value.predict_many(boards), [value.predict(o) for o in boards], atol=2e-5)
    assert value.predict_many([]).shape == (0,)

def test_policy_roundtrip_and_colour_symmetry(tmp_path):
    from pai.chess.policy import DIM,LearnedPolicyPrior
    from pai.chess.representation import BoardAction
    board = chess.Board()
    actions = [BoardAction.from_uci(m.uci()) for m in list(board.legal_moves)]
    rng = np.random.default_rng(7)
    prior = LearnedPolicyPrior([(rng.normal(size=(DIM,8)),np.zeros(8)),(rng.normal(size=(8,1)),np.zeros(1))],
                               np.zeros(DIM),np.ones(DIM),np.zeros((13,64,64,2),dtype=np.int8))
    mirrored = [BoardAction(a.source ^ 56,a.destination ^ 56,a.promotion) for a in actions]
    np.testing.assert_allclose(prior.predict_actions(observation(board),actions),
                               prior.predict_actions(observation(board.mirror()),mirrored),atol=1e-5)
    prior.save(tmp_path / "policy.npz")
    loaded = LearnedPolicyPrior.load(tmp_path / "policy.npz")
    np.testing.assert_array_equal(loaded.predict_actions(observation(board),actions),prior.predict_actions(observation(board),actions))

def test_degenerate_bootstrap_does_not_imply_perfect_certainty():
    rows = [game(i,c,"1/2-1/2") for i in range(10) for c in ("white","black")]
    stats = pair_statistics(rows)
    assert stats["bootstrap_degenerate"]
    assert stats["score_ci95_reported"][0] < .5 < stats["score_ci95_reported"][1]

def test_promoted_manifest_rejects_unapproved_or_changed_artifacts(tmp_path):
    import json,hashlib
    from pai.chess.agent import AgentPlayer
    artifact = tmp_path / "checkpoint"
    artifact.write_text("initial")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"approved":False}))
    with pytest.raises(ValueError,match="promotion"): AgentPlayer.from_manifest(manifest)
    manifest.write_text(json.dumps({"approved":True,"checkpoint_hashes":{str(artifact):hashlib.sha256(b"initial").hexdigest()}}))
    artifact.write_text("changed")
    with pytest.raises(ValueError,match="identity"): AgentPlayer.from_manifest(manifest)

def test_distillation_keeps_colour_orbits_and_trajectory_identity():
    from pai.chess.distill import position_hash,rows_to_arrays
    board = chess.Board()
    mirrored = board.mirror()
    assert position_hash(board.fen()) == position_hash(mirrored.fen())
    later_clock = board.copy()
    later_clock.halfmove_clock = 50
    later_clock.fullmove_number = 20
    assert position_hash(board.fen()) == position_hash(later_clock.fen())
    rows = [(board.fen(),200.,"e2e4","seed:game"),(mirrored.fen(),200.,"e7e5","other:game")]
    x,y = rows_to_arrays(rows,np.zeros((13,64,64,2),dtype=np.int8))
    np.testing.assert_array_equal(x[0],x[1])
    np.testing.assert_array_equal(y,[200.,200.])
