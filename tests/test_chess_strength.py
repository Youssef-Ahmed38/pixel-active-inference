"""Batched learned-rule features, distilled value symmetry and ladder statistics."""
import random
from pathlib import Path

import chess
import numpy as np
import pytest

from pai.chess.distill import DistilledValue, FEATURES, canonical, features, prob_to_cp, cp_to_prob
from pai.chess.ladder import anchored_elo, candidate_score, elo_from_score, pair_statistics, play_game
from pai.chess.rules import LearnedChessRules, safety_features, safety_features_batch
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
