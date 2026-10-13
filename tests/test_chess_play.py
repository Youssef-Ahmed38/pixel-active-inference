import math

import pytest

from pai.chess.play import performance, play_game


def test_truncations_do_not_count_as_draws():
    assert performance([None, None])["completed_games"] == 0
    report = performance([None, 0, .5, 1])
    assert report["completed_games"] == 3
    assert report["wins"] == report["draws"] == report["losses"] == 1
    assert report["relative_elo"] == 0
    assert report["score_interval_95"] == [0, 1]


def test_all_losses_do_not_produce_finite_measured_rating():
    report = performance([0, 0])
    assert report["relative_elo"] is None
    assert report["smoothed_relative_elo"] < 0


@pytest.mark.parametrize("score", [-1, 2, .1, math.nan, math.inf])
def test_invalid_game_scores_rejected(score):
    with pytest.raises(ValueError):
        performance([score])


def test_invalid_game_configuration_rejected_before_execution():
    with pytest.raises(ValueError):
        play_game(None, None, nodes=0)
    with pytest.raises(ValueError):
        play_game(None, None, agent_color=True)


def test_resume_preserves_repetition_and_counters():
    import chess
    from pai.chess.teacher import ChessTeacher, observation
    from pai.chess.state import GameState
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        board.push_uci(uci)
    teacher = ChessTeacher.from_board(board)
    state = GameState.from_belief(teacher.observe())
    assert len(state.history) == 9
    assert state.board == observation(board)
    assert state.halfmove_clock == 8
    assert teacher.board.is_repetition(3)
    assert teacher.board.move_stack == board.move_stack
    assert teacher.board is not board
    assert teacher.record.commands == []


def test_match_progress_survives_interrupted_color_pair(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import pai.chess.play as module

    class Engine:
        id = {"name": "test opponent"}
        options = {"UCI_Elo": SimpleNamespace(min=1000, max=3000)}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def configure(self, options):
            pass

    monkeypatch.setattr(module.chess.engine.SimpleEngine, "popen_uci", lambda *args: Engine())
    calls = []
    def interrupted_game(*args, **kwargs):
        calls.append(kwargs["agent_color"])
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return {"agent_score": 0, "status": "complete", "failure": None,
                "agent_color": "white", "result": "0-1", "moves": 1, "pgn": "*"}, []
    monkeypatch.setattr(module, "play_game", interrupted_game)
    snapshots = []
    with pytest.raises(RuntimeError, match="interrupted"):
        module.stockfish_matches(None, "unused", levels=[1320], budgets=[32], pairs=1,
                                 output_dir=tmp_path, progress=lambda *a, **kw: None,
                                 on_progress=lambda reports: snapshots.append(json.loads(json.dumps(reports))))
    saved = json.loads((tmp_path / "matches.partial.json").read_text())
    assert saved == snapshots[-1]
    assert saved[0]["statistics"]["completed_games"] == 1
    assert len(saved[0]["games"]) == 1
    assert saved[0]["scheduled_games"] == 2
    assert not saved[0]["schedule_complete"]
    assert (tmp_path / "sf1320_n32_p0_white.pgn").exists()
    def finish_game(*args, **kwargs):
        assert kwargs["agent_color"] is False
        return {"agent_score": .5, "status": "complete", "failure": None,
                "agent_color": "black", "result": "1/2-1/2", "moves": 2, "pgn": "*"}, []
    monkeypatch.setattr(module, "play_game", finish_game)
    finished = module.stockfish_matches(None, "unused", levels=[1320], budgets=[32], pairs=1,
                                       existing_reports=saved, progress=lambda *a, **kw: None)
    assert len(finished[0]["games"]) == 2
    assert finished[0]["schedule_complete"]
    assert finished[0]["statistics"]["completed_games"] == 2
    assert len(saved[0]["games"]) == 1  # input snapshot remains unchanged
