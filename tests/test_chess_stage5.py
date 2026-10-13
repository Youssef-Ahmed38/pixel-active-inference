import ast
from pathlib import Path

import pytest
import torch

chess = pytest.importorskip("chess")

from pai.chess.learner import ChessWorldModel, evaluate, fit, tensors
from pai.chess.representation import BoardAction, BoardObservation, candidate_actions
from pai.chess.teacher import ChessTeacher, LESSONS, curriculum, observation, position_key, split_examples
from pai.cognition.contracts import SkillCommand
from pai.cognition.memory import CognitiveMemory


def test_illegal_attempt_preserves_board_and_records_correction(tmp_path):
    teacher = ChessTeacher()
    before = observation(teacher.board)
    assert BoardObservation.from_belief(teacher.observe()) == before
    outcome = teacher.execute(BoardAction.from_uci("e2e5").command())
    assert outcome.stalled and not outcome.executed
    assert observation(teacher.board) == before
    assert "e2e4" in teacher.record.events[-1]["correction"]
    assert teacher.execute(BoardAction.from_uci("e2e4").command()).executed
    memory = CognitiveMemory(tmp_path / "episodes.jsonl")
    memory.add(teacher.record)
    assert memory.recall(teacher.record.goal, domain="chess") == [teacher.record]


@pytest.mark.parametrize("topic,uci", [("castling", "e1g1"), ("en_passant", "e5d6"),
                                      ("promotion", "a7a8q"), ("capture", "d4c5")])
def test_special_move_teacher_effects(topic, uci):
    teacher = ChessTeacher(LESSONS[topic])
    expected = teacher.board.copy()
    expected.push_uci(uci)
    result = teacher.execute(BoardAction.from_uci(uci).command())
    assert result.executed
    assert observation(teacher.board) == observation(expected)
    assert len(result.changed) >= 2


def test_terminal_and_malformed_commands():
    for topic in ("checkmate", "stalemate"):
        teacher = ChessTeacher(LESSONS[topic])
        assert not teacher.demonstrate()
        assert not teacher.execute(SkillCommand("move_piece", arguments=("0000",))).executed
    assert not ChessTeacher().execute(SkillCommand("move_piece", arguments=("bad",))).executed


def test_action_roundtrip_and_teacher_free_candidate_space():
    for uci in ("e2e4", "a7a8n", "a7a8b", "a7a8r", "a7a8q"):
        assert BoardAction.from_uci(uci).uci() == uci
    candidates = candidate_actions()
    assert len(candidates) == 64 * 63 * 5
    assert BoardAction.from_uci("e2e5") in candidates
    with pytest.raises(ValueError):
        BoardAction.from_uci("a7a8k")


def test_curriculum_disjoint_deterministic_and_valid_effects():
    rows = curriculum(seed=8, games=2, plies=5)
    assert rows == curriculum(seed=8, games=2, plies=5)
    train, test = split_examples(rows)
    assert not {position_key(r.before) for r in train} & {position_key(r.before) for r in test}
    assert {r.legal for r in test} == {True, False}
    assert all(r.before == r.after for r in rows if not r.legal)
    assert any(r.before != r.after for r in rows if r.legal)
    assert all(chess.Board(fen).is_valid() for fen in LESSONS.values())


def test_model_training_checkpoint_and_nontrivial_gate(tmp_path):
    torch.manual_seed(2)
    torch.set_num_threads(2)
    train, test = split_examples(curriculum(seed=2, games=2, plies=4))
    model = ChessWorldModel(hidden=32)
    weights = model.legality.weight.detach().clone()
    history = fit(model, train, epochs=1)
    assert history[0]["loss"] > 0
    assert not torch.equal(weights, model.legality.weight)
    report = evaluate(model, test)
    assert sum(report["confusion"].values()) == len(test)
    assert not report["rule_gate_passed"]
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = ChessWorldModel(hidden=32)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert evaluate(restored, test) == report
    action, confidence = restored.choose(test[0].before)
    assert action in candidate_actions() and 0 <= confidence <= 1


def test_agent_sources_do_not_import_teacher_rules():
    # Structural boundary: inference modules cannot silently obtain an oracle mask.
    for name in ("learner.py", "representation.py"):
        tree = ast.parse((Path(__file__).parents[1] / "pai" / "chess" / name).read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        assert not any(i == "chess" or i.startswith("chess.") or i.endswith(".teacher") for i in imports)


def test_reject_everything_cannot_pass_accuracy_gate():
    rows = curriculum(seed=1, games=1, plies=2)
    model = ChessWorldModel(hidden=16)
    with torch.no_grad():
        model.legality.weight.zero_()
        model.legality.bias.fill_(-100)
    report = evaluate(model, rows)
    assert report["legal_recall"] == 0
    assert report["balanced_accuracy"] == .5
    assert not report["rule_gate_passed"]
    assert not evaluate(model, [r for r in rows if not r.legal])["rule_gate_passed"]
