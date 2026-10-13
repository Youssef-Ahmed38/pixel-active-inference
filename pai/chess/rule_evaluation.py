"""Independent answer-key scoring of learned rules over complete action spaces."""
from __future__ import annotations

import numpy as np

from pai.chess.rules import ACTIONS
from pai.chess.representation import BoardAction
from pai.chess.teacher import observation, position_key


def evaluate_rules(model, positions, progress=print):
    seen = set(model.training_positions)
    tp = tn = fp = fn = exact_sets = policy_legal = abstained_correct = terminal = 0
    effects_correct = effects_total = 0
    failures, effect_failures, topics = [], [], {}
    used = []
    excluded = 0
    for board, topic in positions:
        obs = observation(board)
        if position_key(obs) in seen:
            excluded += 1
            continue
        key = position_key(obs)
        used.append(key)
        expected = {m.uci() for m in board.legal_moves}
        predicted = model.classify(obs)
        proposed = [ACTIONS[i] for i in np.flatnonzero(predicted)]
        actual = {a.uci() for a in proposed}
        missed, invalid = expected - actual, actual - expected
        tp += len(expected & actual)
        fp += len(invalid)
        fn += len(missed)
        tn += len(ACTIONS) - len(expected | actual)
        exact = actual == expected
        exact_sets += exact
        if expected:
            policy_legal += bool(proposed and proposed[0].uci() in expected)
        else:
            terminal += 1
            abstained_correct += not proposed
        tally = topics.setdefault(topic, {"positions": 0, "exact_legal_sets": 0, "false_positives": 0,
                                         "missed_legal_moves": 0})
        tally["positions"] += 1
        tally["exact_legal_sets"] += int(exact)
        tally["false_positives"] += len(invalid)
        tally["missed_legal_moves"] += len(missed)
        # Piece effects checked against every legal teacher transition, not just chosen moves.
        for move in board.legal_moves:
            after = board.copy(stack=False)
            after.push(move)
            result = model.predict_piece_board(obs, BoardAction.from_uci(move.uci()))
            correct = result == observation(after).pieces
            effects_correct += correct
            if not correct and len(effect_failures) < 30:
                effect_failures.append({"topic": topic, "fen": board.fen(en_passant="fen"), "move": move.uci()})
            effects_total += 1
        if not exact and len(failures) < 30:
            failures.append({"topic": topic, "fen": board.fen(en_passant="fen"), "extra": sorted(invalid)[:12],
                             "missing": sorted(missed)[:12]})
        if len(used) % 100 == 0:
            progress(f"Evaluated {len(used)} positions: {exact_sets} exact sets, {fp} FP, {fn} FN", flush=True)
    if not used:
        raise ValueError("no independent positions remain after excluding training positions")
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    negative_recall = tn / max(tn + fp, 1)
    return {"positions": len(used), "excluded_training_positions": excluded, "position_overlap": 0,
            "actions_scored": len(used) * len(ACTIONS), "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "legal_precision": precision, "legal_recall": recall, "illegal_recall": negative_recall,
            "balanced_accuracy": (recall + negative_recall) / 2,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "exact_legal_sets": exact_sets, "exact_legal_set_rate": exact_sets / len(used),
            "unmasked_policy": {"legal_choices": policy_legal, "playable_positions": len(used) - terminal,
                                "terminal_positions": terminal, "correct_abstentions": abstained_correct},
            "piece_effects": {"exact": effects_correct, "total": effects_total,
                              "accuracy": effects_correct / max(effects_total, 1)},
            "rule_gate_passed": min(precision, recall, negative_recall) > .99,
            "strict_suite_passed": fp == fn == 0 and effects_correct == effects_total,
            "topics": topics, "failures": failures, "effect_failures": effect_failures}
