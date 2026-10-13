"""Teacher-only state supervision and independent transition evaluation."""
import random

import chess
import numpy as np

from pai.chess.representation import BoardAction
from pai.chess.rule_training import _fit
from pai.chess.state import GameState, LearnedChessStateModel, transition_features
from pai.chess.teacher import observation, position_key


def game_state(board, include_history=True):
    """Export counters and available history; never fabricate pre-FEN history."""
    history = []
    if include_history:
        root = board.copy(stack=True)
        history.append(observation(root))
        while root.move_stack:
            root.pop()
            history.append(observation(root))
        history.reverse()
    return GameState(observation(board), board.halfmove_clock, board.fullmove_number, tuple(history))


def train_state_model(rules, positions, seed=0):
    xs, labels = [], {name: [] for name in
                      ("turn", "ep_offset", "reset_clock", "fullmove_delta", *(f"castle_{i}" for i in range(4)))}
    keys = []
    for board, _ in positions:
        obs = observation(board)
        moves = list(board.legal_moves)
        if not moves:
            continue
        keys.append(position_key(obs))
        actions = [BoardAction.from_uci(m.uci()) for m in moves]
        xs.append(transition_features(obs, [(a.source, a.destination, a.promotion) for a in actions]))
        for move in moves:
            after = board.copy(stack=False)
            after.push(move)
            next_obs = observation(after)
            labels["turn"].append(next_obs.turn)
            labels["ep_offset"].append(64 if next_obs.ep_square == 64 else next_obs.ep_square - move.from_square)
            labels["reset_clock"].append(after.halfmove_clock == 0)
            labels["fullmove_delta"].append(after.fullmove_number - board.fullmove_number)
            for i, flag in enumerate(next_obs.castling):
                labels[f"castle_{i}"].append(flag)
    if not xs:
        raise ValueError("state teaching requires legal transition examples")
    x = np.vstack(xs)
    # Each head sees only relevant generic channels, preventing absolute-square
    # correlations in geometric effects and clock rules.
    channels = {"turn": [2], "fullmove_delta": [2], "ep_offset": [0, 3, 4],
                "reset_clock": list(range(6, 32))}
    for i in range(4):
        channels[f"castle_{i}"] = list(range(6, 160)) + [160 + i]
    trees = {name: _fit(x, y, seed, channels[name]) for name, y in labels.items()}
    fit_errors = {name: int(np.count_nonzero(tree.predict(x) != labels[name])) for name, tree in trees.items()}
    if any(fit_errors.values()):
        raise RuntimeError(f"state model failed teaching fit: {fit_errors}")
    return LearnedChessStateModel(rules, trees, keys,
                                  {"seed": seed, "positions": len(keys), "transitions": len(x),
                                   "fit_errors": fit_errors, "tree_nodes": {k: len(v.feature) for k, v in trees.items()}})


def evaluate_state_model(model, positions, seed=0):
    rng = random.Random(seed)
    excluded = set(model.training_positions) | set(model.rules.training_positions)
    fields = ("pieces", "turn", "castling", "ep_square", "halfmove_clock", "fullmove_number", "history")
    errors = dict.fromkeys(fields, 0)
    count = exact = boards = overlaps = 0
    failures = []
    topics = {}
    for original, topic in positions:
        if position_key(observation(original)) in excluded:
            overlaps += 1
            continue
        board = original.copy(stack=True)
        # Counters well outside training values verify delta/reset extrapolation.
        board.halfmove_clock = rng.randrange(200)
        board.fullmove_number = rng.randrange(1, 500)
        before = game_state(board)
        moves = list(board.legal_moves)
        actions = [BoardAction.from_uci(m.uci()) for m in moves]
        predictions = model.successors(before, actions)
        boards += 1
        tally = topics.setdefault(topic, {"transitions": 0, "exact": 0})
        for move, predicted in zip(moves, predictions):
            after = board.copy(stack=True)
            after.push(move)
            expected = game_state(after)
            wrong = []
            for field in fields:
                target_p, target_e = (predicted.board, expected.board) if field in fields[:4] else (predicted, expected)
                if getattr(target_p, field) != getattr(target_e, field):
                    errors[field] += 1
                    wrong.append(field)
            count += 1
            exact += not wrong
            tally["transitions"] += 1
            tally["exact"] += not wrong
            if wrong and len(failures) < 20:
                failures.append({"fen": board.fen(en_passant="fen"), "move": move.uci(), "fields": wrong})
    return {"positions": boards, "excluded_training_positions": overlaps, "position_overlap": 0,
            "transitions": count, "exact_transitions": exact, "field_errors": errors,
            "strict_suite_passed": bool(count) and count == exact, "topics": topics, "failures": failures}


def evaluate_rollouts(model, positions, seed=0, games=10, plies=40):
    """Advance the model's own state, checking legal sets and state after each ply."""
    rng = random.Random(seed)
    excluded = set(model.training_positions) | set(model.rules.training_positions)
    roots = [b for b, topic in positions if topic == "game" and position_key(observation(b)) not in excluded]
    rng.shuffle(roots)
    steps = legal_sets = exact = completed = 0
    failures = []
    for root in roots[:games]:
        board = root.copy(stack=True)
        state = game_state(board)
        for _ in range(plies):
            predicted = model.rules.legal_actions(state.board)
            expected = {m.uci() for m in board.legal_moves}
            if {a.uci() for a in predicted} != expected:
                failures.append({"fen": board.fen(en_passant="fen"), "kind": "legal_set"})
                break
            legal_sets += 1
            if not predicted:
                completed += 1
                break
            action = rng.choice(predicted)
            state = model.predict(state, action)
            board.push(chess.Move.from_uci(action.uci()))
            steps += 1
            if state != game_state(board):
                failures.append({"fen": board.fen(en_passant="fen"), "kind": "state"})
                break
            exact += 1
        else:
            completed += 1
    return {"rollouts": min(games, len(roots)), "completed": completed, "steps": steps,
            "exact_states": exact, "exact_legal_sets": legal_sets, "failures": failures,
            "strict_suite_passed": bool(steps) and not failures and completed == min(games, len(roots))}
