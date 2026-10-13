"""Teacher-only supervision for endings and engine-distilled position value."""
from collections import Counter
import hashlib
import random

import chess
import chess.engine
import numpy as np
from sklearn.linear_model import Ridge

from pai.chess.evaluation import (LearnedChessEvaluator, LearnedPositionValue, material_features, value_features)
from pai.chess.rule_training import _fit, teaching_positions
from pai.chess.state import transition_features
from pai.chess.state_training import game_state
from pai.chess.teacher import LESSONS, observation, position_key


def state_key(state):
    return hashlib.sha256(repr(state).encode()).hexdigest()


def teacher_key(board):
    obs = observation(board)
    return obs.pieces, obs.turn, obs.castling, obs.ep_square if board.has_legal_en_passant() else 64


def ending_positions(seed=0, material_samples=2400):
    rng = random.Random(seed)
    rows = list(teaching_positions(seed, 6, 64, 40, 40))
    for _ in range(material_samples):
        board = chess.Board(None)
        board.turn = rng.choice((True, False))
        squares = rng.sample(range(64), rng.randrange(2, 10))
        board.set_piece_at(squares[0], chess.Piece(chess.KING, True))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, False))
        for square in squares[2:]:
            kind = rng.choice((chess.BISHOP, chess.KNIGHT)) if rng.random() < .85 else rng.randrange(1, 6)
            if kind == chess.PAWN and chess.square_rank(square) in (0, 7):
                kind = chess.BISHOP
            board.set_piece_at(square, chess.Piece(kind, rng.choice((True, False))))
        if board.is_valid():
            rows.append((board, "material"))
    # History includes positions before a move that can complete a repetition.
    bases = [chess.Board(), chess.Board("4k2r/8/8/8/8/8/8/R3K3 w - - 0 1")]
    for _ in range(8):
        base = chess.Board()
        for color in (True, False):
            file = rng.choice((0, 1, 2, 3, 4, 7))
            source = file + (8 if color else 48)
            destination = source + rng.choice((8, 16)) * (1 if color else -1)
            base.push(chess.Move(source, destination))
        bases.append(base)
    for base in bases:
        base = base.copy(stack=False)
        cycle = "g1f3 g8f6 f3g1 f6g8" if base.piece_at(chess.G1) else "a1a2 h8h7 a2a1 h7h8"
        for mirrored in (False, True):
            board = base.mirror() if mirrored else base.copy()
            moves = [chess.Move.from_uci(u) for u in cycle.split()]
            if mirrored:
                moves = [chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square)) for m in moves]
            for i in range(24):
                rows.append((board.copy(stack=True), "repetition"))
                board.push(moves[i % 4])
    expanded = []
    for board, topic in rows:
        for clock in (0, 98, 99, 100, 101, 149, 150, 151, 201):
            if board.move_stack:
                # Counter overrides must also be coherent with replayed stack
                # entries; these generated cycles contain only reversible moves.
                b = board.root()
                b.halfmove_clock = max(0, clock - len(board.move_stack))
                for move in board.move_stack:
                    b.push(move)
            else:
                b = board.copy(stack=False)
                b.halfmove_clock = clock
            expanded.append((b, topic))
    return expanded


def teacher_features(board, claim_draw):
    history = board.copy(stack=True)
    counts = Counter([teacher_key(history)])
    while history.move_stack:
        history.pop()
        counts[teacher_key(history)] += 1
    moves = list(board.legal_moves)
    next_reps = next_clock = 0
    if claim_draw and (board.halfmove_clock >= 99 or max(counts.values()) >= 2):
        for move in moves:
            after = board.copy(stack=True)
            after.push(move)
            next_reps = max(next_reps, counts[teacher_key(after)] + 1)
            if any(after.legal_moves):
                next_clock = max(next_clock, after.halfmove_clock)
    return [len(moves), int(not board.is_check()), int(board.is_insufficient_material()), board.halfmove_clock,
            counts[teacher_key(board)], next_reps, next_clock, int(board.turn), int(claim_draw)], max(counts.values())


def train_evaluator(state_model, seed=0, material_samples=2400):
    positions = ending_positions(seed, material_samples)
    # Deduplicate public material evidence before fitting.
    materials = {}
    for board, _ in positions:
        row = material_features(observation(board))
        materials[tuple(row)] = board.is_insufficient_material()
    material = _fit(list(materials), list(materials.values()), seed, list(range(36, 54)))
    ep_x, ep_y = [], []
    for board, _ in teaching_positions(seed, 12, 64, 40, 200):
        obs = observation(board)
        moves = list(board.legal_moves)
        if not moves:
            continue
        actions = np.array([(m.from_square, m.to_square, 0 if m.promotion is None else m.promotion - 1) for m in moves])
        ep_x.append(np.column_stack((transition_features(obs, actions), actions[:, 1] == obs.ep_square)))
        ep_y.extend(board.is_en_passant(m) for m in moves)
    ep = _fit(np.vstack(ep_x), ep_y, seed, [0, 3, 4, 164])
    reasons = ("ONGOING", "CHECKMATE", "INSUFFICIENT_MATERIAL", "STALEMATE", "SEVENTYFIVE_MOVES",
               "FIVEFOLD_REPETITION", "FIFTY_MOVES", "THREEFOLD_REPETITION")
    x, y, winner_x, winner_y, gate_x, gate_y, keys, current_x, current_y = [], [], [], [], [], [], [], [], []
    for board, _ in positions:
        keys.append(state_key(game_state(board)))
        for claim in (False, True):
            row, maxhistory = teacher_features(board, claim)
            if not claim:
                current = board.copy(stack=True)
                current_reason = "FIFTY_MOVES" if current.is_fifty_moves() else (
                    "THREEFOLD_REPETITION" if current.is_repetition(3) else "ONGOING")
                current_x.append(row)
                current_y.append(reasons.index(current_reason))
            # History queries replay stacks. Synthetic counter overrides must
            # remain identical between claim modes and supervision channels.
            answer = board.copy(stack=True).outcome(claim_draw=claim)
            reason = "ONGOING" if answer is None else answer.termination.name
            code = reasons.index(reason)
            x.append(row)
            y.append(code)
            winner_x.append([code, int(board.turn)])
            winner_y.append(-1 if answer is None or answer.winner is None else int(answer.winner))
            gate_x.append([board.halfmove_clock, maxhistory])
            # Teacher-side broad eligibility, before evaluating actual moves.
            gate_y.append(board.halfmove_clock >= 99 or maxhistory >= 2)
    trees = {"material": material, "ep_move": ep, "outcome": _fit(x, y, seed),
             "winner": _fit(winner_x, winner_y, seed), "lookahead": _fit(gate_x, gate_y, seed),
             "current_claim": _fit(current_x, current_y, seed, [0, 3, 4])}
    errors = int(np.count_nonzero(trees["outcome"].predict(np.asarray(x)) != y))
    if errors:
        raise RuntimeError("terminal teaching did not fit")
    return LearnedChessEvaluator(state_model, trees, reasons, sorted(set(keys)),
                                  {"seed": seed, "boards": len(positions), "outcome_examples": len(y),
                                   "material_patterns": len(materials), "ep_examples": len(ep_y),
                                   "tree_nodes": {k: len(t.feature) for k, t in trees.items()}})


def evaluate_endings(model, seed=310, material_samples=240):
    training = set(model.training_keys)
    total = correct = excluded = 0
    confusion, failures = {}, []
    canonical = {}
    current_total = current_correct = 0
    for board, topic in ending_positions(seed, material_samples):
        state = game_state(board)
        canonical.setdefault(state.board, model.repetition_key(state.board) == teacher_key(board))
        if state_key(state) in training:
            excluded += 1
            continue
        current_board = board.copy(stack=True)
        current_reason = "FIFTY_MOVES" if current_board.is_fifty_moves() else (
            "THREEFOLD_REPETITION" if current_board.is_repetition(3) else None)
        current_total += 1
        current_correct += model.current_claim(state) == current_reason
        for claim in (False, True):
            expected = board.copy(stack=True).outcome(claim_draw=claim)
            name = "ONGOING" if expected is None else expected.termination.name
            winner = None if expected is None or expected.winner is None else int(expected.winner)
            actual = model.status(state, claim_draw=claim)
            okay = (actual.termination or "ONGOING", actual.winner) == (name, winner)
            total += 1
            correct += okay
            tally = confusion.setdefault(name, {"total": 0, "correct": 0})
            tally["total"] += 1
            tally["correct"] += okay
            if not okay and len(failures) < 12:
                failures.append({"fen": board.fen(en_passant="fen"), "topic": topic, "claim_draw": claim,
                                 "expected": name, "actual": actual.termination,
                                 "features": model.features(state, claim).tolist()})
    return {"examples": total, "correct": correct, "excluded_training_states": excluded,
            "position_overlap": 0, "by_outcome": confusion, "failures": failures,
            "canonical_positions": len(canonical), "canonical_correct": sum(canonical.values()),
            "current_claim_examples": current_total, "current_claim_correct": current_correct,
            "strict_suite_passed": bool(total) and total == correct and all(canonical.values()) and current_total == current_correct}


def engine_labels(path, positions, nodes=2000, progress=print):
    labels = []
    with chess.engine.SimpleEngine.popen_uci(str(path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        name = engine.id.get("name", "unknown")
        for i, (board, topic) in enumerate(positions):
            info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
            score = info["score"].white().score(mate_score=10000)
            labels.append((observation(board), float(np.clip(score, -2000, 2000)), topic))
            if (i + 1) % 200 == 0:
                progress(f"Engine labelled {i + 1}/{len(positions)} positions", flush=True)
    return labels, {"engine": name, "nodes_per_position": nodes, "threads": 1, "score_clip_cp": 2000}


def fit_value(rows, teacher_summary):
    x = np.array([value_features(obs) for obs, _, _ in rows])
    y = np.array([score for _, score, _ in rows])
    # A linear baseline gives interpretable learned material and square values.
    fitted = Ridge(alpha=20, fit_intercept=False).fit(x, y)
    summary = {**teacher_summary, "examples": len(rows), "features": x.shape[1], "ridge_alpha": 20,
               "training_positions": [position_key(obs) for obs, _, _ in rows]}
    return LearnedPositionValue(fitted.coef_, summary)


def evaluate_value(model, rows):
    keys = set(model.summary["training_positions"])
    fresh = [(o, y, t) for o, y, t in rows if position_key(o) not in keys]
    actual = np.array([y for _, y, _ in fresh])
    predicted = np.array([model.predict(o) for o, _, _ in fresh])
    baseline = np.mean(np.abs(actual))
    mae = np.mean(np.abs(predicted - actual))
    mask = np.abs(actual) >= 100
    return {"positions": len(fresh), "excluded_training_positions": len(rows) - len(fresh),
            "mae_cp": float(mae), "zero_baseline_mae_cp": float(baseline),
            "rmse_cp": float(np.sqrt(np.mean((predicted - actual) ** 2))),
            "advantage_sign_accuracy": float(np.mean((predicted[mask] > 0) == (actual[mask] > 0))),
            "advantage_sign_examples": int(mask.sum()), "beats_zero_baseline": bool(mae < baseline)}


def evaluate_value_policy(value, evaluator, positions, engine_path, nodes=2000, limit=24):
    training = set(value.summary["training_positions"])
    cases = [(b, t) for b, t in positions if position_key(observation(b)) not in training and b.outcome() is None][:limit]
    gains, legal, failures = [], 0, []
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64})
        for board, _ in cases:
            state = game_state(board)
            action = value.choose(evaluator.state_model, evaluator, state)
            baseline = evaluator.state_model.rules.choose(state.board)
            if action is None or not board.is_legal(chess.Move.from_uci(action.uci())):
                failures.append(board.fen())
                continue
            legal += 1
            scores = []
            for candidate in (baseline, action):
                child = board.copy(stack=True)
                child.push(chess.Move.from_uci(candidate.uci()))
                answer = child.outcome()
                if answer:
                    score = 0 if answer.winner is None else (10000 if answer.winner else -10000)
                else:
                    engine.configure({"Clear Hash": None})
                    info = engine.analyse(child, chess.engine.Limit(nodes=nodes))
                    score = info["score"].white().score(mate_score=10000)
                scores.append(score if board.turn else -score)
            gains.append(scores[1] - scores[0])
    return {"positions": len(cases), "legal_choices": legal, "failures": failures,
            "comparison": "one-ply value policy versus first learned legal action",
            "teacher_nodes": nodes, "improved": sum(g > 0 for g in gains),
            "tied": sum(g == 0 for g in gains), "worse": sum(g < 0 for g in gains),
            "mean_teacher_score_gain_cp": float(np.mean(gains)) if gains else None,
            "legal_choice_gate_passed": bool(cases) and legal == len(cases)}
