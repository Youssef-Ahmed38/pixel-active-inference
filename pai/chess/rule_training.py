"""Teacher-side induction and correction of relational chess rules."""
from __future__ import annotations

import random

import chess
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from pai.chess.rules import (ACTION_ARRAY, BETWEEN, CategoricalTrees, LearnedChessRules, LearnedTree,
                             safety_features, spatial_features)
from pai.chess.teacher import LESSONS, observation, position_key


def _fit(x, y, seed, features=None):
    model = DecisionTreeClassifier(random_state=seed, criterion="entropy")
    x = np.asarray(x, dtype=np.float32)
    model.fit(x if features is None else x[:, features], y)
    learned = LearnedTree.from_classifier(model)
    if features is not None:
        learned.feature = learned.feature.copy()
        active = learned.feature >= 0
        learned.feature[active] = np.array(features)[learned.feature[active]]
    return learned


def _fit_proposals(x, y, seed, features):
    x, y = np.asarray(x), np.asarray(y)
    # Token IDs are categories, not an ordered scale. No movement rule is
    # assigned to any category; each decision tree is fitted only from examples.
    categories = x[:, 0].astype(int) * 2 + x[:, 39].astype(int)
    groups, guards = {}, {}
    for code in np.unique(categories):
        selected = categories == code
        groups[int(code)] = _fit(x[selected], y[selected], seed, features)
        positive = x[selected & y.astype(bool)]
        # Induce constant constraints on generic clearance/ownership evidence.
        # Neither a required value nor a piece-specific rule is hardcoded.
        if len(positive):
            guards[int(code)] = [(column, float(positive[0, column])) for column in (12, 40, 41, 45)
                                 if np.all(positive[:, column] == positive[0, column])]
    return CategoricalTrees(0, groups, 39, guards)


def teach_attacks(seed=0):
    """Exhaustive isolated-piece teaching, with and without an intervening token.

    All labels come from the teacher's attack API. The fitted tree is compiled
    into a fast finite lookup table; the runtime never receives teacher code.
    """
    features, labels, locations = [], [], []
    for code in range(13):
        for source in range(64):
            board = chess.Board(None)
            if code:
                board.set_piece_at(source, chess.Piece((code - 1) % 6 + 1, code <= 6))
            initial = board.attacks(source)
            for target in range(64):
                dx, dy = target % 8 - source % 8, target // 8 - source // 8
                path = np.flatnonzero(BETWEEN[source, target])
                for blocked in range(2):
                    label = target in initial
                    if blocked and len(path):
                        index = int(path[len(path) // 2])
                        board.set_piece_at(index, chess.Piece(chess.PAWN, chess.WHITE))
                        label = target in board.attacks(source)
                        board.remove_piece_at(index)
                    features.append((code, dx, dy, abs(dx), abs(dy), blocked))
                    labels.append(label)
                    locations.append((code, source, target, blocked))
    tree = _fit(features, labels, seed)
    predicted = tree.predict(np.asarray(features))
    if not np.array_equal(predicted, labels):
        raise RuntimeError("attack teaching did not fit its exhaustive finite curriculum")
    table = np.zeros((13, 64, 64, 2), dtype=np.int8)
    loc = np.array(locations).T
    table[tuple(loc)] = predicted
    return table, {"examples": len(labels), "tree_nodes": len(tree.feature), "fit_accuracy": 1.0}


def refine_destination_constraints(model, positions):
    """Learn occupancy invariants from the original positive teaching examples.

    A destination token's numeric ID is not an ordered measure of occupancy.
    Add a binary occupancy constraint only where teacher examples support a
    constant value for that learned category. No piece-specific values are set.
    """
    evidence = {}
    for board, _topic in positions:
        moves = list(board.pseudo_legal_moves)
        if not moves:
            continue
        actions = np.array([(m.from_square, m.to_square, 0 if m.promotion is None else m.promotion - 1)
                            for m in moves])
        x = spatial_features(observation(board), actions, model.attack_table)
        for row in x:
            category = int(row[0]) * 2 + int(row[39])
            evidence.setdefault(category, set()).add(float(row[46]))
    if model.proposal.relation_column != 39:
        raise ValueError("destination refinement requires the relational categorical model")
    count = 0
    for category, values in evidence.items():
        guards = model.proposal.guards.setdefault(category, [])
        guards[:] = [(col, value) for col, value in guards if col != 46]
        if len(values) == 1:
            guards.append((46, next(iter(values))))
            count += 1
    model.training_summary["destination_occupancy_constraints"] = count
    return model


def teaching_positions(seed=0, games=60, plies=80, sparse=400, special=200, capture_lessons=False):
    """Valid games and generated special-rule positions; independent by seed."""
    rng = random.Random(seed)
    positions = {}

    def add(board, topic):
        if board.is_valid():
            positions.setdefault(position_key(observation(board)), (board.copy(stack=False), topic))

    for topic, fen in LESSONS.items():
        add(chess.Board(fen), topic)
        add(chess.Board(fen).mirror(), topic)
    if capture_lessons:
        # Random games rarely put a capturable major piece beside a pawn on
        # its initial rank. Teach all ranks/files and captured token types.
        for color in (chess.WHITE, chess.BLACK):
            for rank in range(1, 7):
                for file in range(8):
                    for target_file in (file - 1, file + 1):
                        if not 0 <= target_file < 8:
                            continue
                        source = file + rank * 8
                        target = target_file + (rank + (1 if color else -1)) * 8
                        for kind in range(1, 6):
                            for _attempt in range(12):
                                board = chess.Board(None)
                                board.turn = color
                                board.set_piece_at(source, chess.Piece(chess.PAWN, color))
                                board.set_piece_at(target, chess.Piece(kind, not color))
                                kings = rng.sample([s for s in range(64) if s not in (source, target)], 2)
                                board.set_piece_at(kings[0], chess.Piece(chess.KING, chess.WHITE))
                                board.set_piece_at(kings[1], chess.Piece(chess.KING, chess.BLACK))
                                if board.is_valid():
                                    add(board, "pawn_capture_lessons")
                                    break
    for _ in range(games):
        board = chess.Board()
        for ply in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            if ply % 4 == 0:
                add(board, "game")
            board.push(rng.choice(legal))
    for _ in range(sparse * 3):
        board = chess.Board(None)
        board.turn = bool(rng.randrange(2))
        slots = rng.sample(range(64), rng.randrange(3, 15))
        board.set_piece_at(slots[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(slots[1], chess.Piece(chess.KING, chess.BLACK))
        for square in slots[2:]:
            kind = rng.randrange(1, 6)
            if kind == chess.PAWN and square // 8 in (0, 7):
                kind = chess.KNIGHT
            board.set_piece_at(square, chess.Piece(kind, bool(rng.randrange(2))))
        add(board, "sparse")
        if sum(topic == "sparse" for _, topic in positions.values()) >= sparse:
            break
    for _ in range(special):
        if _ % 10 == 0:
            for topic in ("checkmate", "stalemate"):
                board = chess.Board(LESSONS[topic])
                for square in rng.sample(list(range(32)), rng.randrange(1, 4)):
                    board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
                if rng.random() < .5:
                    board = board.mirror()
                terminal = board.is_checkmate() if topic == "checkmate" else board.is_stalemate()
                if terminal:
                    add(board, topic)
        # Keep both rooks, vary permissions, clutter, attacks, and side to move.
        board = chess.Board(LESSONS["castling"])
        board.turn = bool(rng.randrange(2))
        board.castling_rights = sum(1 << s for s in (0, 7, 56, 63) if rng.random() < .7)
        for square in rng.sample(list(range(8, 56)) + [1, 2, 3, 5, 6, 57, 58, 59, 61, 62], rng.randrange(5)):
            board.set_piece_at(square, chess.Piece(rng.randrange(1, 6), bool(rng.randrange(2))))
        add(board, "castling")
        # Generate en passant with different files, kings and extra pieces.
        board = chess.Board(None)
        file = rng.randrange(8)
        neighbor = rng.choice([f for f in (file - 1, file + 1) if 0 <= f < 8])
        board.set_piece_at(file + 32, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(neighbor + 32, chess.Piece(chess.PAWN, chess.WHITE))
        board.ep_square = file + 40
        reserved = {file + 32, neighbor + 32, file + 40, file + 48}
        slots = rng.sample([s for s in range(64) if s not in reserved], 5)
        board.set_piece_at(slots[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(slots[1], chess.Piece(chess.KING, chess.BLACK))
        for square in slots[2:]:
            board.set_piece_at(square, chess.Piece(rng.randrange(2, 6), bool(rng.randrange(2))))
        if rng.random() < .5:
            board = board.mirror()
        add(board, "en_passant")
        # Both quiet and capturing promotions, including every underpromotion.
        board = chess.Board(None)
        file = rng.randrange(8)
        board.set_piece_at(file + 48, chess.Piece(chess.PAWN, chess.WHITE))
        slots = rng.sample([s for s in range(64) if s != file + 48], 5)
        board.set_piece_at(slots[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(slots[1], chess.Piece(chess.KING, chess.BLACK))
        for square in slots[2:]:
            board.set_piece_at(square, chess.Piece(rng.randrange(2, 6), bool(rng.randrange(2))))
        for f in (file - 1, file, file + 1):
            if 0 <= f < 8 and not board.piece_at(f + 56) and rng.random() < .5:
                board.set_piece_at(f + 56, chess.Piece(rng.randrange(2, 6), chess.BLACK))
        if rng.random() < .5:
            board = board.mirror()
        add(board, "promotion")
    return list(positions.values())


def train_rules(positions, seed=0, corrections=12, samples=256, progress=print):
    rng = np.random.default_rng(seed)
    attack_table, attack_summary = teach_attacks(seed)
    progress(f"Attack induction: {attack_summary['examples']} examples, exact finite fit", flush=True)
    proposal_x, proposal_y, effects_x, moved_y, pattern_y, safety_x, safety_y = [], [], [], [], [], [], []
    patterns = []
    pattern_ids = {}
    cached = []
    for index, (board, topic) in enumerate(positions):
        obs = observation(board)
        pseudo = list(board.pseudo_legal_moves)
        positive = [(m.from_square, m.to_square, 0 if m.promotion is None else m.promotion - 1) for m in pseudo]
        # Half arbitrary syntax, half occupied sources; all teacher proposals included.
        arbitrary = ACTION_ARRAY[rng.choice(len(ACTION_ARRAY), samples // 2, replace=False)]
        occupied = list(board.piece_map())
        hard = np.column_stack((rng.choice(occupied, samples // 2), rng.integers(64, size=samples // 2),
                                np.where(rng.random(samples // 2) < .7, 0, rng.integers(1, 5, size=samples // 2))))
        hard = hard[hard[:, 0] != hard[:, 1]]
        actions = np.unique(np.vstack((arbitrary, hard, np.asarray(positive).reshape(-1, 3))), axis=0).astype(np.int16)
        x = spatial_features(obs, actions, attack_table)
        pseudo_set = set(positive)
        y = np.array([tuple(a) in pseudo_set for a in actions])
        proposal_x.append(x)
        proposal_y.append(y)
        moves = {key: move for key, move in zip(positive, pseudo)}
        for action, feature in zip(actions[y], x[y]):
            move = moves[tuple(action)]
            after = board.copy(stack=False)
            after.push(move)
            following = observation(after)
            src, dst, _ = action
            edits = tuple((i - int(src), following.pieces[i]) for i in range(64)
                          if i not in (src, dst) and following.pieces[i] != obs.pieces[i])
            if edits not in pattern_ids:
                pattern_ids[edits] = len(patterns)
                patterns.append(edits)
            effects_x.append(feature)
            moved_y.append(following.pieces[dst])
            pattern_y.append(pattern_ids[edits])
            safety_x.append(safety_features(following.pieces, obs.turn, attack_table))
            safety_y.append(board.is_legal(move))
        cached.append((board, obs))
        if (index + 1) % 300 == 0:
            progress(f"Prepared {index + 1}/{len(positions)} teaching positions", flush=True)
    proposal_features = list(range(15)) + [16] + list(range(23, 35)) + list(range(39, 46))
    model = LearnedChessRules(attack_table, _fit_proposals(np.vstack(proposal_x), np.concatenate(proposal_y), seed, proposal_features),
                              _fit(effects_x, moved_y, seed, [0, 3]),
                              _fit(effects_x, pattern_y, seed, [0, 8, 9, 16]), patterns,
                              _fit(safety_x, safety_y, seed), [position_key(obs) for _, obs in cached])
    corrected = []
    # Mine wrong predictions from the COMPLETE action space on training boards only.
    for round in range(corrections):
        new_x, new_y = [], []
        for board, obs in cached:
            x = spatial_features(obs, ACTION_ARRAY, attack_table)
            predicted = model.proposal.predict(x).astype(bool)
            truth = np.zeros(len(ACTION_ARRAY), dtype=bool)
            for move in board.pseudo_legal_moves:
                p = 0 if move.promotion is None else move.promotion - 1
                # Enumeration excludes src==dst, then promotion is innermost.
                offset = move.to_square if move.to_square < move.from_square else move.to_square - 1
                truth[(move.from_square * 63 + offset) * 5 + p] = True
            wrong = np.flatnonzero(predicted != truth)
            if len(wrong):
                new_x.append(x[wrong])
                new_y.append(truth[wrong])
        count = sum(len(y) for y in new_y)
        corrected.append(count)
        progress(f"Correction round {round + 1}: {count} wrong full-space proposals", flush=True)
        if not count:
            break
        proposal_x.extend(new_x)
        proposal_y.extend(new_y)
        model.proposal = _fit_proposals(np.vstack(proposal_x), np.concatenate(proposal_y), seed, proposal_features)
    model.training_summary = {"seed": seed, "positions": len(positions), "attack": attack_summary,
                              "proposal_examples": sum(len(y) for y in proposal_y),
                              "effect_examples": len(effects_x), "edit_patterns": len(patterns),
                              "correction_errors": corrected,
                              "tree_nodes": {k: len(getattr(model, k).feature) for k in
                                             ("proposal", "moved_token", "extra_edits", "safety")}}
    refine_destination_constraints(model, positions)
    return model
