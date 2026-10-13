"""Teacher-side tactical, budget and skill-execution checks for learned search."""
from dataclasses import asdict
import random

import chess

from pai.chess.representation import BoardAction
from pai.chess.state_training import game_state
from pai.chess.teacher import ChessTeacher, observation, position_key


def tactical_positions(seed, per_topic=4):
    rng = random.Random(seed)
    rows, seen = [], set()
    for topic in ("poisoned_rook", "mate_in_one"):
        while sum(t == topic for _, t in rows) < per_topic:
            board = chess.Board(None)
            if topic == "poisoned_rook":
                file = rng.randrange(8)
                white_file, black_file = rng.choice([f for f in range(8) if f != file]), rng.choice([f for f in range(8) if f != file])
                board.set_piece_at(file, chess.Piece(chess.QUEEN, True))
                board.set_piece_at(file + 48, chess.Piece(chess.ROOK, False))
                board.set_piece_at(file + 56, chess.Piece(chess.QUEEN, False))
                board.set_piece_at(white_file, chess.Piece(chess.KING, True))
                board.set_piece_at(black_file + 56, chess.Piece(chess.KING, False))
            else:
                file, king_file = rng.randrange(5), rng.randrange(5, 8)
                board.set_piece_at(file, chess.Piece(chess.ROOK, True))
                board.set_piece_at(king_file, chess.Piece(chess.KING, True))
                board.set_piece_at(62, chess.Piece(chess.KING, False))
                for f in (5, 6, 7):
                    board.set_piece_at(f + 8, chess.Piece(chess.PAWN, True))
                    board.set_piece_at(f + 48, chess.Piece(chess.PAWN, False))
            if rng.randrange(2):
                board = board.mirror()
            if not board.is_valid() or board.is_check():
                continue
            key = position_key(observation(board))
            if key not in seen:
                seen.add(key)
                rows.append((board, topic))
    return rows


def evaluate_search(planner, seed=410, budgets=(32, 128), per_topic=4, excluded_positions=()):
    teaching = set(planner.state_model.rules.training_positions) | set(planner.state_model.training_positions)
    teaching |= set(planner.value.summary.get("training_positions", ()))
    ordering = getattr(planner, "ordering_value", planner.value)
    teaching |= set(ordering.summary.get("training_positions", ()))
    if getattr(planner,"policy_prior",None) is not None:
        teaching |= set(planner.policy_prior.summary.get("training_positions",()))
    teaching |= set(excluded_positions)
    generated = tactical_positions(seed, per_topic)
    positions = [(b, t) for b, t in generated if position_key(observation(b)) not in teaching]
    results = []
    for budget in budgets:
        legal = node_ok = exact_sets = pv_ok = mates = traps_avoided = completed_depth = 0
        failures, cases = [], []
        for board, topic in positions:
            state = game_state(board)
            predicted = {a.uci() for a in planner.evaluator.legal_actions(state.board)}
            equal = predicted == {m.uci() for m in board.legal_moves}
            exact_sets += equal
            result = planner.search(state, max_depth=3, max_nodes=budget)
            node_ok += result.nodes <= budget
            completed_depth += result.completed_depth
            move = None if result.action is None else chess.Move.from_uci(result.action.uci())
            accepted = move is not None and board.is_legal(move) and not result.claim_draw
            legal += accepted
            valid_pv = True
            replay = board.copy(stack=True)
            predicted_state = state
            for action in result.principal_variation:
                m = chess.Move.from_uci(action.uci())
                if not replay.is_legal(m):
                    valid_pv = False
                    break
                predicted_state = planner.state_model.predict(predicted_state, action)
                replay.push(m)
                if game_state(replay) != predicted_state:
                    valid_pv = False
                    break
            pv_ok += valid_pv
            success = False
            if accepted:
                after = board.copy()
                after.push(move)
                if topic == "mate_in_one":
                    success = after.is_checkmate()
                    mates += success
                else:
                    target = board.piece_at(move.to_square)
                    success = target is None or target.piece_type != chess.ROOK
                    traps_avoided += success
            if not equal or not accepted or not valid_pv:
                failures.append({"fen": board.fen(), "topic": topic})
            cases.append({"fen": board.fen(), "topic": topic, "tactical_success": success,
                          "action": None if result.action is None else result.action.uci(),
                          **{k: v for k, v in asdict(result).items() if k not in ("action", "principal_variation")}})
        results.append({"node_budget": budget, "positions": len(positions), "legal_choices": legal,
                        "exact_root_legal_sets": exact_sets, "valid_predicted_variations": pv_ok,
                        "within_node_budget": node_ok, "mate_in_one_solved": mates,
                        "poisoned_rook_avoided": traps_avoided,
                        "mean_completed_depth": completed_depth / len(positions) if positions else 0,
                        "correctness_gate_passed": bool(positions) and not failures and node_ok == len(positions),
                        "failures": failures, "cases": cases})
    return {"seed": seed, "generated_positions": len(generated),
            "excluded_teaching_boards": len(generated) - len(positions), "budgets": results}


def evaluate_claim_execution(planner):
    rows = []
    for clock in (99, 100):
        teacher = ChessTeacher(f"3q2k1/3r4/8/8/8/8/8/3Q2K1 w - - {clock} 1")
        state = game_state(teacher.board)
        available, witness = planner.claim_option(state)
        from pai.cognition.contracts import SkillCommand
        command = SkillCommand("claim_draw", arguments=() if witness is None else (witness.uci(),))
        success = available and teacher.execute(command).executed
        rows.append({"case": f"fifty_clock_{clock}", "passed": bool(success and game_state(teacher.board) == state),
                     "witness": None if witness is None else witness.uci()})
    teacher = ChessTeacher()
    for text in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1".split():
        teacher.execute(BoardAction.from_uci(text).command())
    state = game_state(teacher.board)
    available, witness = planner.claim_option(state)
    command = SkillCommand("claim_draw", arguments=() if witness is None else (witness.uci(),))
    success = available and teacher.execute(command).executed
    rows.append({"case": "threefold_before_move", "passed": bool(success and game_state(teacher.board) == state),
                 "witness": None if witness is None else witness.uci()})
    return {"cases": rows, "all_passed": all(r["passed"] for r in rows)}
