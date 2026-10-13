"""Teacher-owned game execution, self-play collection and match measurement."""
from dataclasses import asdict
import json
import io
import copy
import math
from pathlib import Path
import random

import chess
import chess.engine
import chess.pgn
import numpy as np

from pai.chess.evaluation import LearnedPositionValue, value_features
from pai.chess.search import LearnedChessSearch
from pai.chess.state import GameState
from pai.chess.representation import BoardAction
from pai.chess.teacher import ChessTeacher, observation, position_key


def opening(seed, plies=6):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def play_game(planner, root, *, agent_color=None, engine=None, engine_limit=None,
              nodes=64, max_plies=96, exploration=0.0, seed=0, preserve_history=False):
    if nodes < 1 or max_plies < 1 or not 0 <= exploration <= 1:
        raise ValueError("invalid game budget or exploration probability")
    if agent_color is not None and engine is None:
        raise ValueError("an opponent engine is required")
    rng = random.Random(seed)
    # A supplied opening is the declared root. Pre-root history is not invented.
    teacher = (ChessTeacher.from_board(root, seed) if preserve_history else
               ChessTeacher(root.fen(en_passant="fen"), episode=seed))
    initial = teacher.board.root()
    resumed_plies = len(teacher.board.move_stack)
    samples, searches, commands, failure = [], [], [], None
    for ply in range(max_plies):
        if teacher.game_outcome() is not None:
            break
        state = GameState.from_belief(teacher.observe())
        is_agent = agent_color is None or teacher.board.turn == agent_color
        if is_agent:
            result = planner.search(state, max_depth=3, max_nodes=nodes)
            command = result.command()
            if command is not None and not result.claim_draw and rng.random() < exploration:
                choices = planner.evaluator.legal_actions(state.board)
                command = rng.choice(choices).command()
            score = result.score_cp
            if score is not None:
                samples.append({"pieces": state.board.pieces, "turn": state.board.turn,
                                "castling": state.board.castling, "ep_square": state.board.ep_square,
                                "target_cp": float(np.clip(score * (1 if state.board.turn else -1), -2000, 2000)),
                                "completed_depth": result.completed_depth, "ply": ply + resumed_plies})
            searches.append({k: v for k, v in asdict(result).items() if k not in ("action", "principal_variation")})
        else:
            move = engine.play(teacher.board.copy(stack=True), engine_limit, game=seed).move
            command = None if move is None else BoardAction.from_uci(move.uci()).command()
        if command is None or not teacher.execute(command).executed:
            failure = {"ply": ply, "actor": "agent" if is_agent else "engine",
                       "fen": teacher.board.fen(en_passant="fen"),
                       "command": None if command is None else asdict(command)}
            break
        commands.append(asdict(command))
    outcome = teacher.game_outcome()
    result_text = "*" if outcome is None else outcome.result()
    status = "execution_failure" if failure else ("complete" if outcome is not None else "truncated")
    game = chess.pgn.Game()
    game.setup(initial)
    game.headers["Result"] = result_text
    game.headers["Termination"] = status if outcome is None else outcome.termination.name
    node = game
    for move in teacher.board.move_stack:
        node = node.add_variation(move)
    if commands and commands[-1]["name"] == "claim_draw":
        node.comment = "Draw claimed" + (" before " + commands[-1]["arguments"][0] if commands[-1]["arguments"] else "")
    score = None
    if agent_color is not None and outcome is not None:
        score = .5 if outcome.winner is None else float(outcome.winner == agent_color)
    return {"status": status, "result": result_text, "agent_score": score,
            "agent_color": None if agent_color is None else ("white" if agent_color else "black"),
            "termination": None if outcome is None else outcome.termination.name,
            "moves": len(teacher.board.move_stack), "resumed_plies": resumed_plies,
            "commands": commands, "failure": failure,
            "samples": samples, "searches": searches, "pgn": str(game)}, teacher.record


def collect_selfplay(planner, games=4, nodes=64, max_plies=48, seed=500, progress=print):
    records, replay = [], []
    for i in range(games):
        game, _ = play_game(planner, opening(seed + i), nodes=nodes, max_plies=max_plies,
                            exploration=.15, seed=seed + i)
        for row in game["samples"]:
            row["game_id"] = i
            replay.append(row)
        records.append(game)
        progress(f"Self-play {i + 1}/{games}: {game['status']}, {game['moves']} moves, {len(game['samples'])} value targets", flush=True)
    return records, replay


def fit_selfplay_value(baseline, replay, prior_weight=100.0, step=.25):
    """Regularized fitted value iteration from deeper self-play search targets.

    No targets are fabricated for truncated game outcomes. Search targets are
    bootstrapped estimates and do not demonstrate an improvement by themselves.
    """
    if not replay or prior_weight <= 0 or not 0 < step <= 1:
        raise ValueError("nonempty replay and valid update parameters required")
    from pai.chess.representation import BoardObservation
    observations = [BoardObservation(tuple(r["pieces"]), r["turn"], tuple(r["castling"]), r["ep_square"]) for r in replay]
    x = np.array([value_features(o) for o in observations], dtype=np.float64)
    y = np.array([r["target_cp"] for r in replay], dtype=np.float64)
    weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]) * prior_weight,
                              x.T @ y + prior_weight * baseline.weights)
    updated = (1 - step) * baseline.weights + step * weights
    summary = {**baseline.summary, "update": "regularized self-play fitted value iteration",
               "selfplay_examples": len(replay), "prior_weight": prior_weight, "update_fraction": step,
               "training_positions": sorted(set(baseline.summary.get("training_positions", ())) |
                                             {position_key(o) for o in observations})}
    candidate = LearnedPositionValue(updated, summary)
    return candidate, {"examples": len(replay), "baseline_target_mae_cp": float(np.mean(np.abs(x @ baseline.weights - y))),
                       "candidate_target_mae_cp": float(np.mean(np.abs(x @ candidate.weights - y))),
                       "weight_change_l2": float(np.linalg.norm(candidate.weights - baseline.weights))}


def performance(scores):
    """Conditional relative performance, with a conservative bounded-score interval."""
    scores = [float(s) for s in scores if s is not None]
    if any(s not in (0.0, .5, 1.0) for s in scores):
        raise ValueError("completed game scores must be losses, draws or wins")
    if not scores:
        return {"completed_games": 0, "score_rate": None, "relative_elo": None, "score_interval_95": None}
    n, rate = len(scores), float(np.mean(scores))
    radius = math.sqrt(math.log(40) / (2 * n))
    low, high = max(0.0, rate - radius), min(1.0, rate + radius)
    def elo(p):
        return None if p <= 0 or p >= 1 else 400 * math.log10(p / (1 - p))
    return {"completed_games": n, "wins": scores.count(1.0), "draws": scores.count(.5), "losses": scores.count(0.0),
            "score_rate": rate, "relative_elo": elo(rate),
            "smoothed_relative_elo": elo((sum(scores) + .5) / (n + 1)),
            "score_interval_95": [low, high], "relative_elo_interval_95": [elo(low), elo(high)],
            "rating_interpretation": "relative to this opponent/configuration; not a human Elo rating",
            "interval_assumption": "bounded independent game scores; paired openings may be correlated"}


def stockfish_matches(planner, engine_path, levels=(1320, 1800, 2400), budgets=(32, 128),
                      pairs=1, engine_nodes=1000, max_plies=96, seed=600, output_dir=None, progress=print,
                      on_progress=None, existing_reports=None):
    reports = copy.deepcopy(existing_reports or [])
    group_keys = [(r["configured_uci_elo"], r["agent_nodes"]) for r in reports]
    if len(group_keys) != len(set(group_keys)):
        raise ValueError("duplicate saved match groups")
    expected_groups = {(level, budget) for level in levels for budget in budgets}
    if any(key not in expected_groups for key in group_keys):
        raise ValueError("saved match groups are outside the requested schedule")
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True})
        option = engine.options["UCI_Elo"]
        if any(level < option.min or level > option.max for level in levels):
            raise ValueError("requested engine rating outside supported range")
        for level in levels:
            engine.configure({"UCI_Elo": level})
            for budget in budgets:
                report = next((r for r in reports if (r["configured_uci_elo"], r["agent_nodes"]) == (level, budget)), None)
                if report is None:
                    report = {"engine": engine.id.get("name"), "configured_uci_elo": level, "agent_nodes": budget,
                              "opponent_nodes": engine_nodes, "scheduled_games": pairs * 2,
                              "schedule_complete": False, "games": []}
                    reports.append(report)
                elif (report["engine"] != engine.id.get("name") or report["opponent_nodes"] != engine_nodes
                      or report.get("scheduled_games", pairs * 2) != pairs * 2):
                    raise ValueError("saved match configuration does not match schedule")
                games = report["games"]
                completed_slots = {(g["pair_id"], g["agent_color"]) for g in games}
                if (len(completed_slots) != len(games) or any(pair not in range(pairs) or color not in ("white", "black")
                                                           for pair, color in completed_slots)):
                    raise ValueError("invalid or duplicate saved game slots")
                for pair in range(pairs):
                    root = opening(seed + pair)
                    for color in (True, False):
                        if (pair, "white" if color else "black") in completed_slots:
                            continue
                        engine.configure({"Clear Hash": None})
                        game, _ = play_game(planner, root, agent_color=color, engine=engine,
                                             engine_limit=chess.engine.Limit(nodes=engine_nodes), nodes=budget,
                                             max_plies=max_plies, seed=seed + pair)
                        game["pair_id"] = pair
                        games.append(game)
                        report.update(statistics=performance([g["agent_score"] for g in games]),
                                      truncated_games=sum(g["status"] == "truncated" for g in games),
                                      execution_failures=[g["failure"] for g in games if g["failure"]],
                                      schedule_complete=len(games) == pairs * 2)
                        progress(f"Stockfish {level}, agent {budget} nodes, pair {pair + 1}, {game['agent_color']}: {game['status']} {game['result']} ({game['moves']} moves)", flush=True)
                        if output_dir:
                            path = Path(output_dir)
                            path.mkdir(parents=True, exist_ok=True)
                            (path / f"sf{level}_n{budget}_p{pair}_{game['agent_color']}.pgn").write_text(game["pgn"] + "\n", encoding="utf-8")
                            temporary = path / "matches.partial.json.tmp"
                            temporary.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
                            temporary.replace(path / "matches.partial.json")
                        if on_progress is not None:
                            on_progress(reports)
    return reports


def resume_matches(planner, reports, engine_path, max_plies=256, seed=600, on_progress=None):
    """Continue unfinished PGNs only; retain their root, colors and full history."""
    reports = copy.deepcopy(reports)
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True})
        for report in reports:
            if report.get("engine") != engine.id.get("name"):
                raise ValueError("resume requires the same opponent engine identity")
            engine.configure({"UCI_Elo": report["configured_uci_elo"]})
            for index, previous in enumerate(report["games"]):
                if previous["status"] != "truncated":
                    continue
                game = chess.pgn.read_game(io.StringIO(previous["pgn"]))
                if game is None or game.errors:
                    raise ValueError("cannot resume a malformed PGN")
                board = game.end().board()
                if len(board.move_stack) != previous["moves"]:
                    raise ValueError("PGN does not match saved move count")
                engine.configure({"Clear Hash": None})
                resumed, _ = play_game(planner, board, agent_color=previous["agent_color"] == "white",
                    engine=engine, engine_limit=chess.engine.Limit(nodes=report["opponent_nodes"]),
                    nodes=report["agent_nodes"], max_plies=max_plies,
                    seed=seed + previous["pair_id"], preserve_history=True)
                resumed["pair_id"] = previous["pair_id"]
                for field in ("commands", "samples", "searches"):
                    resumed[field] = previous[field] + resumed[field]
                resumed["resume_segments"] = previous.get("resume_segments", []) + [
                    {"start_plies": previous["moves"], "additional_budget": max_plies}]
                report["games"][index] = resumed
                report["statistics"] = performance([g["agent_score"] for g in report["games"]])
                report["truncated_games"] = sum(g["status"] == "truncated" for g in report["games"])
                report["execution_failures"] = [g["failure"] for g in report["games"] if g["failure"]]
                print(f"Resumed SF {report['configured_uci_elo']}, {report['agent_nodes']} nodes, "
                      f"{resumed['agent_color']}: {resumed['status']} {resumed['result']} ({resumed['moves']} moves)", flush=True)
                if on_progress is not None:
                    on_progress(reports)
    return reports
