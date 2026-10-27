"""Measured playing strength: games between the learned agent and limited Stockfish.

The teaching environment owns the rules, executes every command and declares the
result. Agents receive only public observations. Openings are declared roots.
Statistics treat an opening pair (the same root with colours swapped) as the
independent unit, so colour-correlated pairs are not counted as two samples.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

from pai.chess.representation import BoardAction
from pai.chess.state import GameState
from pai.chess.teacher import ChessTeacher
from pai.chess.agent import AgentPlayer

ENGINE = "runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe"


# --------------------------------------------------------------------------- players

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EnginePlayer:
    """Stockfish with limited skill / UCI_Elo and a depth, node or time limit."""

    def __init__(self, config, engine_path=ENGINE):
        engine_path = config.get("engine_path", engine_path)
        self.engine = chess.engine.SimpleEngine.popen_uci(str(Path(engine_path).resolve()))
        options = {"Threads": 1, "Hash": 16}
        if "uci_elo" in config:
            options.update({"UCI_LimitStrength": True, "UCI_Elo": int(config["uci_elo"])})
        if "skill" in config:
            options["Skill Level"] = int(config["skill"])
        self.engine.configure(options)
        self.limit = chess.engine.Limit(depth=config.get("depth"), nodes=config.get("nodes"),
                                        time=config.get("movetime"))
        self.config = dict(config)
        self.name = self.engine.id.get("name")

    def command(self, teacher, rng):
        move = self.engine.play(teacher.board.copy(stack=True), self.limit).move
        return (None if move is None else BoardAction.from_uci(move.uci()).command()), None

    def new_game(self):
        self.engine.configure({"Clear Hash": None})

    def close(self):
        self.engine.quit()


def make_player(config):
    return EnginePlayer(config) if config["kind"] == "engine" else AgentPlayer(config)


# --------------------------------------------------------------------------- openings

def balanced_openings(count, seed, plies=(4, 8), max_cp=80, depth=12, engine_path=ENGINE, excluded_keys=()):
    """Random legal openings kept only if the teacher judges them balanced."""
    rng = random.Random(seed)
    roots,seen = [],set()
    excluded_keys = set(excluded_keys)
    from pai.chess.teacher import observation,position_key
    with chess.engine.SimpleEngine.popen_uci(str(Path(engine_path).resolve())) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        while len(roots) < count:
            board = chess.Board()
            for _ in range(rng.randrange(plies[0], plies[1] + 1)):
                board.push(rng.choice(list(board.legal_moves)))
            if board.is_game_over():
                continue
            fen = board.fen()
            declared = chess.Board(fen)
            if (fen in seen or position_key(observation(declared)) in excluded_keys or
                    position_key(observation(declared.mirror())) in excluded_keys): continue
            score = engine.analyse(board, chess.engine.Limit(depth=depth))["score"].white().score(mate_score=100000)
            if abs(score) <= max_cp:
                roots.append(fen)
                seen.add(fen)
    return roots


# --------------------------------------------------------------------------- games

def play_game(white, black, root_fen, max_plies=400, seed=0):
    teacher = ChessTeacher(root_fen, episode=seed)
    rng = random.Random(seed)
    for p in (white, black):
        if hasattr(p, "prepare"):
            p.prepare(ChessTeacher().observe())
        if hasattr(p, "new_game"):
            p.new_game()
    failure, agent_stats = None, []
    for ply in range(max_plies):
        if teacher.game_outcome() is not None:
            break
        player = white if teacher.board.turn else black
        start = time.perf_counter()
        if hasattr(player,"command_from_belief"):
            command,stats = player.command_from_belief(teacher.observe())
        else:
            command, stats = player.command(teacher, rng)
        measured = time.perf_counter() - start
        if stats is not None:
            agent_stats.append({**stats, "side": "white" if teacher.board.turn else "black", "wall_seconds": measured})
        if command is None or not teacher.execute(command).executed:
            failure = {"ply": ply, "side": "white" if teacher.board.turn else "black",
                       "fen": teacher.board.fen(), "command": None if command is None else asdict(command)}
            break
    outcome = teacher.game_outcome()
    game = chess.pgn.Game()
    game.setup(chess.Board(root_fen))
    node = game
    for move in teacher.board.move_stack:
        node = node.add_variation(move)
    result = "*" if outcome is None else outcome.result()
    game.headers["Result"] = result
    status = "execution_failure" if failure else ("complete" if outcome is not None else "truncated")
    return {"status": status, "result": result, "termination": None if outcome is None else outcome.termination.name,
            "plies": len(teacher.board.move_stack), "failure": failure, "pgn": str(game),
            "search_records": agent_stats,
            "commands": [asdict(c) for c in teacher.record.commands],
            "agent_depth_mean": float(np.mean([s["depth"] for s in agent_stats])) if agent_stats else None,
            "agent_seconds_mean": float(np.mean([s["seconds"] for s in agent_stats])) if agent_stats else None}


_PLAYERS = {}


def _player(config):
    key = json.dumps(config, sort_keys=True)
    if key not in _PLAYERS:
        _PLAYERS[key] = make_player(config)
    return _PLAYERS[key]


def run_job(job):
    """Worker entry: one game. job has id, white, black, root, max_plies, seed."""
    start = time.time()
    result = play_game(_player(job["white"]), _player(job["black"]), job["root"], job["max_plies"], job["seed"])
    result.update({k: job[k] for k in ("id", "group", "pair", "candidate_color", "root")})
    result.update({k: job[k] for k in ("seed", "white", "black", "provenance") if k in job})
    result["wall_seconds"] = time.time() - start
    return result


def candidate_score(game):
    """Score of the named candidate side (agent or challenger), None if unresolved."""
    if game["status"] != "complete":
        return None
    if game["result"] not in ("1-0", "0-1", "1/2-1/2") or game["candidate_color"] not in ("white", "black"):
        raise ValueError("valid completed result and candidate colour required")
    if game["result"] == "1/2-1/2":
        return 0.5
    white_won = game["result"] == "1-0"
    return float(white_won == (game["candidate_color"] == "white"))


def run_schedule(jobs, output, workers=6, progress=print):
    """Run jobs in a process pool; append each finished game to a JSONL file."""
    from multiprocessing import Pool
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output.exists():
        done = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [j for j in jobs if j["id"] not in done]
    progress(f"{len(done)} saved games, {len(pending)} to play", flush=True)
    if not pending:
        return
    with Pool(min(workers, len(pending))) as pool, open(output, "a", encoding="utf-8") as handle:
        for i, game in enumerate(pool.imap_unordered(run_job, pending)):
            handle.write(json.dumps(game) + "\n")
            handle.flush()
            progress(f"[{i + 1}/{len(pending)}] {game['group']} pair {game['pair']} cand={game['candidate_color']}: "
                     f"{game['status']} {game['result']} ({game['plies']} plies, {game['wall_seconds']:.0f}s)", flush=True)


def load_games(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- statistics

def elo_from_score(p):
    return None if p <= 0 or p >= 1 else -400.0 * math.log10(1.0 / p - 1.0)


def pair_statistics(games, bootstrap=20000, seed=0):
    """Score with a pair-cluster bootstrap interval and its Elo-difference mapping."""
    if bootstrap < 1:
        raise ValueError("positive bootstrap count required")
    by_pair = {}
    slots = set()
    unresolved = 0
    for g in games:
        slot = (g["pair"], g["candidate_color"])
        if slot in slots:
            raise ValueError("duplicate pair/colour game")
        slots.add(slot)
        s = candidate_score(g)
        if s is None:
            unresolved += 1
            continue
        by_pair.setdefault(g["pair"], []).append(s)
    pairs = [v for v in by_pair.values()]
    scores = [s for v in pairs for s in v]
    if not scores:
        return {"games": 0, "unresolved_games": unresolved}
    n = len(scores)
    rate = float(np.mean(scores))
    rng = np.random.default_rng(seed)
    sums = np.array([sum(v) for v in pairs])
    counts = np.array([len(v) for v in pairs])
    idx = rng.integers(0, len(pairs), size=(bootstrap, len(pairs)))
    boot = sums[idx].sum(1) / counts[idx].sum(1)
    low, high = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    radius = math.sqrt(math.log(40)/(2*len(pairs)))
    conservative = [max(0.,rate-radius),min(1.,rate+radius)]
    degenerate = bool(np.all(boot == boot[0]))
    reported_ci = conservative if degenerate else [low,high]
    # Wilson interval on the per-game score (independence assumption) for reference.
    z = 1.959964
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    complete_pairs = all(len(v) == 2 for v in pairs) and unresolved == 0
    advantages = [np.mean(v) - .5 for v in pairs if len(v) == 2]
    positive, negative = int(sum(a > 0 for a in advantages)), int(sum(a < 0 for a in advantages))
    informative = positive + negative
    sign_p = sum(math.comb(informative, k) for k in range(positive, informative + 1)) / 2 ** informative if informative else 1.0
    return {"games": n, "pairs": len(pairs), "unresolved_games": unresolved,
            "wins": scores.count(1.0), "draws": scores.count(0.5), "losses": scores.count(0.0),
            "score": rate, "score_ci95_pair_bootstrap": [low, high],
            "score_ci95_wilson_games": [centre - half, centre + half],
            "elo_difference": elo_from_score(rate),
            "elo_difference_ci95": [elo_from_score(v) for v in reported_ci],
            "score_ci95_reported":reported_ci,"score_ci95_hoeffding_pairs":conservative,
            "bootstrap_degenerate":degenerate,
            "lower_bound_above_half": low > 0.5, "complete_colour_pairs": complete_pairs,
            "paired_sign_test_p_one_sided": sign_p,
            "promotion_gate_passed": bool(complete_pairs and len(pairs) >= 10 and reported_ci[0] > .5 and sign_p < .05),
            "statistical_assumptions": "independent opening clusters; sign symmetry under equal-strength null"}


def anchored_elo(groups, bootstrap=5000, seed=0):
    """Maximum-likelihood agent rating from rungs with a configured UCI_Elo anchor.

    Uses the logistic Elo model with draws as half points; a pair-cluster
    bootstrap gives the interval. Valid only relative to the configured engine
    ratings under *these* time controls, which Stockfish does not calibrate.
    """
    rows = []  # (anchor, pair_scores list)
    for anchor, games in groups:
        by_pair = {}
        for g in games:
            s = candidate_score(g)
            if s is not None:
                by_pair.setdefault(g.get("root",g["pair"]), []).append(s)
        rows.append((anchor, by_pair))
    if not rows or not any(p for _, p in rows):
        return None

    def fit(sample):
        anchors = np.array([a for a, s in sample for _ in s], dtype=float)
        scores = np.array([x for _, s in sample for x in s], dtype=float)
        if len(scores) == 0:
            return None
        if scores.sum() == 0 or scores.sum() == len(scores):
            return -math.inf if scores.sum() == 0 else math.inf
        # The likelihood is maximised where expected total score equals the
        # observed total; expected score increases monotonically with rating.
        lo, hi = -2000.0, 6000.0
        for _ in range(60):
            mid = (lo + hi) / 2
            expected = (1 / (1 + 10 ** ((anchors - mid) / 400))).sum()
            lo, hi = (mid, hi) if expected < scores.sum() else (lo, mid)
        return (lo + hi) / 2

    flat = lambda r: [(a, [x for pair in pairs.values() for x in pair]) for a, pairs in r]
    point = fit(flat(rows))
    if point is None or not math.isfinite(point):
        return {"estimate": None, "reason": "all anchored scores are zero or all are one"}
    rng = np.random.default_rng(seed)
    cluster_keys = list(dict.fromkeys(key for _,pairs in rows for key in pairs))
    boots = []
    for _ in range(bootstrap):
        # The same opening appears at several levels. Resample it once across
        # all levels, retaining those cross-level dependencies.
        picked = [cluster_keys[i] for i in rng.integers(0,len(cluster_keys),len(cluster_keys))]
        sample = []
        for a, pairs in rows:
            if pairs:
                sample.append((a, [x for key in picked for x in pairs.get(key,())]))
        b = fit(sample)
        if b is not None:
            boots.append(b)
    low, high = np.percentile(np.array(boots), [2.5, 97.5], method="nearest")
    finite = lambda v: float(v) if math.isfinite(v) else None  # None: unbounded at that end
    return {"estimate": point, "ci95": [finite(low), finite(high)],
            "bootstrap_degenerate_fraction": float(np.mean(~np.isfinite(boots))),
            "bootstrap_draws": bootstrap,
            "opening_clusters":len(cluster_keys),
            "note": "relative to configured UCI_Elo under these short limits; not a calibrated human rating"}
