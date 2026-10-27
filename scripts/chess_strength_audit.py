"""Independently replay raw games and recompute points from adjudication evidence."""
import argparse
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import chess
import chess.pgn
from pai.chess.ladder import candidate_score,pair_statistics
from pai.chess.provenance import metadata

def audit_game(row):
    game = chess.pgn.read_game(io.StringIO(row["pgn"]))
    if game is None or game.errors: raise ValueError("invalid PGN")
    board = game.board()
    if board.fen() != chess.Board(row["root"]).fen(): raise ValueError("root mismatch")
    moves = list(game.mainline_moves())
    for move in moves:
        if not board.is_legal(move): raise ValueError("illegal recorded move")
        board.push(move)
    if len(moves) != row["plies"]: raise ValueError("move count mismatch")
    outcome = board.outcome()
    if outcome is None and row["status"] == "complete":
        command = row["commands"][-1]
        if command["name"] != "claim_draw": raise ValueError("unjustified terminal result")
        witness = command["arguments"]
        if witness:
            move = chess.Move.from_uci(witness[0])
            if not board.is_legal(move): raise ValueError("illegal claim witness")
            board.push(move)
        reason = row["termination"]
        eligible = ((reason == "FIFTY_MOVES" and board.is_fifty_moves()) or
                    (reason == "THREEFOLD_REPETITION" and board.is_repetition(3)))
        if not eligible: raise ValueError("unjustified draw claim")
        result = "1/2-1/2"
    else:
        result = "*" if outcome is None else outcome.result()
    if result != row["result"] or game.headers["Result"] != result:
        raise ValueError("result mismatch")
    return candidate_score(row)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,default=Path("results/chess_strength_raw_audit.json"))
    args = p.parse_args()
    results,seeds = {},set()
    for path in args.games:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        scores = [audit_game(row) for row in rows]
        seeds.update(row["seed"] for row in rows)
        results[str(path)] = {"games":len(rows),"audited":len(scores),
                             "points":sum(score for score in scores if score is not None),
                             "unresolved":sum(score is None for score in scores)}
    report = {**metadata(sorted(seeds),{"controls":"retained per-game white/black configs in audited files"}),
              "files":results,"strict_audit_passed":True}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(results,indent=2))

if __name__ == "__main__": main()
