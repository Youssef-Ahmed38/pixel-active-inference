"""Collect agent-only self-play first, then have the teacher label its positions."""
import argparse
import gzip
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chess.engine
import numpy as np
from pai.chess.ladder import AgentPlayer
from pai.chess.teacher import ChessTeacher
from pai.chess.play import opening
from pai.chess.provenance import metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--value", type=Path, default=Path("runs/chess_strength_data/value.npz"))
    p.add_argument("--directory", type=Path, default=Path("runs/chess_strength_data"))
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--plies", type=int, default=64)
    p.add_argument("--seed", type=int, default=36400)
    args = p.parse_args()
    player = AgentPlayer({"value": str(args.value), "time": .03, "nodes": 128, "reuse_pv": True})
    games, positions = [], []
    for index in range(args.games):
        teacher = ChessTeacher(opening(args.seed + index).fen(), episode=args.seed + index)
        player.prepare(teacher.observe())
        moves, failure = [], None
        for _ in range(args.plies):
            if teacher.game_outcome() is not None: break
            positions.append(teacher.board.fen())
            command, stats = player.command_from_belief(teacher.observe())
            if command is None or not teacher.execute(command).executed:
                failure = "rejected agent command"
                break
            moves.append({"command": command.arguments, "search": stats})
        games.append({"seed": args.seed + index, "failure": failure, "moves": moves,
                      "result": None if teacher.game_outcome() is None else teacher.game_outcome().result()})
        print(f"Collected self-play {index + 1}/{args.games}", flush=True)
    # Engine connection is opened only after all agent decisions finish.
    rows = []
    with chess.engine.SimpleEngine.popen_uci(str(args.engine.resolve())) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        for fen in positions:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=8))
            rows.append((fen, float(np.clip(info["score"].pov(board.turn).score(mate_score=100000), -1500, 1500)),
                         info["pv"][0].uci()))
    with gzip.open(args.directory / "train.jsonl.gz", "rt") as stream:
        teacher_rows = [json.loads(line) for line in stream]
    with gzip.open(args.directory / "train_mixed.jsonl.gz", "wt") as stream:
        for row in teacher_rows + rows: stream.write(json.dumps(row) + "\n")
    report = {**metadata(range(args.seed, args.seed + args.games), {"agent": {"movetime": .03, "nodes": 128}, "teacher_after_play": {"depth": 8}}),
              "games": games, "selfplay_labels": len(rows), "teacher_game_labels": len(teacher_rows),
              "limitations": ["truncated self-play supplies positions, never invented terminal targets", "engine labels collected after agent play"]}
    Path("results/chess_strength_selfplay_data.json").write_text(json.dumps(report, indent=2) + "\n")

if __name__ == "__main__": main()
