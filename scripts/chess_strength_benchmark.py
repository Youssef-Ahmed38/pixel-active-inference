"""Compare NumPy and compiled learned legality on identical public boards."""
import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import chess
import numpy as np
from threadpoolctl import threadpool_limits
import pai.chess.rules as module
from pai.chess.ladder import balanced_openings
from pai.chess.teacher import observation
from pai.chess.provenance import metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True)
    p.add_argument("--seed",type=int,default=84400)
    args = p.parse_args()
    threadpool_limits(limits=1)
    roots = balanced_openings(10,args.seed,depth=8,engine_path=args.engine)
    boards = [observation(chess.Board(fen)) for fen in roots]
    rules = module.LearnedChessRules.load("runs/chess_rules/rules.json")
    compiled = module._compiled
    if compiled is None: raise ValueError("Numba required for compiled benchmark")
    results,sets = {},{}
    try:
        for name,backend in (("numpy_reference",None),("compiled",compiled)):
            module._compiled = backend
            # Explicit warm-up outside timed repetitions.
            sets[name] = [[a.uci() for a in rules.legal_actions(board)] for board in boards]
            elapsed = []
            for board in boards:
                start = time.perf_counter()
                for _ in range(30): rules.legal_actions(board)
                elapsed.append((time.perf_counter()-start)/30)
            results[name] = {"mean_seconds_per_board":float(np.mean(elapsed)),"per_board_seconds":elapsed}
    finally: module._compiled = compiled
    report = {**metadata([args.seed],{"benchmark":"30 warm repetitions per board; one numerical thread"}),
              "roots":roots,"results":results,"exact_legal_set_equivalence":sets["compiled"]==sets["numpy_reference"],
              "speedup":results["numpy_reference"]["mean_seconds_per_board"]/results["compiled"]["mean_seconds_per_board"],
              "limitations":["warm legality microbenchmark only; not whole-search throughput","shared-host load can affect timings"]}
    Path("results/chess_strength_legality_benchmark.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("roots","results")},indent=2))

if __name__ == "__main__": main()
