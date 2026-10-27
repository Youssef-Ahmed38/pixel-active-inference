"""Record the engine identity and measured thinking-time curve from raw games."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import chess.engine
import numpy as np
from pai.chess.provenance import metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True)
    args = p.parse_args()
    with chess.engine.SimpleEngine.popen_uci(str(args.engine)) as engine:
        identity = engine.id
        elo_range = [engine.options["UCI_Elo"].min,engine.options["UCI_Elo"].max]
    data = json.loads(Path("results/chess_strength_final_ladder.json").read_text())
    curve = {}
    for seconds in (.2,1.):
        samples = [s for game in data["raw_games"] if game["white"].get("time",game["black"].get("time"))==seconds for s in game["search_records"]]
        curve[str(seconds)] = {"moves":len(samples),"mean_completed_depth":float(np.mean([s["depth"] for s in samples])),
                              "mean_inspected_nodes":float(np.mean([s["nodes"] for s in samples])),
                              "mean_search_seconds":float(np.mean([s["seconds"] for s in samples])),
                              "max_search_seconds":max(s["seconds"] for s in samples)}
    report = {**metadata([74400],{"both_sides_movetime_seconds":[.2,1.]}),"engine_id":identity,
              "supported_uci_elo":elo_range,"thinking_time_curve":curve,
              "limitations":["soft deadlines checked between model operations; recorded overruns are retained",
                              "curve also changes opponent time budget","confidence intervals overlap; monotonic strength gain is not established"]}
    Path("results/chess_strength_protocol.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
