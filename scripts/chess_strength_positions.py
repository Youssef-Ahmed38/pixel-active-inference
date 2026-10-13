"""Fresh paired value comparison, clustered by teacher game trajectory."""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from pai.chess.distill import observation_from_fen,position_hash,evaluate_rows
from pai.chess.hrm_value import load_value,mirror_observation
from pai.chess.teacher import position_key
from pai.chess.provenance import metadata

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--data",type=Path,default=Path("runs/chess_strength_final/test.jsonl.gz"))
    p.add_argument("--output",type=Path,default=Path("results/chess_strength_final_positions.json"))
    args = p.parse_args()
    candidate,incumbent = load_value(args.candidate),load_value("runs/chess_rules/value.json")
    known = set(candidate.summary.get("training_positions",())) | set(incumbent.summary.get("training_positions",()))
    with gzip.open(args.data,"rt") as stream: rows = [json.loads(line) for line in stream]
    fresh,seen = [],set()
    for row in rows:
        if len(row) < 4: raise ValueError("final data requires teacher-game cluster identities")
        obs = observation_from_fen(row[0])
        orbit = position_hash(row[0])
        if orbit in seen or position_key(obs) in known or position_key(mirror_observation(obs)) in known: continue
        seen.add(orbit);fresh.append(row)
    if not fresh: raise ValueError("no fresh positions remain")
    obs = [observation_from_fen(row[0]) for row in fresh]
    pred = lambda value: np.asarray(value.predict_many(obs) if hasattr(value,"predict_many") else [value.predict(o) for o in obs])*np.asarray([1 if o.turn else -1 for o in obs])
    c,b = np.clip(pred(candidate),-1500,1500),np.clip(pred(incumbent),-1500,1500)
    target = np.asarray([row[1] for row in fresh])
    delta = np.abs(b-target)-np.abs(c-target)
    clusters = {}
    for row,d in zip(fresh,delta): clusters.setdefault(row[3],[]).append(float(d))
    means = np.asarray([np.mean(v) for v in clusters.values()])
    rng = np.random.default_rng(51400)
    boot = means[rng.integers(0,len(means),(20000,len(means)))].mean(1)
    ci = np.quantile(boot,[.025,.975]).tolist()
    positive,negative = sum(means>0),sum(means<0)
    n = int(positive+negative)
    sign_p = sum(math.comb(n,k) for k in range(int(positive),n+1))/2**n if n else 1.
    data_meta = json.loads(args.data.with_name("test.meta.json").read_text())
    report = {**metadata(data_meta["worker_seeds"],{"teacher":{"depth":data_meta["teacher_depth"]}}),
              "generated_positions":len(rows),"fresh_positions":len(fresh),"excluded_overlap_or_duplicates":len(rows)-len(fresh),
              "candidate":evaluate_rows(candidate,fresh),"incumbent":evaluate_rows(incumbent,fresh),
              "teacher_game_clusters":len(means),"cluster_mean_mae_gain_cp":float(means.mean()),
              "cluster_gain_ci95":ci,"paired_cluster_sign_p_one_sided":sign_p,
              "position_gate_passed":len(means)>=10 and ci[0]>0 and sign_p<.05,
              "raw_positions":[{"fen":row[0],"source_game":row[3],"teacher_stm_cp":row[1],
                                "candidate_stm_cp":float(cp),"incumbent_stm_cp":float(bp)} for row,cp,bp in zip(fresh,c,b)],
              "limitations":["game clusters from the same teacher/generator; not independent positions",
                              "clipped centipawn value error does not itself establish match strength"]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="raw_positions"},indent=2))

if __name__ == "__main__": main()
