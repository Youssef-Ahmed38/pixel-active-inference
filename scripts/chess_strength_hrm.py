"""Try the shared HRM on the new distillation corpus; keep it as a candidate."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from pai.chess.distill import evaluate_rows, observation_from_fen
from pai.chess.evaluation import LearnedPositionValue
from pai.chess.hrm_training import fit_hrm_value
from pai.chess.provenance import metadata

def read(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("runs/chess_strength_data"))
    p.add_argument("--limit", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=41400)
    p.add_argument("--output", type=Path, default=Path("results/chess_strength_hrm_development.json"))
    args = p.parse_args()
    torch.set_num_threads(2)
    baseline = LearnedPositionValue.load("runs/chess_rules/value.json")
    rows = read(args.directory / "train.jsonl.gz")
    rng = np.random.default_rng(args.seed)
    selected = [rows[i] for i in rng.choice(len(rows), min(args.limit, len(rows)), replace=False)]
    teaching = []
    for row in selected:
        fen, score, _ = row[:3]
        obs = observation_from_fen(fen)
        teaching.append((obs, score * (1 if obs.turn else -1), "teacher_game"))
    candidate, history = fit_hrm_value(baseline, teaching, args.epochs, args.seed)
    candidate.save(args.directory / "hrm_candidate.pt")
    validation = read(args.directory / "val.jsonl.gz")
    report = {**metadata([args.seed, 10400, 10401, 10402, 10403, 20400, 20401], {"teacher": {"depth": 8}}),
              "training_positions": len(teaching), "history": history,
              "candidate": evaluate_rows(candidate, validation), "incumbent": evaluate_rows(baseline, validation),
              "limitations": ["shared HRM value only; no learned chess halting", "fixed two thinking segments",
                              "development corpus; no final strength conclusion"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))

if __name__ == "__main__": main()
