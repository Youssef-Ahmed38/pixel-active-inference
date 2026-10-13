"""Fit a fast learned linear value on the new mixed teacher/self-play corpus."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from pai.chess.distill import evaluate_rows, observation_from_fen, position_hash
from pai.chess.evaluation import LearnedPositionValue, value_features
from pai.chess.provenance import metadata
from pai.chess.teacher import position_key
from pai.chess.hrm_value import mirror_observation

def read(path):
    with gzip.open(path, "rt") as stream: return [json.loads(line) for line in stream]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("runs/chess_strength_data"))
    args = p.parse_args()
    training = read(args.directory / "train_mixed.jsonl.gz")
    validation = read(args.directory / "val.jsonl.gz")
    held = {position_hash(row[0]) for row in validation}
    training = [row for row in training if position_hash(row[0]) not in held]
    obs = [observation_from_fen(row[0]) for row in training]
    x = np.asarray([value_features(o) for o in obs], dtype=float)
    y = np.asarray([row[1] * (1 if o.turn else -1) for row, o in zip(training, obs)])
    choices = []
    for ridge in (1., 10., 100., 1000.):
        candidate = LearnedPositionValue(np.linalg.solve(x.T @ x + ridge * np.eye(391), x.T @ y))
        metric = evaluate_rows(candidate, validation)
        choices.append((metric["mae_cp"], candidate, ridge, metric))
    _, candidate, ridge, metric = min(choices, key=lambda row: row[0])
    keys = set()
    for row in training + validation:
        o = observation_from_fen(row[0])
        keys.update((position_key(o), position_key(mirror_observation(o))))
    candidate.summary = {"training_positions": sorted(keys), "ridge": ridge, "mixed_teaching_rows": len(training),
                         "teacher_depth": 8, "seed": 0}
    candidate.save(args.directory / "linear_candidate.json")
    report = {**metadata([10400, 10401, 10402, 10403, 20400, 20401, *range(36400,36408)], {"teacher": {"depth": 8}}),
              "candidate": metric, "ridge": ridge,
              "incumbent": evaluate_rows(LearnedPositionValue.load("runs/chess_rules/value.json"), validation)}
    Path("results/chess_strength_linear_development.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
