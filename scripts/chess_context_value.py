"""Fit contextual attack evidence on teaching data and evaluate a fresh seed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.linear_model import Ridge

from pai.chess.evaluation import LearnedPositionValue
from pai.chess.evaluation_training import engine_labels
from pai.chess.hrm_training import partition, prediction_metrics
from pai.chess.hrm_value import mirror_observation
from pai.chess.representation import BoardObservation
from pai.chess.rules import LearnedChessRules
from pai.chess.rule_training import teaching_positions
from pai.chess.teacher import position_key, observation
from pai.chess.threat_value import ContextPositionValue, context_features


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--teaching", type=Path, default=ROOT / "runs/chess_hrm/teaching.json")
    p.add_argument("--eval-seed", type=int, default=703)
    p.add_argument("--output", type=Path, default=ROOT / "results/chess_context_value.json")
    args = p.parse_args()
    data = json.loads(args.teaching.read_text())
    rows = []
    for row in data["rows"]:
        b = row["board"]
        rows.append((BoardObservation(tuple(b["pieces"]), b["turn"], tuple(b["castling"]), b["ep_square"]), row["score"], row["topic"]))
    baseline = LearnedPositionValue.load(ROOT / "runs/chess_rules/value.json")
    rules = LearnedChessRules.load(ROOT / "runs/chess_rules/rules.json")
    train, validation = partition(rows)
    x = np.array([context_features(o, rules.attack_table) for o, _, _ in train])
    residual = np.array([y - baseline.predict(o) for o, y, _ in train])
    choices = []
    for alpha in (1, 10, 100, 1000):
        fitted = Ridge(alpha=alpha, fit_intercept=False).fit(x, residual)
        candidate = ContextPositionValue(baseline, rules, fitted.coef_)
        choices.append((prediction_metrics(candidate, validation)["mae_cp"], alpha, candidate))
    metric, alpha, candidate = min(choices, key=lambda row: row[0])
    keys = set(baseline.summary["training_positions"])
    keys.update(position_key(o) for o, _, _ in rows)
    keys.update(position_key(mirror_observation(o)) for o, _, _ in rows)
    candidate.summary = {"training_positions": sorted(keys), "features": 24,
                         "training_examples": len(train), "validation_examples": len(validation),
                         "selected_alpha": alpha, "validation_mae_cp": metric}
    candidate.save(ROOT / "runs/chess_context/candidate.json")
    positions = [(b,t) for b,t in teaching_positions(args.eval_seed, 15, 90, 60, 40) if b.outcome() is None
                 and position_key(observation(b)) not in keys and position_key(mirror_observation(observation(b))) not in keys][:240]
    final, _ = engine_labels(args.engine, positions, data["teacher"]["nodes_per_position"])
    base, updated = prediction_metrics(baseline, final), prediction_metrics(candidate, final)
    report = {"eval_seed": args.eval_seed, "teaching_seed": data["seed"],
              "validation_choices": [{"alpha": a, "mae_cp": m} for m,a,_ in choices],
              "baseline": base, "candidate": updated,
              "candidate_accepted_for_trial": updated["mae_cp"] < base["mae_cp"],
              "limitations": ["24 learned attack-context weights; no match-strength conclusion",
                              "internal validation shares generated game trajectories", "final seed is separate"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
