"""Train the chess HRM on CPU and validate on fresh teacher-labeled boards."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch

from pai.chess.evaluation import LearnedPositionValue
from pai.chess.evaluation_training import engine_labels
from pai.chess.hrm_training import fit_hrm_value, prediction_metrics, orbit_key
from pai.chess.rule_training import teaching_positions
from pai.chess.teacher import observation, position_key
from pai.chess.hrm_value import mirror_observation


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--seed", type=int, default=700)
    p.add_argument("--eval-seed", type=int, default=701)
    p.add_argument("--train-limit", type=int, default=1200)
    p.add_argument("--eval-limit", type=int, default=240)
    p.add_argument("--nodes", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--directory", type=Path, default=ROOT / "runs/chess_hrm")
    p.add_argument("--output", type=Path, default=ROOT / "results/chess_hrm.json")
    args = p.parse_args()
    if args.seed == args.eval_seed or min(args.train_limit, args.eval_limit, args.nodes, args.epochs) < 1:
        p.error("positive budgets and different training/evaluation seeds required")
    torch.set_num_threads(2)
    args.directory.mkdir(parents=True, exist_ok=True)
    baseline = LearnedPositionValue.load(ROOT / "runs/chess_rules/value.json")
    # Entire generated game trajectories are split between the teaching and final seed sets.
    training = [(b, t) for b, t in teaching_positions(args.seed, 35, 90, 80, 60) if b.outcome() is None][:args.train_limit]
    rows, teacher = engine_labels(args.engine, training, args.nodes)
    args.directory.joinpath("teaching.json").write_text(json.dumps({"seed": args.seed, "teacher": teacher,
        "rows": [{"board": asdict(o), "score": y, "topic": t} for o, y, t in rows]}, indent=2) + "\n", encoding="utf-8")
    candidate, history = fit_hrm_value(baseline, rows, args.epochs, args.seed)
    candidate.save(args.directory / "candidate.pt")
    excluded = {orbit_key(o) for o, _, _ in rows}
    known = set(candidate.summary["training_positions"])
    positions = [(b, t) for b, t in teaching_positions(args.eval_seed, 15, 90, 60, 40)
                 if b.outcome() is None and orbit_key(observation(b)) not in excluded and
                 position_key(observation(b)) not in known and position_key(mirror_observation(observation(b))) not in known][:args.eval_limit]
    final, _ = engine_labels(args.engine, positions, args.nodes)
    base_metrics = prediction_metrics(baseline, final)
    segment_metrics = {str(segments): prediction_metrics(candidate, final, segments) for segments in (1, 2)}
    report = {"seed": args.seed, "eval_seed": args.eval_seed, "teacher": teacher,
              "training": {k: v for k, v in candidate.summary.items() if k != "training_positions"},
              "history": history, "baseline": base_metrics, "hrm_by_segments": segment_metrics,
              "candidate_accepted_for_trial": segment_metrics["2"]["mae_cp"] < base_metrics["mae_cp"],
              "limitations": ["finite symbolic chess value distillation; not self-play improvement",
                  "fixed computation; no trained chess halting", "value error is not match strength"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
