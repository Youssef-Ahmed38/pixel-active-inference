"""Train learned ending evaluation and an optional Stockfish-distilled value."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai.chess.rules import LearnedChessRules
from pai.chess.state import LearnedChessStateModel
from pai.chess.evaluation import LearnedChessEvaluator, LearnedPositionValue
from pai.chess.evaluation_training import (train_evaluator, evaluate_endings, engine_labels,
                                           fit_value, evaluate_value, evaluate_value_policy)
from pai.chess.rule_training import teaching_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=Path("runs/chess_rules/rules.json"))
    parser.add_argument("--state", type=Path, default=Path("runs/chess_rules/state.json"))
    parser.add_argument("--endings", type=Path, default=Path("runs/chess_rules/endings.json"))
    parser.add_argument("--value", type=Path, default=Path("runs/chess_rules/value.json"))
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=310)
    parser.add_argument("--output", type=Path, default=Path("results/chess_evaluation.json"))
    parser.add_argument("--material-samples", type=int, default=2400)
    parser.add_argument("--eval-material-samples", type=int, default=240)
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--value-only", action="store_true")
    args = parser.parse_args()
    for name in ("material_samples", "eval_material_samples", "nodes"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    state_model = LearnedChessStateModel.load(args.state, LearnedChessRules.load(args.rules))
    report = {"eval_seed": args.eval_seed, "limitations": ["finite evaluation is not all-position proof",
              "standard chess symbolic public state", "FEN omits prior repetition history",
              "value is a small engine-distilled baseline, not measured playing strength",
              "bounded search is available separately; self-play and Elo remain unfinished"]}
    if not args.value_only:
        if args.evaluate_only:
            endings = LearnedChessEvaluator.load(args.endings, state_model)
        else:
            print("Teaching terminal and draw evaluation", flush=True)
            endings = train_evaluator(state_model, args.seed, args.material_samples)
            endings.save(args.endings)
        print("Evaluating fresh ending states", flush=True)
        report["endings_training"] = endings.summary
        report["endings"] = evaluate_endings(endings, args.eval_seed, args.eval_material_samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report["endings"], indent=2), flush=True)
    if args.engine:
        if args.evaluate_only:
            value = LearnedPositionValue.load(args.value)
        else:
            positions = teaching_positions(args.seed + 20, 100, 100, 200, 120)
            positions = [(b, t) for b, t in positions if b.outcome() is None]
            rows, summary = engine_labels(args.engine, positions, args.nodes)
            value = fit_value(rows, summary)
            value.save(args.value)
        positions = teaching_positions(args.eval_seed, 12, 80, 40, 40)
        keys = set(value.summary["training_positions"])
        from pai.chess.teacher import observation, position_key
        positions = [(b, t) for b, t in positions if b.outcome() is None and position_key(observation(b)) not in keys]
        rows, _ = engine_labels(args.engine, positions, args.nodes)
        report["value_training"] = {k: v for k, v in value.summary.items() if k != "training_positions"}
        report["value"] = evaluate_value(value, rows)
        endings = LearnedChessEvaluator.load(args.endings, state_model)
        report["value_policy"] = evaluate_value_policy(value, endings, positions, args.engine, args.nodes)
        print(json.dumps(report["value"], indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
