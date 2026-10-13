"""Teach full public chess transitions and evaluate unseen positions."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai.chess.rule_training import teaching_positions
from pai.chess.rules import LearnedChessRules
from pai.chess.state import LearnedChessStateModel
from pai.chess.state_training import evaluate_rollouts, evaluate_state_model, train_state_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=Path("runs/chess_rules/rules.json"))
    parser.add_argument("--model", type=Path, default=Path("runs/chess_rules/state.json"))
    parser.add_argument("--output", type=Path, default=Path("results/chess_state.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=210)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    rules = LearnedChessRules.load(args.rules)
    if args.evaluate_only:
        model = LearnedChessStateModel.load(args.model, rules)
    else:
        positions = teaching_positions(args.seed, 60, 80, 400, 800, capture_lessons=True)
        print(f"Teaching full state from {len(positions)} positions", flush=True)
        model = train_state_model(rules, positions, args.seed)
        model.save(args.model)
    positions = teaching_positions(args.eval_seed, 12, 80, 100, 120)
    metrics = evaluate_state_model(model, positions, args.eval_seed)
    rollouts = evaluate_rollouts(model, positions, args.eval_seed)
    report = {"scope": "learned full public state transitions and available history",
              "eval_seed": args.eval_seed, "training": model.training_summary, "evaluation": metrics,
              "rollouts": rollouts,
              "limitations": ["finite suites do not prove all-position correctness",
                              "history begins at supplied root; FEN cannot recover earlier repetitions",
                              "draw adjudication, learned value, strategic search and Elo remain unfinished"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(json.dumps(rollouts, indent=2))


if __name__ == "__main__":
    main()
