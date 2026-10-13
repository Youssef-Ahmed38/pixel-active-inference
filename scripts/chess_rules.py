"""Induce relational chess rules and evaluate fresh full-action-space suites."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pai.chess.rules import LearnedChessRules
from pai.chess.rule_training import teaching_positions, train_rules
from pai.chess.rule_evaluation import evaluate_rules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=108)
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--plies", type=int, default=80)
    parser.add_argument("--sparse", type=int, default=400)
    parser.add_argument("--special", type=int, default=200)
    parser.add_argument("--corrections", type=int, default=12)
    parser.add_argument("--eval-games", type=int, default=12)
    parser.add_argument("--eval-sparse", type=int, default=80)
    parser.add_argument("--eval-special", type=int, default=80)
    parser.add_argument("--model", type=Path, default=Path("runs/chess_rules/rules.json"))
    parser.add_argument("--output", type=Path, default=Path("results/chess_rules.json"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    for name in ("games", "plies", "sparse", "special", "eval_games", "eval_sparse", "eval_special"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.corrections < 0:
        parser.error("--corrections cannot be negative")
    if args.evaluate_only:
        model = LearnedChessRules.load(args.model)
    else:
        train = teaching_positions(args.seed, args.games, args.plies, args.sparse, args.special, capture_lessons=True)
        print(f"Teaching {len(train)} positions", flush=True)
        model = train_rules(train, args.seed, args.corrections)
        model.training_summary["generator"] = {"games": args.games, "plies": args.plies,
                                                "sparse": args.sparse, "special": args.special,
                                                "capture_lessons": True}
        model.save(args.model)
    # A separate generator seed; no evaluation examples enter correction mining.
    positions = teaching_positions(args.eval_seed, args.eval_games, args.plies, args.eval_sparse, args.eval_special)
    metrics = evaluate_rules(model, positions)
    report = {"scope": "teacher-induced standard chess legality and piece effects", "eval_seed": args.eval_seed,
              "evaluation_generator": {"games": args.eval_games, "plies": args.plies,
                                       "sparse": args.eval_sparse, "special": args.eval_special},
              "training": model.training_summary, "evaluation": metrics,
              "limitations": ["finite tests do not prove correctness on every board",
                              "symbolic public board input", "no Chess960 or variant support",
                              "no learned value, strategic search, Elo, or draw-history adjudication",
                              "effect prediction covers pieces; next metadata is not predicted"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("failures", "effect_failures")}, indent=2))


if __name__ == "__main__":
    main()
