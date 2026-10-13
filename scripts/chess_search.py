"""Evaluate bounded learned chess search and draw-claim execution."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai.chess import LearnedChessRules, LearnedChessStateModel, LearnedChessEvaluator, LearnedPositionValue, LearnedChessSearch
from pai.chess.search_evaluation import evaluate_search, evaluate_claim_execution
from pai.chess.mcts import LearnedChessMCTS
from pai.chess.hrm_value import load_value
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("runs/chess_rules"))
    parser.add_argument("--planner", choices=("alpha-beta", "mcts"), default="alpha-beta")
    parser.add_argument("--value-checkpoint", type=Path)
    parser.add_argument("--ordering-checkpoint", type=Path,
                        help="optional separate move-ordering value for alpha-beta")
    parser.add_argument("--quiescence", type=int, default=0)
    parser.add_argument("--reuse-pv", action="store_true")
    parser.add_argument("--eval-seed", type=int, default=410)
    parser.add_argument("--nodes", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--per-topic", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/chess_search.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.quiescence < 0 or (args.planner == "mcts" and args.quiescence):
        parser.error("quiescence must be nonnegative and currently supports alpha-beta only")
    if args.planner == "mcts" and (args.ordering_checkpoint or args.reuse_pv):
        parser.error("separate move ordering and PV reuse currently support alpha-beta only")
    if args.per_topic < 1 or any(n < 1 for n in args.nodes):
        parser.error("topic count and node budgets must be positive")
    r = LearnedChessRules.load(args.directory / "rules.json")
    s = LearnedChessStateModel.load(args.directory / "state.json", r)
    e = LearnedChessEvaluator.load(args.directory / "endings.json", s)
    v = load_value(args.value_checkpoint or args.directory / "value.json")
    planner = (LearnedChessMCTS(s, e, v) if args.planner == "mcts" else
               LearnedChessSearch(s, e, v, load_value(args.ordering_checkpoint) if args.ordering_checkpoint else None))
    if args.quiescence or args.reuse_pv:
        from functools import partial
        planner.search = partial(planner.search, max_quiescence=args.quiescence, reuse_pv=args.reuse_pv)
    print("Evaluating fresh tactical positions", flush=True)
    metrics = evaluate_search(planner, args.eval_seed, args.nodes, args.per_topic)
    claims = evaluate_claim_execution(planner)
    report = {"planner": args.planner, "quiescence": args.quiescence, "reuse_pv": args.reuse_pv,
              "value_checkpoint": str(args.value_checkpoint or args.directory / "value.json"),
              "ordering_checkpoint": str(args.ordering_checkpoint) if args.ordering_checkpoint else None,
              "evaluation": metrics, "claim_execution": claims,
              "limitations": ["finite synthetic tactical families", "no measured Elo or self-play improvement",
                              "node budget counts inspected states, not every model operation",
                              "time limit is checked between search nodes, not a hard-real-time bound",
                              "HRM value is experimental; chess halting is not trained",
                              "MCTS completed_depth is zero: sampled depth is recorded separately as deepest_ply"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"budgets": [{k: val for k, val in b.items() if k != "cases"} for b in metrics["budgets"]],
                      "claims": claims}, indent=2))


if __name__ == "__main__":
    main()
