"""Compare fixed chess value/search candidates on shared fresh tactical boards."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from pai.chess import LearnedChessRules, LearnedChessStateModel, LearnedChessEvaluator, LearnedChessSearch
from pai.chess.hrm_value import load_value
from pai.chess.search_evaluation import evaluate_search, evaluate_claim_execution


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("runs/chess_rules"))
    p.add_argument("--candidate", type=Path, default=Path("runs/chess_context/candidate.json"))
    p.add_argument("--seeds", type=int, nargs="+", default=[710, 711])
    p.add_argument("--nodes", type=int, nargs="+", default=[32, 128])
    p.add_argument("--per-topic", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results/chess_search_comparison.json"))
    args = p.parse_args()
    if args.per_topic < 1 or any(n < 1 for n in args.nodes):
        p.error("topic count and node budgets must be positive")
    if len(args.seeds) != len(set(args.seeds)) or len(args.nodes) != len(set(args.nodes)):
        p.error("seeds and budgets must be distinct")
    torch.set_num_threads(2)
    rules = LearnedChessRules.load(args.directory / "rules.json")
    state = LearnedChessStateModel.load(args.directory / "state.json", rules)
    endings = LearnedChessEvaluator.load(args.directory / "endings.json", state)
    paths = {"baseline": args.directory / "value.json", "candidate": args.candidate}
    values = {name: load_value(path) for name, path in paths.items()}
    # All variants see exactly the same boards, excluding any model's teaching.
    excluded = set().union(*(set(v.summary.get("training_positions", ())) for v in values.values()))
    seen = set()
    report = {"seeds": args.seeds, "node_budgets": args.nodes, "variants": {},
              "model_manifest": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [*(args.directory / n for n in ("rules.json", "state.json", "endings.json")), *paths.values()]},
              "limitations": ["synthetic tactical families do not establish playing strength",
                              "candidate variants are experimental and do not replace checkpoints",
                              "seeds share a generator; repeated boards are excluded"]}
    for seed in args.seeds:
        seed_boards = set()
        for name, value in values.items():
            for reuse in (False, True):
                label = name + ("_pv" if reuse else "")
                planner = LearnedChessSearch(state, endings, value)
                planner.search = partial(planner.search, reuse_pv=reuse)
                result = evaluate_search(planner, seed, args.nodes, args.per_topic, excluded | seen)
                variant = report["variants"].setdefault(label, {"evaluations": [], "claims": evaluate_claim_execution(planner)})
                variant["evaluations"].append(result)
                from pai.chess.teacher import position_key, observation
                import chess
                seed_boards.update(position_key(observation(chess.Board(c["fen"])))
                                   for c in result["budgets"][0]["cases"])
                print(f"seed {seed}: {label} completed", flush=True)
        seen.update(seed_boards)
    for variant in report["variants"].values():
        variant["totals"] = []
        for budget in args.nodes:
            rows = [b for e in variant["evaluations"] for b in e["budgets"] if b["node_budget"] == budget]
            keys = ("positions", "legal_choices", "exact_root_legal_sets", "valid_predicted_variations",
                    "within_node_budget", "mate_in_one_solved", "poisoned_rook_avoided")
            total = {k: sum(r[k] for r in rows) for k in keys}
            total.update(node_budget=budget, correctness_gate_passed=all(r["correctness_gate_passed"] for r in rows))
            variant["totals"].append(total)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["totals"] for k, v in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
