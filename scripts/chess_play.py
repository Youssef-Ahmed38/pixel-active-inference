"""Run self-play updates, teacher validation and measured Stockfish games."""
import argparse
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pai.chess import LearnedChessRules, LearnedChessStateModel, LearnedChessEvaluator, LearnedPositionValue, LearnedChessSearch
from pai.chess.play import collect_selfplay, fit_selfplay_value, stockfish_matches, resume_matches
from pai.chess.rule_training import teaching_positions
from pai.chess.evaluation_training import engine_labels, evaluate_value
from pai.chess.teacher import position_key
from pai.chess.hrm_value import load_value
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, default=Path("runs/chess_rules"))
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/chess_play.json"))
    p.add_argument("--artifacts", type=Path, default=Path("runs/chess_play"))
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--selfplay-nodes", type=int, default=64)
    p.add_argument("--max-plies", type=int, default=96)
    p.add_argument("--levels", nargs="+", type=int, default=[1320, 1800, 2400])
    p.add_argument("--budgets", nargs="+", type=int, default=[32, 128])
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--engine-nodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--selfplay-only", action="store_true")
    p.add_argument("--matches-only", action="store_true")
    p.add_argument("--resume-report", type=Path)
    p.add_argument("--value-checkpoint", type=Path)
    p.add_argument("--ordering-checkpoint", type=Path)
    p.add_argument("--quiescence", type=int, default=0)
    p.add_argument("--reuse-pv", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    if any(n < 1 for n in [args.games, args.selfplay_nodes, args.max_plies, args.pairs, args.engine_nodes, *args.budgets]):
        p.error("game counts and budgets must be positive")
    if args.quiescence < 0:
        p.error("quiescence must be nonnegative")
    rules = LearnedChessRules.load(args.directory / "rules.json")
    state = LearnedChessStateModel.load(args.directory / "state.json", rules)
    endings = LearnedChessEvaluator.load(args.directory / "endings.json", state)
    value_path = args.value_checkpoint or args.directory / "value.json"
    baseline = load_value(value_path)
    if not isinstance(baseline, LearnedPositionValue) and not (args.matches_only or args.resume_report):
        p.error("experimental value checkpoints currently support matches; linear self-play fitting requires the original JSON value checkpoint")
    ordering = load_value(args.ordering_checkpoint) if args.ordering_checkpoint else None
    def make_planner(value):
        from functools import partial
        result = LearnedChessSearch(state, endings, value, ordering)
        if args.quiescence or args.reuse_pv:
            result.search = partial(result.search, max_quiescence=args.quiescence, reuse_pv=args.reuse_pv)
        return result
    planner = make_planner(baseline)
    manifest = {name: hashlib.sha256((args.directory / name).read_bytes()).hexdigest()
                for name in ("rules.json", "state.json", "endings.json")}
    manifest["value.json"] = hashlib.sha256(value_path.read_bytes()).hexdigest()
    if args.ordering_checkpoint:
        manifest["ordering_value"] = hashlib.sha256(args.ordering_checkpoint.read_bytes()).hexdigest()
    search_config = {"quiescence": args.quiescence, "reuse_pv": args.reuse_pv}
    if args.resume_report:
        report = json.loads(args.resume_report.read_text(encoding="utf-8"))
        if report.get("model_manifest") is not None and report["model_manifest"] != manifest:
            p.error("resume checkpoints do not match the original match run")
        previous_config = {"quiescence": 0, "reuse_pv": False}
        previous_config.update(report.get("search_config", {}))
        if previous_config != search_config:
            p.error("resume search settings do not match the original match run")
        if "model_manifest" not in report:
            report["legacy_model_identity_unverified"] = True
        report["resume_model_manifest"] = manifest
        args.output.parent.mkdir(parents=True, exist_ok=True)
        def save_progress(matches):
            report["matches"] = matches
            report["resumed_from"] = str(args.resume_report)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        matches = resume_matches(planner, report.get("matches", []), args.engine, args.max_plies,
                                 report["seed"] + 100, save_progress)
        schedule = report.get("match_schedule")
        if schedule is not None:
            matches = stockfish_matches(planner, args.engine, schedule["levels"], schedule["budgets"],
                                        schedule["pairs"], schedule["engine_nodes"], args.max_plies,
                                        schedule["seed"], args.artifacts / "matches",
                                        on_progress=save_progress, existing_reports=matches)
        save_progress(matches)
        return
    args.artifacts.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "model_manifest": manifest, "search_config": search_config,
              "limitations": ["small samples cannot establish playing strength or improvement",
              "UCI_Elo settings are not calibrated to these node-limited games",
              "truncations are not counted as draws", "self-play targets are bootstrapped search estimates"]}
    if not args.matches_only:
        games, replay = collect_selfplay(planner, args.games, args.selfplay_nodes, min(args.max_plies, 48), args.seed)
        (args.artifacts / "selfplay.json").write_text(json.dumps(games, indent=2) + "\n", encoding="utf-8")
        (args.artifacts / "replay.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
        candidate, fit = fit_selfplay_value(baseline, replay)
        candidate.save(args.artifacts / "candidate.json")
        positions = teaching_positions(args.seed + 17, 8, 80, 40, 30)
        rows, _ = engine_labels(args.engine, [(b,t) for b,t in positions if b.outcome() is None], 4000)
        excluded = set(baseline.summary.get("training_positions", ())) | set(candidate.summary["training_positions"])
        rows = [(obs, score, topic) for obs, score, topic in rows if position_key(obs) not in excluded]
        base_gate, candidate_gate = evaluate_value(baseline, rows), evaluate_value(candidate, rows)
        # Keep a separate candidate. No incumbent overwrite on a noisy/degraded run.
        accepted = candidate_gate["mae_cp"] < base_gate["mae_cp"] and not any(g["failure"] for g in games)
        report["selfplay"] = {"games": len(games), "complete": sum(g["status"] == "complete" for g in games),
                              "truncated": sum(g["status"] == "truncated" for g in games),
                              "failures": [g["failure"] for g in games if g["failure"]], "fit": fit,
                              "baseline_validation": base_gate, "candidate_validation": candidate_gate,
                              "candidate_accepted_for_trial": accepted}
        if accepted:
            planner = make_planner(candidate)
            report["model_manifest"]["value.json"] = hashlib.sha256((args.artifacts / "candidate.json").read_bytes()).hexdigest()
            report["match_value_checkpoint"] = str(args.artifacts / "candidate.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.selfplay_only:
        report["match_schedule"] = {"levels": args.levels, "budgets": args.budgets, "pairs": args.pairs,
                                    "engine_nodes": args.engine_nodes, "seed": args.seed + 100}
        report["matches"] = []
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.selfplay_only:
        def save_matches(matches):
            report["matches"] = matches
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output)
        report["matches"] = stockfish_matches(planner, args.engine, args.levels, args.budgets, args.pairs,
                                              args.engine_nodes, args.max_plies, args.seed + 100, args.artifacts / "matches",
                                              on_progress=save_matches)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "matches"}, indent=2))


if __name__ == "__main__":
    main()
