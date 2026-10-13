"""Equal-movetime ladder and paired promotion with immutable opening books."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pai.chess.ladder import anchored_elo, balanced_openings, file_hash, load_games, pair_statistics, run_schedule
from pai.chess.provenance import metadata

def jobs_for(group, candidate, opponent, roots, max_plies, seed_base, provenance):
    return [{"id": f"{group}|{pair}|{color}", "group": group, "pair": pair,
             "candidate_color": color, "white": candidate if color == "white" else opponent,
             "black": opponent if color == "white" else candidate, "root": root,
             "max_plies": max_plies, "seed": seed_base + pair, "provenance": provenance}
            for pair, root in enumerate(roots) for color in ("white", "black")]

def summarise(games, meta):
    groups, anchors = {}, {}
    for name, config in meta["groups"].items():
        rows = [g for g in games if g["group"] == name]
        stats = pair_statistics(rows)
        stats.update(players=config, scheduled_games=meta["pairs"] * 2,
                     schedule_complete=len(rows) == meta["pairs"] * 2,
                     failures=[g["failure"] for g in rows if g["failure"]])
        groups[name] = stats
        if config["kind"] == "ladder":
            seconds = str(config["candidate"]["time"])
            anchors.setdefault(seconds, []).append((config["opponent"]["uci_elo"], rows))
    return {**{k: meta[k] for k in ("provenance", "opening_seed", "roots", "checkpoint_hashes")},
            "groups": groups, "conditional_elo_by_movetime": {t: anchored_elo(g) for t, g in anchors.items()},
            "rating_scope": "conditional Elo relative to configured Stockfish anchors under stated limits; not calibrated human Elo",
            "raw_games": games}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["ladder", "promote", "summarise"])
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--directory", type=Path, default=Path("runs/chess_rules"))
    p.add_argument("--agent-value", type=Path, default=Path("runs/chess_rules/value.json"))
    p.add_argument("--incumbent-value", type=Path, default=Path("runs/chess_rules/value.json"))
    p.add_argument("--times", type=float, nargs="+", default=[.1, .3])
    p.add_argument("--rungs", type=int, nargs="+", default=[1320, 1500, 1700])
    p.add_argument("--opponent-depth", type=int)
    p.add_argument("--nodes", type=int, default=1000000)
    p.add_argument("--quiescence", type=int, default=0)
    p.add_argument("--reuse-pv", action="store_true")
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--opening-seed", type=int, required=True)
    p.add_argument("--max-plies", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if min(args.times) <= 0 or min(args.pairs, args.max_plies, args.nodes) < 1 or not 1 <= args.workers <= 4 or args.quiescence < 0:
        p.error("positive times/counts, nonnegative quiescence and 1–4 workers required")
    if len(set(args.times)) != len(args.times) or len(set(args.rungs)) != len(args.rungs):
        p.error("times and rungs must be distinct")
    meta_path = args.output.with_suffix(".meta.json")
    report_path = args.report or args.output.with_suffix(".json")
    if args.mode == "summarise":
        meta = json.loads(meta_path.read_text())
    else:
        hashes = {str(path): file_hash(path) for path in (args.agent_value, args.incumbent_value,
                   *(args.directory / n for n in ("rules.json", "state.json", "endings.json")))}
        provenance = metadata([args.opening_seed], {"both_sides_movetime_seconds": args.times,
                              "opponent_extra_depth_cap": args.opponent_depth, "agent_node_safety_cap": args.nodes})
        groups = {}
        for seconds in args.times:
            candidate = {"kind": "agent", "directory": str(args.directory), "value": str(args.agent_value),
                         "time": seconds, "nodes": args.nodes, "quiescence": args.quiescence, "reuse_pv": args.reuse_pv}
            if args.mode == "promote":
                opponents = [("incumbent", {**candidate, "value": str(args.incumbent_value)})]
            else:
                opponents = [(f"elo{elo}", {"kind": "engine", "uci_elo": elo, "movetime": seconds,
                                           "engine_path": str(args.engine.resolve()),
                                           **({"depth": args.opponent_depth} if args.opponent_depth else {})}) for elo in args.rungs]
            for label, opponent in opponents:
                name = f"t{seconds}_{label}"
                groups[name] = {"kind": "promotion" if args.mode == "promote" else "ladder", "candidate": candidate, "opponent": opponent}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta["groups"] != groups or meta["checkpoint_hashes"] != hashes or meta["opening_seed"] != args.opening_seed
                    or meta["pairs"] != args.pairs or meta["max_plies"] != args.max_plies
                    or meta["provenance"]["source_sha256"] != provenance["source_sha256"]):
                p.error("existing run differs in code, players, checkpoints or opening schedule")
        else:
            roots = balanced_openings(args.pairs, args.opening_seed, depth=8, engine_path=args.engine)
            if len(set(roots)) != len(roots):
                p.error("opening generator produced duplicate roots")
            meta = {"provenance": provenance, "opening_seed": args.opening_seed, "pairs": args.pairs,
                    "max_plies": args.max_plies, "roots": roots, "groups": groups, "checkpoint_hashes": hashes}
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        jobs = []
        for name, group in groups.items():
            jobs.extend(jobs_for(name, group["candidate"], group["opponent"], meta["roots"], args.max_plies,
                                 args.opening_seed, meta["provenance"]))
        run_schedule(jobs, args.output, args.workers)
    report = summarise(load_games(args.output), meta)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"groups": report["groups"], "conditional_elo_by_movetime": report["conditional_elo_by_movetime"]}, indent=2))

if __name__ == "__main__":
    main()
