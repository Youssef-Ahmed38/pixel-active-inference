"""Rating ladder and paired promotion matches for the learned chess agent.

Ladder: the agent plays both colours from each balanced opening against limited
Stockfish rungs. Promote: candidate agent versus incumbent agent, same roots,
both colours; the candidate replaces the incumbent only if the pair-bootstrap
95% lower bound of its score is above 0.5 on fresh opening seeds.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pai.chess.ladder import (anchored_elo, balanced_openings, file_hash, load_games, pair_statistics,
                              run_schedule)

RUNGS = {
    # Stockfish Skill Level with a depth limit (weak, not Elo-anchored).
    "sf_s0_d1": {"kind": "engine", "skill": 0, "depth": 1},
    "sf_s0_d3": {"kind": "engine", "skill": 0, "depth": 3},
    "sf_s0_d6": {"kind": "engine", "skill": 0, "depth": 6},
    "sf_s3_d6": {"kind": "engine", "skill": 3, "depth": 6},
    "sf_s6_d8": {"kind": "engine", "skill": 6, "depth": 8},
    # Configured UCI_Elo anchors (Stockfish minimum is 1320), 50 ms per move.
    "sf_elo1320_t0.05": {"kind": "engine", "uci_elo": 1320, "movetime": 0.05},
    "sf_elo1500_t0.05": {"kind": "engine", "uci_elo": 1500, "movetime": 0.05},
    "sf_elo1700_t0.05": {"kind": "engine", "uci_elo": 1700, "movetime": 0.05},
    "sf_elo2000_t0.05": {"kind": "engine", "uci_elo": 2000, "movetime": 0.05},
    "sf_elo1320_n1000": {"kind": "engine", "uci_elo": 1320, "nodes": 1000},
}


def agent_config(value, time_limit, nodes, quiescence, directory):
    config = {"kind": "agent", "directory": str(directory).replace("\\", "/"),
              "value": str(value).replace("\\", "/"), "time": time_limit, "quiescence": quiescence}
    if nodes:
        config["nodes"] = nodes
    return config


def jobs_for(group, candidate, opponent, roots, max_plies, seed_base):
    jobs = []
    for pair, root in enumerate(roots):
        for colour in ("white", "black"):
            white, black = (candidate, opponent) if colour == "white" else (opponent, candidate)
            jobs.append({"id": f"{group}|{pair}|{colour}", "group": group, "pair": pair, "candidate_color": colour,
                         "white": white, "black": black, "root": root, "max_plies": max_plies,
                         "seed": seed_base + pair})
    return jobs


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarise(games, groups_meta, path):
    groups = {}
    for g in games:
        groups.setdefault(g["group"], []).append(g)
    report = {"groups": {}, "execution_failures": [g["failure"] for g in games if g["failure"]]}
    anchored = []
    for name, rows in sorted(groups.items()):
        stats = pair_statistics(rows)
        meta = groups_meta.get(name, {})
        terminations = {}
        for g in rows:
            key = g["termination"] or g["status"]
            terminations[key] = terminations.get(key, 0) + 1
        stats.update({"opponent": meta.get("opponent"), "candidate": meta.get("candidate"),
                      "mean_plies": sum(g["plies"] for g in rows) / len(rows),
                      "candidate_mean_completed_depth": _mean([g["agent_depth_mean"] for g in rows]),
                      "terminations": terminations})
        report["groups"][name] = stats
        opponent = meta.get("opponent") or {}
        if "uci_elo" in opponent and meta.get("kind") == "ladder":
            anchored.append((opponent["uci_elo"], rows))
    if anchored:
        report["anchored_elo"] = anchored_elo(anchored)
        report["anchored_rungs"] = [a for a, _ in anchored]
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("ladder", "promote", "summarise"))
    p.add_argument("--directory", type=Path, default=Path("runs/chess_rules"))
    p.add_argument("--agent-value", type=Path, default=Path("runs/chess_rules/value.json"))
    p.add_argument("--incumbent-value", type=Path, default=Path("runs/chess_rules/value.json"))
    p.add_argument("--time", type=float, default=1.0, help="agent seconds per move")
    p.add_argument("--nodes", type=int, default=0, help="optional agent node cap")
    p.add_argument("--incumbent-time", type=float, default=None)
    p.add_argument("--incumbent-nodes", type=int, default=0)
    p.add_argument("--quiescence", type=int, default=0)
    p.add_argument("--incumbent-quiescence", type=int, default=0)
    p.add_argument("--rungs", nargs="+", default=["sf_s0_d1", "sf_elo1320_t0.05"])
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--opening-seed", type=int, required=True)
    p.add_argument("--max-plies", type=int, default=400)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output", type=Path, required=True, help="JSONL of finished games (resumable)")
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if not 1 <= args.workers <= 6:
        p.error("at most 6 workers on the shared machine")
    for rung in args.rungs:
        if rung not in RUNGS:
            p.error(f"unknown rung {rung}; choose from {sorted(RUNGS)}")
    report_path = args.report or args.output.with_suffix(".json")
    meta_path = args.output.with_suffix(".meta.json")
    if args.mode == "summarise":
        groups_meta = json.loads(meta_path.read_text())["groups"]
    else:
        candidate = agent_config(args.agent_value, args.time, args.nodes, args.quiescence, args.directory)
        roots = balanced_openings(args.pairs, args.opening_seed)
        jobs, groups_meta = [], {}
        if args.mode == "ladder":
            for rung in args.rungs:
                jobs += jobs_for(rung, candidate, RUNGS[rung], roots, args.max_plies, args.opening_seed)
                groups_meta[rung] = {"kind": "ladder", "opponent": RUNGS[rung], "candidate": candidate}
        else:
            incumbent = agent_config(args.incumbent_value, args.incumbent_time or args.time, args.incumbent_nodes,
                                     args.incumbent_quiescence, args.directory)
            jobs = jobs_for("candidate_vs_incumbent", candidate, incumbent, roots, args.max_plies, args.opening_seed)
            groups_meta["candidate_vs_incumbent"] = {"kind": "promotion", "opponent": incumbent,
                                                     "candidate": candidate}
        hashes = {str(v).replace("\\", "/"): file_hash(v) for v in {args.agent_value, args.incumbent_value}}
        hashes.update({n: file_hash(args.directory / n) for n in ("rules.json", "state.json", "endings.json")})
        meta = {"mode": args.mode, "opening_seed": args.opening_seed, "pairs": args.pairs, "roots": roots,
                "max_plies": args.max_plies, "checkpoint_sha256_16": hashes, "groups": groups_meta,
                "command": " ".join(sys.argv)}
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            same = (old["roots"] == roots and old["checkpoint_sha256_16"] == hashes
                    and all(old["groups"].get(k) == v for k, v in groups_meta.items() if k in old["groups"]))
            if not same:
                p.error("existing game file was produced with different openings, checkpoints or players")
            old["groups"].update(groups_meta)
            meta["groups"] = groups_meta = old["groups"]
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        run_schedule(jobs, args.output, args.workers)
    report = summarise(load_games(args.output), groups_meta, report_path)
    print(json.dumps({k: v for k, v in report.items() if k != "execution_failures"}, indent=2))
    print("execution failures:", len(report["execution_failures"]))


if __name__ == "__main__":
    main()
