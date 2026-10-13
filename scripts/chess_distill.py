"""Teacher-distilled chess value: generate labelled positions, train, evaluate.

Splits use disjoint worker seeds: train 10000+, validation 20000+, fresh test
30000+ (the test split is only scored once, for the final report).
"""
import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from pai.chess.distill import (DistilledValue, evaluate_rows, generate_teacher_positions, position_hash,
                               rows_to_arrays, train_value)
from pai.chess.distill import register_rows
from pai.chess.provenance import metadata

SPLIT_SEEDS = {"train": 10000, "val": 20000, "test": 30000}


def load_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [tuple(json.loads(line)) for line in handle]


def generate(args):
    engine = os.path.abspath(args.engine)
    per = -(-args.positions // args.workers)
    base = SPLIT_SEEDS[args.split] + args.seed_offset
    jobs = [(engine, base + i, per, args.depth) for i in range(args.workers)]
    start = time.time()
    with Pool(args.workers) as pool:
        parts = pool.map(generate_teacher_positions, jobs)
    rows = [row for part in parts for row in part][:args.positions]
    path = args.directory / f"{args.split}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    meta = {"split": args.split, "positions": len(rows), "worker_seeds": [j[1] for j in jobs],
            "teacher_depth": args.depth, "multipv_for_move_sampling": 4, "seconds": time.time() - start}
    meta.update(metadata(meta["worker_seeds"], {"teacher": {"depth": args.depth}}))
    (args.directory / f"{args.split}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(meta, flush=True)


def train(args):
    import torch
    from pai.chess.rules import LearnedChessRules
    from pai.chess.evaluation import LearnedPositionValue
    rules = LearnedChessRules.load(args.rules)
    train_rows = load_rows(args.train_rows or args.directory / "train.jsonl.gz")
    val_rows = load_rows(args.directory / "val.jsonl.gz")
    held = {position_hash(row[0]) for row in val_rows}
    before = len(train_rows)
    train_rows = [r for r in train_rows if position_hash(r[0]) not in held]
    # Validation must not contain training positions either (common openings).
    train_keys = {position_hash(row[0]) for row in train_rows}
    val_rows = [r for r in val_rows if position_hash(r[0]) not in train_keys]
    print(f"train {len(train_rows)} (removed {before - len(train_rows)} overlapping), val {len(val_rows)}", flush=True)
    x, y = rows_to_arrays(train_rows, rules.attack_table)
    xv, yv = rows_to_arrays(val_rows, rules.attack_table)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.set_num_threads(6)
    start = time.time()
    layers, history = train_value(x.astype(np.float32), y, xv.astype(np.float32), yv, hidden=tuple(args.hidden),
                                  epochs=args.epochs, seed=args.seed, batch=args.batch, device=device)
    meta = json.loads((args.directory / "train.meta.json").read_text())
    summary = {"type": "distilled_value", "teacher": "Stockfish 19 (teaching environment)",
               "teacher_depth": meta["teacher_depth"], "train_positions": len(train_rows),
               "validation_positions": len(val_rows), "removed_overlap": before - len(train_rows),
               "hidden": args.hidden, "epochs": args.epochs, "seed": args.seed, "device": device,
               "selected_epoch": int(np.argmin([h["validation_mae_cp"] for h in history])),
               "train_seconds": time.time() - start, "history": history,
               "inputs": "side-to-move token planes, learned-rule attack counts, castling"}
    summary["training_positions"] = sorted(train_keys | {position_hash(row[0]) for row in val_rows})
    value = DistilledValue(layers, rules.attack_table, summary)
    register_rows(value, train_rows + val_rows)
    value.save(args.output_model)
    incumbent = LearnedPositionValue.load(args.incumbent)
    report = {"summary": {k: v for k, v in summary.items() if k not in ("history", "training_positions", "training_position_hashes")}, "history": history,
              "validation": {"candidate": evaluate_rows(value, val_rows),
                             "incumbent_linear": evaluate_rows(incumbent, val_rows)}}
    report.update(metadata([args.seed, *meta["worker_seeds"]], {"teacher": {"depth": meta["teacher_depth"]}}))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["validation"], indent=2))


def test(args):
    from pai.chess.evaluation import LearnedPositionValue
    value = DistilledValue.load(args.output_model)
    rows = load_rows(args.directory / "test.jsonl.gz")
    train_keys = {position_hash(row[0]) for split in ("train", "val") for row in load_rows(args.directory / f"{split}.jsonl.gz")}
    fresh = [r for r in rows if position_hash(r[0]) not in train_keys]
    report = {"fresh_test_seed_base": SPLIT_SEEDS["test"], "positions_total": len(rows),
              "excluded_training_overlap": len(rows) - len(fresh),
              "candidate": evaluate_rows(value, fresh),
              "incumbent_linear": evaluate_rows(LearnedPositionValue.load(args.incumbent), fresh)}
    test_meta = json.loads((args.directory / "test.meta.json").read_text())
    report.update(metadata(test_meta["worker_seeds"], {"teacher": {"depth": test_meta["teacher_depth"]}}))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("generate", "train", "test", "register"))
    p.add_argument("--engine", default="runs/chess_teacher/stockfish/stockfish-windows-x86-64-universal.exe")
    p.add_argument("--directory", type=Path, default=Path("runs/chess_distill"))
    p.add_argument("--split", choices=tuple(SPLIT_SEEDS), default="train")
    p.add_argument("--positions", type=int, default=20000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--rules", type=Path, default=Path("runs/chess_rules/rules.json"))
    p.add_argument("--incumbent", type=Path, default=Path("runs/chess_rules/value.json"))
    p.add_argument("--output-model", type=Path, default=Path("runs/chess_distill/value.npz"))
    p.add_argument("--report", type=Path, default=Path("results/chess_distill_value.json"))
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 64])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--train-rows", type=Path)
    args = p.parse_args()
    if args.workers < 1 or args.workers > 6:
        p.error("use 1-6 workers on the shared machine")
    if args.command == "register":
        value = DistilledValue.load(args.output_model)
        rows = load_rows(args.train_rows or args.directory / "train.jsonl.gz") + load_rows(args.directory / "val.jsonl.gz")
        register_rows(value, rows)
        value.save(args.output_model)
        print("Registered public positions and their colour mirrors", flush=True)
    else:
        {"generate": generate, "train": train, "test": test}[args.command](args)


if __name__ == "__main__":
    main()
