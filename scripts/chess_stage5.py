"""Original neural baseline. Use chess_rules.py for the current rule learner."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from pai.chess.learner import ChessWorldModel, evaluate, fit
from pai.chess.teacher import curriculum, position_key, split_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--plies", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--policy-positions", type=int, default=12)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/chess_stage5/model.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/chess_stage5.json"))
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.policy_positions < 1:
        parser.error("--threads and --policy-positions must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    train, test = split_examples(curriculum(args.seed, args.games, args.plies))
    model = ChessWorldModel().to(args.device)
    train_keys = sorted({position_key(row.before) for row in train})
    history = []
    if args.evaluate_only:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        train_keys = checkpoint["training_positions"]
        # Exclude anything the loaded model actually saw, even for a different seed.
        seen = set(train_keys)
        test = [row for row in test if position_key(row.before) not in seen]
    else:
        history = fit(model, train, epochs=args.epochs, seed=args.seed)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"schema": 1, "state_dict": model.state_dict(),
                    "training_positions": train_keys, "seed": args.seed,
                    "games": args.games, "plies": args.plies}, args.checkpoint)
    assert not set(train_keys) & {position_key(row.before) for row in test}
    report = {"stage": 5, "scope": "teacher and learned rule/effect baseline",
              "seed": args.seed, "games": args.games, "plies": args.plies,
              "checkpoint": str(args.checkpoint), "training_positions": len(train_keys),
              "test_positions": len({position_key(row.before) for row in test}),
              "position_overlap": 0, "training": history, "held_out": evaluate(model, test),
              "remaining": ["learned search and self-play", "Stockfish matches and measured Elo",
                            "optional physical chess"]}
    positions = {}
    for row in test:
        group = positions.setdefault(position_key(row.before), {"observation": row.before, "legal": set()})
        if row.legal:
            group["legal"].add(row.action)
    choices = random.Random(args.seed).sample(list(positions.values()), min(args.policy_positions, len(positions)))
    # Score after the agent chooses from the whole syntactic action space.
    trials = []
    for group in choices:
        action, confidence = model.choose(group["observation"])
        trials.append({"position": position_key(group["observation"]), "move": action.uci(),
                       "confidence": confidence, "legal": action in group["legal"]})
    report["unmasked_policy"] = {"positions": len(trials), "legal_moves": sum(t["legal"] for t in trials),
                                 "trials": trials}
    if not report["held_out"]["rule_gate_passed"]:
        report["remaining"].insert(0, "99% learned-rule gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("training",)}, indent=2))


if __name__ == "__main__":
    main()
