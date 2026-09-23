"""Evaluate several override sets and print a comparison table.

    python scripts/sweep.py --decoder runs/pixelai_decoder/decoder.pt \
        --variant "agent.beta=4 agent.pi_mu=16" --variant "agent.beta=10 agent.pi_mu=40" eval.episodes=10

Each --variant is a space-separated list of overrides applied on top of the shared ones.
Results go to results/sweep_<name>.json.
"""

import json
from pathlib import Path

import torch
from _cli import parse

from pai.config import Config, apply_overrides
from pai.eval.reach import evaluate
from pai.models import load_decoder

COLUMNS = ["start_ee_dist_median", "success_rate", "final_ee_dist_median", "steps_to_success_median", "final_belief_err_mean"]


def extra(ap):
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--task", default="reach", choices=["reach", "perception"])
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--name", default="sweep")


if __name__ == "__main__":
    args, base = parse(__doc__, extra)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = load_decoder(args.decoder, device) if args.decoder else None
    rows = []
    for i, variant in enumerate(args.variant):
        cfg = Config(apply_overrides(base.to_dict(), variant.split() + ["eval.save_gif=false"]))
        s = evaluate(cfg, decoder, device, args.task, tag=f"{args.name}_{i}")
        rows.append({"variant": variant, **{k: s[k] for k in COLUMNS}})
        print(f"{variant:45s} " + " ".join(f"{k}={s[k]}" for k in COLUMNS), flush=True)
    out = Path("results") / f"sweep_{args.name}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"-> {out}")
