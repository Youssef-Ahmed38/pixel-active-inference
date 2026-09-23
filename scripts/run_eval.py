"""Evaluate PixelAI on reaching or perception, optionally under disturbances.

    python scripts/run_eval.py --decoder runs/pixelai_decoder/decoder.pt
    python scripts/run_eval.py --task perception --decoder runs/pixelai_decoder/decoder.pt
    python scripts/run_eval.py --no-vision agent.goal_mode=joints            # privileged sanity check
    python scripts/run_eval.py --decoder ... --disturbance occlusion         # a preset from configs/disturbances.yaml
"""

import json

import torch
import yaml
from _cli import parse

from pai.config import REPO_ROOT
from pai.eval.reach import evaluate
from pai.models import load_decoder


def extra(ap):
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--task", default="reach", choices=["reach", "perception"])
    ap.add_argument("--no-vision", action="store_true", help="proprioception only (sets agent.pi_v=0)")
    ap.add_argument("--disturbance", default=None, help="preset name in configs/disturbances.yaml")
    ap.add_argument("--tag", default=None)


if __name__ == "__main__":
    args, cfg = parse(__doc__, extra)
    if args.no_vision:
        cfg.agent.pi_v = 0.0
    if args.disturbance:
        presets = yaml.safe_load((REPO_ROOT / "configs" / "disturbances.yaml").read_text())
        cfg.env.disturbances = presets[args.disturbance]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = load_decoder(args.decoder, device) if args.decoder else None
    tag = args.tag or ("pixelai" + (f"_{args.disturbance}" if args.disturbance else ""))
    print(json.dumps(evaluate(cfg, decoder, device, args.task, tag), indent=2))
