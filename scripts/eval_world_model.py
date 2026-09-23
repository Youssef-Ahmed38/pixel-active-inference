"""Multi-step prediction error of the entity world model (open-loop rollouts on held-out episodes).

    python scripts/eval_world_model.py --config configs/tabletop.yaml --horizons 1 5 20

Reports position error (mm) at each horizon, for all entities and for entities that move, and
compares with a "nothing moves except the gripper by its command" baseline. Writes
results/world_model_eval.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from _cli import parse

from pai.world.entities import POS
from pai.world.model import load_world_model


def extra(ap):
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--data-dir", default="data/slice_heldout", help="episodes never used for training")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.1)


@torch.no_grad()
def main():
    args, cfg = parse(__doc__, extra)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = load_world_model(cfg.slice.world_model, device)
    files = sorted(Path(args.data_dir).glob("ep_*.npz"))[: args.episodes]
    H = max(args.horizons)
    errs = {h: [] for h in args.horizons}
    moving = {h: [] for h in args.horizons}
    base = {h: [] for h in args.horizons}
    for f in files:
        d = np.load(f)
        tok = torch.as_tensor(d["tokens"], device=device)
        act = torch.as_tensor(d["actions"], device=device)
        T = len(act)
        starts = torch.arange(0, T - H, 5, device=device)
        if len(starts) == 0:
            continue
        x = tok[starts].clone()
        b = tok[starts].clone()
        for h in range(1, H + 1):
            x = wm.predict(x, act[starts + h - 1]).next_tokens
            b[:, 0, POS] = b[:, 0, POS] + act[starts + h - 1, :3] * args.dt  # baseline: commanded gripper motion only
            if h in errs:
                truth = tok[starts + h][..., POS]
                e = (x[..., POS] - truth).norm(dim=-1)
                eb = (b[..., POS] - truth).norm(dim=-1)
                mv = (truth - tok[starts][..., POS]).norm(dim=-1) > 0.005
                errs[h].append(e.flatten())
                base[h].append(eb.flatten())
                moving[h].append(torch.stack([e[mv], eb[mv]], -1))
    out = {}
    for h in args.horizons:
        e, eb, m = torch.cat(errs[h]), torch.cat(base[h]), torch.cat(moving[h])
        out[h] = {"all_mm": round(1000 * e.mean().item(), 2), "moving_mm": round(1000 * m[:, 0].mean().item(), 2),
                  "baseline_all_mm": round(1000 * eb.mean().item(), 2),
                  "baseline_moving_mm": round(1000 * m[:, 1].mean().item(), 2)}
        print(f"{h:3d} steps ({h * args.dt:.1f} s): all {out[h]['all_mm']:6.2f} mm  moving {out[h]['moving_mm']:6.2f} mm"
              f"   | baseline all {out[h]['baseline_all_mm']:6.2f}  moving {out[h]['baseline_moving_mm']:6.2f}")
    Path("results").mkdir(exist_ok=True)
    Path("results/world_model_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
