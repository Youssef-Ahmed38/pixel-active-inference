"""Stage 2, step 1: motor babbling builds the L0 body model (random vs curiosity-driven).

    python scripts/locomotion_babble.py --config configs/locomotion.yaml

Writes runs/locomotion/body_model.pt (trained on both explorers' data) and
results/locomotion/babble.json with held-out prediction errors for each explorer.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts._cli import parse  # noqa: E402
from pai.locomotion.babble import BabbleConfig, babble, fit_model  # noqa: E402
from pai.locomotion.env import EnvConfig  # noqa: E402
from pai.locomotion.world_model import heldout_error  # noqa: E402


def main():
    args, cfg = parse(__doc__, lambda ap: (ap.add_argument("--out", default="runs/locomotion"),
                                          ap.add_argument("--results", default="results/locomotion"),
                                          ap.add_argument("--fit-steps", type=int, default=4000),
                                          ap.add_argument("--seed", type=int, default=100)))
    torch.set_num_threads(2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ecfg = EnvConfig.from_dict(cfg.get("env", {}).to_dict() if cfg.get("env") else {})
    bcfg = BabbleConfig(**(cfg.get("babble", {}) or {}))
    out, res = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    report = {"config": vars(bcfg), "seeds": {}}
    data = {}
    t0 = time.time()
    for mode, seed in (("random", args.seed), ("curious", args.seed + 1)):
        print(f"babbling ({mode}), seed {seed} ...")
        d, falls, _ = babble(mode, bcfg, ecfg, seed, dev)
        data[mode] = d
        report["seeds"][mode] = seed
        report[f"{mode}_transitions"] = int(len(d[0]))
        report[f"{mode}_falls"] = falls
        spread = d[0].std(0)
        report[f"{mode}_state_spread"] = float(np.mean(spread))
        print(f"  {len(d[0])} transitions, {falls} falls, {time.time() - t0:.0f}s")
    # held-out data from fresh seeds of both explorers
    print("held-out babbling (fresh seeds) ...")
    held = {}
    for mode, seed in (("random", args.seed + 10), ("curious", args.seed + 11)):
        small = BabbleConfig(**{**vars(bcfg), "steps": max(100, bcfg.steps // 4)})
        held[mode], _, _ = babble(mode, small, ecfg, seed, dev)
    models = {}
    for mode in ("random", "curious"):
        print(f"fitting body model on {mode} data ...")
        models[mode] = fit_model(data[mode], dev, args.fit_steps, seed=1)
    for trained in models:
        for test in held:
            report[f"error_{trained}_on_{test}"] = heldout_error(models[trained], *held[test], dev)
    print("fitting the combined body model ...")
    both = tuple(np.concatenate([data["random"][i], data["curious"][i]]) for i in range(3))
    combined = fit_model(both, dev, args.fit_steps * 2, seed=2)
    for test in held:
        report[f"error_combined_on_{test}"] = heldout_error(combined, *held[test], dev)
    torch.save({"model": combined.state_dict(), "members": combined.members,
                "transitions": int(len(both[0])), "report": report}, out / "body_model.pt")
    report["seconds"] = round(time.time() - t0, 1)
    (res / "babble.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k.startswith("error") or k.endswith("falls")}, indent=2))


if __name__ == "__main__":
    main()
