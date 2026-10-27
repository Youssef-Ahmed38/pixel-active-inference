"""Stage 2, step 2: the G1 practises stand -> balance -> walk -> turn -> run (PPO + curiosity).

    python scripts/locomotion_train.py --config configs/locomotion.yaml
    python scripts/locomotion_train.py --config configs/locomotion.yaml train.max_hours=0.05  # smoke

Resumes from <out>/ckpt_last.pt when it exists. Uses runs/locomotion/body_model.pt (from
scripts/locomotion_babble.py) to initialise the curiosity body model when present.

Under ``torchrun`` (Kaggle, 2 x T4) every rank is an independent replicate with its own seed
(``train.seed + rank``) and output folder ``<out>/rank<k>``; the simulator runs on the CPU, so
the GPUs only carry the networks. ``--no-curiosity-rank 1`` makes rank 1 the plain-PPO baseline
(same seed offset, no epistemic bonus) for the learning-curve comparison.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._cli import parse  # noqa: E402
from pai.locomotion.env import EnvConfig  # noqa: E402
from pai.locomotion.ppo import TrainConfig, Trainer  # noqa: E402


def main():
    args, cfg = parse(__doc__, lambda ap: (ap.add_argument("--out", default="runs/locomotion/policy"),
                                          ap.add_argument("--body-model", default="runs/locomotion/body_model.pt"),
                                          ap.add_argument("--device", default="cuda"),
                                          ap.add_argument("--no-curiosity-rank", type=int, default=-1,
                                                          help="this rank trains without the epistemic "
                                                               "bonus (the plain-PPO baseline)")))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ecfg = EnvConfig.from_dict(cfg.get("env").to_dict() if cfg.get("env") else {})
    tdict = cfg.get("train").to_dict() if cfg.get("train") else {}
    if "hidden" in tdict:
        tdict["hidden"] = tuple(tdict["hidden"])
    tcfg = TrainConfig(**tdict)
    out = Path(args.out)
    device = args.device
    if world > 1:
        tcfg.seed += rank
        out = out / f"rank{rank}"
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}"
        ecfg.nthread = max(1, (os.cpu_count() or 2) // world)
    if rank == args.no_curiosity_rank:
        tcfg.curiosity = 0.0
    print(f"rank {rank}/{world}: seed {tcfg.seed}, out {out}, {tcfg.num_envs} bodies, "
          f"{ecfg.nthread} simulator threads, curiosity {tcfg.curiosity}", flush=True)
    tr = Trainer(out, tcfg, ecfg, device=device, body_model_path=Path(args.body_model),
                 log=lambda s: print(s, flush=True))
    if (out / "ckpt_last.pt").is_file():
        tr.restore(out / "ckpt_last.pt")
    tr.run()
    print(f"done: {tr.it} iterations, {tr.total_steps / 1e6:.1f}M steps, stage "
          f"{ecfg.stages[tr.env.stage].name}, {tr.elapsed() / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
