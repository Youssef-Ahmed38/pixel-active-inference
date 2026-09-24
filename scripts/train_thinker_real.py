"""Train the HRM thinker and a same-size transformer to imitate the Bayesian teacher on REAL episodes.

    python scripts/train_thinker_real.py results/slice_evidence.npz                 # 3 seeds, both models
    python scripts/train_thinker_real.py a_evidence.npz b_evidence.npz --tag ab train.steps=500 seeds=5
    python scripts/train_thinker_real.py --fake 60 train.steps=100 seeds=1        # smoke run on a fake file

Distillation: KL(teacher || thinker) + ce_weight * CE(true cause), deep supervision and learned
halting for the HRM (pai/thinker/train.py, unchanged). Both models get the same optimizer steps and
batch size (same examples seen); the HRM spends more compute per step. Each seed draws its own
episode split (`split_per_seed`) and initialisation, and both models share it, so the comparison
is paired and the spread over seeds includes split variance (small data: that variance is real).
Weights are selected by validation KL to the teacher; the test split is scored once.

Week-4 gate: HRM agrees with the teacher (argmax) on > 90% of test episodes.
Writes <out_dir>/real_<tag>.json, real_<tag>.md, real_<tag>_<model>_s<seed>.jsonl and checkpoints.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai.config import REPO_ROOT, Config, apply_overrides  # noqa: E402
from pai.thinker.evaluate import ece, run_think, score  # noqa: E402
from pai.thinker.real import RealConfig, RealEvidence, confusion, write_fake_evidence  # noqa: E402
from pai.thinker.train import build_model, n_params, train_thinker  # noqa: E402
from pai.train.common import Progress, save_checkpoint  # noqa: E402

DEFAULTS = {
    "seed": 0,
    "seeds": 3,
    "out_dir": "results/thinker",
    "models": ["hrm", "transformer"],
    "gate": 0.9,
    "device": "auto",  # auto (CUDA if available) | cpu | cuda
    "data": {"window": 20, "after": 5, "jitter": 3, "n_phase": 10, "val_frac": 0.2, "test_frac": 0.2,
             "split_per_seed": True},
    # Small models for small data; transformer = the HRM's H + L blocks stacked once (matched params).
    "model": {"hrm": {"d_model": 64, "n_heads": 4, "h_layers": 2, "l_layers": 2, "h_cycles": 2, "l_steps": 2,
                      "max_segments": 4, "patch": 1},
              "transformer": {"d_model": 64, "n_heads": 4, "n_layers": 4, "patch": 1},
              "mlp": {"hidden": 128, "n_layers": 3}},
    "train": {"batch_size": 64, "steps": 2000, "lr": 1.0e-3, "warmup_steps": 100, "weight_decay": 0.1,
              "ce_weight": 0.1, "q_weight": 0.5, "ponder_cost": 0.01, "q_target": "oracle", "amp": "auto",
              "log_every": 100, "select": "best_val"},  # best_val | last
    "eval": {"batch": 1024},
}


class ValSelect:
    """Duck-typed logger for `train_thinker`: at every log step it scores the validation split,
    keeps the weights with the lowest validation KL to the teacher, appends a jsonl row and prints
    a progress line with ETA."""

    def __init__(self, model, val: dict, device, log_path: Path, progress: Progress, batch: int):
        self.model, self.val, self.device, self.path, self.progress, self.batch = model, val, device, log_path, progress, batch
        self.best_kl, self.best_step, self.best_state, self.last_step = float("inf"), 0, None, 0
        self.t0 = time.time()

    def log(self, step: int, **row) -> None:
        self.model.eval()
        probs, segs, _ = run_think(self.model, self.val, self.batch, self.device)
        self.model.train()
        s = score(probs, self.val)
        if s["kl_to_teacher"] < self.best_kl:
            self.best_kl, self.best_step = s["kl_to_teacher"], step
            self.best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        row = {"time": round(time.time() - self.t0, 2), "step": step, **row,
               **{f"val_{k}": v for k, v in s.items()}, "val_segments": segs.float().mean().item()}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.progress.update(step - self.last_step, extra=f"step {step} loss {row['loss']:.3f} "
                             f"val agree {s['teacher_agree']:.2f} KL {s['kl_to_teacher']:.3f}")
        self.last_step = step


def evaluate(model, data: RealEvidence, split: str, device, batch: int) -> dict:
    d = data.split(split)
    model.eval()
    probs, segs, _ = run_think(model, d, batch, device)
    pred = probs.argmax(-1).cpu().numpy()
    labels = data.labels[d["episode"].cpu().numpy()]
    return {**score(probs, d), "n": len(pred), "mean_segments": segs.float().mean().item(),
            "confusion": confusion(labels, pred, data.causes),
            "confusion_vs_teacher": confusion(np.array(data.causes)[d["teacher"].argmax(-1).cpu().numpy()],
                                              pred, data.causes)}


def teacher_reference(data: RealEvidence, split: str) -> dict:
    d = data.split(split)
    t = d["teacher"]
    labels = data.labels[d["episode"].cpu().numpy()]
    return {"acc": (t.argmax(-1) == d["y"]).float().mean().item(), "ece": ece(t, d["y"]), "n": len(labels),
            "confusion": confusion(labels, t.argmax(-1).cpu().numpy(), data.causes)}


def _agg(runs: list[dict], key: str) -> dict:
    v = np.array([r[key] for r in runs], dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std()), "values": v.tolist()}


def _sum_conf(confs: list[dict]) -> dict:
    out: dict = {}
    for c in confs:
        for lab, row in c.items():
            for k, n in row.items():
                out.setdefault(lab, {}).setdefault(k, 0)
                out[lab][k] += n
    return out


def _pct(a: dict) -> str:
    return f"{100 * a['mean']:.1f} ± {100 * a['std']:.1f}%"


def _conf_table(conf: dict, causes) -> list[str]:
    lines = ["| true \\ predicted | " + " | ".join(causes) + " |", "|---|" + "---|" * len(causes)]
    for lab, row in conf.items():
        lines.append(f"| {lab} | " + " | ".join(str(row[c]) for c in causes) + " |")
    return lines


def write_markdown(res: dict, path: Path) -> None:
    d, causes = res["data"], res["data"]["causes"]
    t = res["teacher"]
    lines = [f"# Thinker on real episodes: {res['tag']}", "",
             f"{d['episodes']} episodes from {', '.join(f'`{Path(p).name}`' for p in d['files'])}; labels "
             f"{d['label_counts']}. Window {d['window']} steps ({d['after']} after the surprise peak), "
             f"train jitter ±{d['jitter']}. {res['n_seeds']} seeds (each: own split "
             f"{d['split_sizes']} and init), {res['config']['train']['steps']} steps x batch "
             f"{res['config']['train']['batch_size']} per model. Mean ± std over seeds, test split. "
             f"Generated by `scripts/train_thinker_real.py`.", "",
             "| Model | Params | Train time / seed | Agree w/ teacher | Acc (true cause) | KL to teacher | ECE | Mean segments |",
             "|---|---|---|---|---|---|---|---|",
             f"| Bayesian teacher | - | - | 100% | {_pct(t['acc'])} | 0 | {t['ece']['mean']:.3f} | - |"]
    for name, m in res["models"].items():
        a = m["test"]
        lines.append(f"| {name} | {m['params'] / 1e3:.1f}k | {m['train_wall_s']['mean']:.0f} s | {_pct(a['teacher_agree'])} | "
                     f"{_pct(a['acc'])} | {a['kl_to_teacher']['mean']:.4f} | {a['ece']['mean']:.3f} | "
                     f"{a['mean_segments']['mean']:.2f} |")
    if "hrm" in res["models"]:
        g = res["models"]["hrm"]["test"]["teacher_agree"]
        lines += ["", f"**Week-4 gate** (HRM agreement > {100 * res['config']['gate']:.0f}%): "
                  f"{'PASS' if g['mean'] > res['config']['gate'] else 'FAIL'} ({_pct(g)}; worst seed "
                  f"{100 * min(g['values']):.1f}%)."]
    lines += ["", "## Confusion on the test split (summed over seeds)", "", "**Teacher** (argmax)", ""]
    lines += _conf_table(t["confusion"], causes)
    for name, m in res["models"].items():
        lines += ["", f"**{name}** vs true label", ""] + _conf_table(m["confusion"], causes)
        lines += ["", f"**{name}** vs teacher argmax", ""] + _conf_table(m["confusion_vs_teacher"], causes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("args", nargs="*", help="evidence .npz files and key=value overrides (e.g. train.steps=500)")
    ap.add_argument("--tag", default=None, help="output name real_<tag>; default: first file's name")
    ap.add_argument("--fake", type=int, default=0, help="write and use a fake evidence file with N episodes")
    a = ap.parse_intermixed_args()
    files = [x for x in a.args if "=" not in x]
    cfg = Config(apply_overrides(DEFAULTS, [x for x in a.args if "=" in x]))
    out = Path(cfg.out_dir) if Path(cfg.out_dir).is_absolute() else REPO_ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if a.fake:
        files.append(str(write_fake_evidence(out / "fake_evidence.npz", a.fake, seed=cfg.seed)))
    if not files:
        ap.error("give at least one evidence file (or --fake N)")
    tag = a.tag or Path(files[0]).name.removesuffix(".npz").removesuffix("_evidence")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
    data = RealEvidence(files, RealConfig.from_dict(cfg.data.to_dict()), seed=cfg.seed, device=device)
    print(f"== {data.n_episodes} episodes, labels {data.summary()['label_counts']}, in_dim {data.n_channels}, "
          f"device {device}")

    tcfg, ecfg = cfg.train.to_dict(), cfg.eval.to_dict()
    seeds = [cfg.seed + i for i in range(cfg.seeds)]
    progress = Progress(len(seeds) * len(cfg.models) * tcfg["steps"], "train", every=1, unit=" steps")
    runs: dict[str, list[dict]] = {k: [] for k in cfg.models}
    teacher_runs, params, splits = [], {}, {}
    t_all = time.time()
    for s in seeds:
        data.set_split(s if cfg.data.split_per_seed else cfg.seed)
        splits[s] = data.summary()["split_sizes"]
        teacher_runs.append(teacher_reference(data, "test"))
        val = data.split("val")
        for kind in cfg.models:
            torch.manual_seed(s)
            model = build_model(kind, data, cfg.model.to_dict())
            params[kind] = n_params(model)
            log_path = out / f"real_{tag}_{kind}_s{s}.jsonl"
            log_path.unlink(missing_ok=True)
            sel = ValSelect(model, val, device, log_path, progress, ecfg["batch"])
            print(f"-- seed {s}, {kind}: {params[kind] / 1e3:.1f}k params")
            info = train_thinker(model, data, tcfg, device, sel, seed=s)
            if tcfg["select"] == "best_val" and sel.best_state is not None:
                model.load_state_dict(sel.best_state)
            save_checkpoint(out / "checkpoints" / f"real_{tag}_{kind}_s{s}.pt",
                            {"kind": kind, "model": model.state_dict(), "config": cfg.to_dict(), "causes": data.causes,
                             "in_dim": data.n_channels, "split_seed": data.split_seed})
            ev = evaluate(model, data, "test", device, ecfg["batch"])
            ev.update(seed=s, best_step=sel.best_step, train=info, val=evaluate(model, data, "val", device, ecfg["batch"]))
            runs[kind].append(ev)
            print(f"   test: agree {ev['teacher_agree']:.3f} acc {ev['acc']:.3f} KL {ev['kl_to_teacher']:.4f} "
                  f"ECE {ev['ece']:.3f} segs {ev['mean_segments']:.2f} (best val step {sel.best_step})", flush=True)

    keys = ["teacher_agree", "acc", "kl_to_teacher", "ece", "mean_segments"]
    res = {"tag": tag, "config": cfg.to_dict(), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "n_seeds": len(seeds), "data": {**data.summary(), "split_sizes": splits[seeds[0]], "splits_per_seed": splits},
           "teacher": {"acc": _agg(teacher_runs, "acc"), "ece": _agg(teacher_runs, "ece"),
                       "confusion": _sum_conf([r["confusion"] for r in teacher_runs])},
           "models": {k: {"params": params[k],
                          "train_wall_s": _agg([r["train"] for r in rs], "wall_s"),
                          "test": {key: _agg(rs, key) for key in keys},
                          "val": {key: _agg([r["val"] for r in rs], key) for key in keys},
                          "confusion": _sum_conf([r["confusion"] for r in rs]),
                          "confusion_vs_teacher": _sum_conf([r["confusion_vs_teacher"] for r in rs]),
                          "runs": rs} for k, rs in runs.items()},
           "total_wall_s": round(time.time() - t_all, 1)}
    if "hrm" in res["models"]:
        res["gate_pass"] = res["models"]["hrm"]["test"]["teacher_agree"]["mean"] > cfg.gate
    (out / f"real_{tag}.json").write_text(json.dumps(res, indent=2))
    write_markdown(res, out / f"real_{tag}.md")
    print(f"wrote {out / f'real_{tag}.json'} and real_{tag}.md ({res['total_wall_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
