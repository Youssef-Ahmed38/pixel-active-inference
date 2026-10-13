"""Replicate paired observation/scratch practice with separate saved policies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer-seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    p.add_argument("--updates", type=int, default=24)
    p.add_argument("--output", type=Path, default=ROOT / "results/bike_replicates.json")
    args = p.parse_args()
    reports = []
    for seed in args.optimizer_seeds:
        output = ROOT / f"results/bike_practice_seed{seed}.json"
        command = [sys.executable, str(ROOT / "scripts/bike_practice.py"),
                   "--checkpoint", str(ROOT / "runs/bike/diverse/observed_policy.json"),
                   "--updates", str(args.updates), "--seed", str(seed),
                   "--practice-seeds", *map(str, range(100, 110)),
                   "--eval-seeds", *map(str, range(140, 160)),
                   "--artifacts", str(ROOT / f"runs/bike/practice_seed{seed}"), "--output", str(output)]
        subprocess.run(command, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        report = json.loads(output.read_text())
        reports.append({"seed": seed, "report": str(output), "conditions": {
            name: {key: row[key] for key in ("first_all_successful_practice_attempt", "baseline_successes", "practiced_successes")}
            for name, row in report["conditions"].items()}})
        print(json.dumps(reports[-1], indent=2), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"replicates": reports, "layouts_per_replicate": 20,
            "limitations": ["replicates share the same evaluation layouts and observed-motion initialization",
                            "not 60 independent physical layouts", "no Stage 2 transfer comparison"]}, indent=2) + "\n", encoding="utf-8")
