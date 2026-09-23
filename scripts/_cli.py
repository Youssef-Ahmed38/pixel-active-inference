"""Shared CLI: every script takes --config plus `key.sub=value` overrides."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pai.config import load_config  # noqa: E402


def parse(description: str, extra=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*", help="config overrides, e.g. train.batch_size=128")
    if extra:
        extra(ap)
    args = ap.parse_args()
    return args, load_config(args.config, args.overrides)
