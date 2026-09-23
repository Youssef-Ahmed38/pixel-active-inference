"""Tiny YAML config system: nested dicts with attribute access and CLI overrides.

    cfg = load_config("configs/default.yaml", ["agent.k_a=2.0", "env.image_size=128"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict with attribute access; nested dicts are converted recursively."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        return {k: _unwrap(v) for k, v in self.items()}


def _wrap(v: Any) -> Any:
    """Nested dicts, including dicts inside lists, become Config (e.g. a list of cameras)."""
    if isinstance(v, dict) and not isinstance(v, Config):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _unwrap(v: Any) -> Any:
    if isinstance(v, Config):
        return v.to_dict()
    if isinstance(v, list):
        return [_unwrap(x) for x in v]
    return copy.deepcopy(v)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_overrides(data: dict, overrides: Iterable[str]) -> dict:
    """Apply `a.b.c=value` overrides; values are parsed as YAML (so 1e-3, true, [1,2] work)."""
    data = copy.deepcopy(data)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        node = data
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return data


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config. A top-level `base:` key names another YAML file to inherit from."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if "base" in data:
        base = load_config(path.parent / data.pop("base")).to_dict()
        data = _merge(base, data)
    return Config(apply_overrides(data, overrides))
