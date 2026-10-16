"""Minimal `.env` loading."""

from __future__ import annotations

import os
from pathlib import Path

_loaded: set[Path] = set()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_env(path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read `.env` into `os.environ`. Idempotent; returns what it set."""
    target = Path(path) if path else repo_root() / ".env"
    if target in _loaded and not override:
        return {}
    _loaded.add(target)
    if not target.exists():
        return {}

    applied: dict[str, str] = {}
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied
