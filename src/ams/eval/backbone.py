"""Run one experiment config on a named backbone (`configs/backbones.yaml`)."""

from __future__ import annotations

from pathlib import Path

import yaml

from ams.eval.context import Config

BACKBONES = Path(__file__).resolve().parents[3] / "configs" / "backbones.yaml"


def apply_backbone(cfg: Config, name: str | None, path: Path = BACKBONES) -> None:
    if not name:
        return
    table = yaml.safe_load(path.read_text(encoding="utf-8"))
    if name not in table:
        raise ValueError(f"unknown backbone {name!r}; known: {sorted(table)}")
    b = table[name]
    models = dict(cfg.raw.get("models") or {})
    models.update(b["models"])
    cfg.raw["models"] = models
    for key in ("fallback_provider", "reasoning_effort", "fallbacks"):
        if key in b:
            cfg.raw[key] = b[key]
        else:
            cfg.raw.pop(key, None)
    if "workers" in b:
        cfg.raw.setdefault("runtime", {})["workers"] = b["workers"]
    cfg.raw["backbone"] = name
    cfg.raw["name"] = f"{cfg.raw.get('name', 'run')}_{name}"
