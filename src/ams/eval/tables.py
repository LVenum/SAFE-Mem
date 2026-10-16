"""CSV output from dataclass rows."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any


def write_csv(path: str | Path, rows: Sequence[Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    if not is_dataclass(rows[0]):
        raise TypeError("write_csv expects dataclass rows")
    names = [f.name for f in fields(rows[0])]
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({n: _cell(getattr(row, n)) for n in names})
    return p


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        return value.replace("\n", " ").strip()
    return value
