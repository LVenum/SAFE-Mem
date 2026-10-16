"""Per-run token and cost accounting."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path

from ams.llm.types import Usage


class UsageLedger:
    """Thread-safe accumulator. Optionally appends one JSON line per call."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._by_stage: dict[str, Usage] = defaultdict(Usage)
        self._calls: dict[str, int] = defaultdict(int)
        self._cached: dict[str, int] = defaultdict(int)
        self._unpriced: set[str] = set()
        self._path = Path(path) if path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")
        else:
            self._fh = None

    def record(
        self,
        *,
        provider: str,
        model: str,
        usage: Usage,
        stage: str = "unattributed",
        cached: bool = False,
        priced: bool = True,
    ) -> None:
        with self._lock:
            self._calls[stage] += 1
            if cached:
                self._cached[stage] += 1
            else:
                self._by_stage[stage] = self._by_stage[stage] + usage
            if not priced:
                self._unpriced.add(f"{provider}:{model}")
            if self._fh is not None:
                self._fh.write(
                    json.dumps(
                        {
                            "stage": stage,
                            "provider": provider,
                            "model": model,
                            "cached": cached,
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cost_usd": usage.cost_usd,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self._fh.flush()

    def total(self) -> Usage:
        with self._lock:
            out = Usage()
            for usage in self._by_stage.values():
                out = out + usage
            return out

    def summary(self) -> dict:
        with self._lock:
            return {
                "stages": {
                    stage: {
                        "calls": self._calls[stage],
                        "cache_hits": self._cached[stage],
                        "api_calls": self._calls[stage] - self._cached[stage],
                        "input_tokens": u.input_tokens,
                        "output_tokens": u.output_tokens,
                        "cost_usd": round(u.cost_usd, 6),
                    }
                    for stage, u in sorted(self._by_stage.items())
                },
                "total_calls": sum(self._calls.values()),
                "total_cache_hits": sum(self._cached.values()),
                "total_cost_usd": round(sum(u.cost_usd for u in self._by_stage.values()), 6),
                "unpriced_models": sorted(self._unpriced),
            }

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
