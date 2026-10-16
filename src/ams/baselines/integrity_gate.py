"""Binary integrity gating, Fides-style — ours, and never to be called Fides."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ams.baselines.base import DefenceResult
from ams.memory.base import MemoryEntry


def untrusted(entry: MemoryEntry) -> bool:
    md = entry.metadata or {}
    if "source_context" not in md:
        return True
    return bool(str(md["source_context"]).strip())


@dataclass
class IntegrityGate:
    consequential: object = None

    @property
    def name(self) -> str:
        return f"integrity_gate[{'all' if self.consequential is None else 'sensitive'}]"

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        items = list(memories)
        tainted = [m for m in items if untrusted(m)]
        if tainted and (self.consequential is None or self.consequential(query)):
            return DefenceResult(
                kept=(),
                flagged=tuple(tainted),
                reason="blocked: consequential request over untrusted memory",
                detail={"verdict": "block", "n_untrusted": len(tainted)},
            )
        return DefenceResult(kept=tuple(items), reason="allowed", detail={"verdict": "allow"})
