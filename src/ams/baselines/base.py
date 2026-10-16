"""The defence seam."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ams.memory.base import MemoryEntry


@dataclass(frozen=True, slots=True)
class DefenceResult:
    """Which memories survived, and why."""

    kept: tuple[MemoryEntry, ...]
    flagged: tuple[MemoryEntry, ...] = ()
    reason: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def kept_ids(self) -> frozenset[str]:
        return frozenset(m.mid for m in self.kept)

    @property
    def flagged_ids(self) -> frozenset[str]:
        return frozenset(m.mid for m in self.flagged)


@runtime_checkable
class Defence(Protocol):
    @property
    def name(self) -> str: ...

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult: ...


class NoDefence:
    """The control. Keeps everything."""

    @property
    def name(self) -> str:
        return "no_defence"

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        return DefenceResult(kept=tuple(memories), reason="no filtering")
