"""The memory-system seam."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """One stored item, with whatever the store knows about where it came from."""

    entry_id: str
    text: str
    parent_ids: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WriteResult:
    written: tuple[MemoryEntry, ...]
    discarded: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult(Sequence):
    """What a store surfaced, and what it decided on the way."""

    entries: tuple[MemoryEntry, ...] = ()
    required: str = ""
    eligible_empty: bool = False
    mode: str = ""
    verdicts: dict = field(default_factory=dict)

    def __getitem__(self, i):
        return self.entries[i]

    def __len__(self) -> int:
        return len(self.entries)


@runtime_checkable
class MemorySystem(Protocol):
    @property
    def name(self) -> str: ...

    def write(
        self, *, session_id: str, user_query: str, context: str, principal: str = "user"
    ) -> WriteResult:
        """Process one interaction and store whatever this system would store."""
        ...

    def retrieve(
        self, *, session_id: str, query: str, k: int = 4, principal: str = "user"
    ) -> RetrievalResult:
        """Return the entries this system would surface for `query`."""
        ...

    def reset(self, session_id: str | None = None) -> None:
        """Clear state. Cases must not leak into one another."""
        ...
