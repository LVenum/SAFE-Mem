"""Raw-text memory: store what was read, summarise only when answering."""

from __future__ import annotations

import hashlib
import re

from ams.llm import EmbedRequest, LLMClient
from ams.memory.base import MemoryEntry, RetrievalResult, WriteResult
from ams.memory.index import VectorIndex

_MESSAGE_BREAK = re.compile(r"\n-{3,}\n")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_HEADER_LINE = re.compile(r"^[A-Z][A-Za-z-]*: ")


def chunk(context: str, *, max_chars: int = 1200) -> list[str]:
    """Split on message separators; oversize messages fall back to paragraphs."""
    out: list[str] = []
    for msg in _MESSAGE_BREAK.split(context):
        msg = msg.strip()
        if not msg:
            continue
        if len(msg) <= max_chars:
            out.append(msg)
            continue
        parts = [p.strip() for p in _PARAGRAPH_BREAK.split(msg) if p.strip()]
        if len(parts) > 1 and all(_HEADER_LINE.match(line) for line in parts[0].splitlines()):
            head, parts = parts[0], parts[1:]
            parts = [f"{head}\n\n{p}" for p in parts]
        out.extend(parts)
    return out


class RawMemory:
    """Verbatim chunks plus embedding retrieval. No LLM at write time."""

    def __init__(self, *, embed_client: LLMClient, embed_model: str, max_chars: int = 1200) -> None:
        self._embed = embed_client
        self._embed_model = embed_model
        self._max_chars = max_chars
        self._store: dict[str, list[MemoryEntry]] = {}
        self._vectors: dict[str, list[tuple[float, ...]]] = {}

    @property
    def name(self) -> str:
        return "raw_chunks"

    def write(
        self, *, session_id: str, user_query: str, context: str, principal: str = "user"
    ) -> WriteResult:
        chunks = chunk(context, max_chars=self._max_chars)
        if not chunks:
            return WriteResult(written=())
        entries = tuple(
            MemoryEntry(
                entry_id=f"r{hashlib.sha256(f'{session_id}|{i}|{c}'.encode()).hexdigest()[:12]}",
                text=c,
                parent_ids=(session_id,),
                metadata={"position": i, "user_query": user_query},
            )
            for i, c in enumerate(chunks)
        )
        self._store.setdefault(session_id, []).extend(entries)
        vecs = self._embed.embed(
            EmbedRequest(texts=tuple(e.text for e in entries), model=self._embed_model)
        ).vectors
        idx = self._vectors.setdefault(session_id, VectorIndex())
        idx.add(range(len(idx), len(idx) + len(vecs)), vecs)
        return WriteResult(written=entries)

    def retrieve(
        self, *, session_id: str, query: str, k: int = 4, principal: str = "user"
    ) -> RetrievalResult:
        entries = self._store.get(session_id, [])
        if not entries:
            return RetrievalResult()
        qv = self._embed.embed(EmbedRequest(texts=(query,), model=self._embed_model)).vectors[0]
        sims = self._vectors[session_id].scores(qv)
        order = sorted(range(len(entries)), key=lambda i: sims[i], reverse=True)
        return RetrievalResult(entries=tuple(entries[i] for i in order[:k]))

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._store.clear()
            self._vectors.clear()
        else:
            self._store.pop(session_id, None)
            self._vectors.pop(session_id, None)
