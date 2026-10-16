"""Extract-and-store memory: the floor every other architecture is read against."""

from __future__ import annotations

import hashlib
import json

from ams.core.prompts import PromptLibrary
from ams.llm import ChatRequest, EmbedRequest, LLMClient, Message
from ams.memory.base import MemoryEntry, RetrievalResult, WriteResult
from ams.memory.index import VectorIndex


class SimpleMemory:
    """Fact extraction plus embedding retrieval."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        embed_client: LLMClient,
        embed_model: str,
        prompts: PromptLibrary | None = None,
        prompt_version: str = "v1",
        temperature: float = 0.0,
        max_facts: int = 6,
        gate=None,
    ) -> None:
        self._client = client
        self._model = model
        self._embed = embed_client
        self._embed_model = embed_model
        self._prompts = prompts or PromptLibrary()
        self._version = prompt_version
        self._temperature = temperature
        self._max_facts = max_facts
        self._gate = gate
        self._store: dict[str, list[MemoryEntry]] = {}
        self._vectors: dict[str, list[tuple[float, ...]]] = {}

    @property
    def name(self) -> str:
        return "simple_extract"

    def write(
        self, *, session_id: str, user_query: str, context: str, principal: str = "user"
    ) -> WriteResult:
        prompt = self._prompts.get("memory_extract", self._version)
        rendered = prompt.render(query=user_query.strip(), context=context.strip())
        resp = self._client.complete(
            ChatRequest(
                messages=(Message("user", rendered),),
                model=self._model,
                temperature=self._temperature,
                max_tokens=400,
            )
        )
        facts = _parse_facts(resp.text, self._max_facts)
        discarded: tuple[str, ...] = ()
        if facts and self._gate is not None:
            verdict = self._gate.admit(user_query=user_query, context=context, facts=facts)
            facts, discarded = list(verdict.admitted), verdict.rejected
        if not facts:
            return WriteResult(written=(), discarded=discarded)

        entries = tuple(
            MemoryEntry(
                entry_id=f"e{hashlib.sha256(f'{session_id}|{f}'.encode()).hexdigest()[:12]}",
                text=f,
                parent_ids=(session_id,),
                metadata={"source_context": context, "user_query": user_query},
            )
            for f in facts
        )
        self._store.setdefault(session_id, []).extend(entries)
        vecs = self._embed.embed(
            EmbedRequest(texts=tuple(e.text for e in entries), model=self._embed_model)
        ).vectors
        idx = self._vectors.setdefault(session_id, VectorIndex())
        idx.add(range(len(idx), len(idx) + len(vecs)), vecs)
        return WriteResult(written=entries, discarded=discarded)

    def prefetch(self, *, user_query: str, context: str) -> None:
        """The extraction call `write` will make, without storing (cache warm-up).
        A write gate, if any, still runs inside the sequential write."""
        prompt = self._prompts.get("memory_extract", self._version)
        resp = self._client.complete(
            ChatRequest(
                messages=(
                    Message(
                        "user", prompt.render(query=user_query.strip(), context=context.strip())
                    ),
                ),
                model=self._model,
                temperature=self._temperature,
                max_tokens=400,
            )
        )
        facts = _parse_facts(resp.text, self._max_facts)
        if facts and self._gate is None:
            self._embed.embed(EmbedRequest(texts=tuple(facts), model=self._embed_model))

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


def _parse_facts(text: str, limit: int) -> list[str]:
    """One fact per line, bullets and numbering stripped."""
    out: list[str] = []
    try:
        facts = json.loads(text.strip().removeprefix("```json").removesuffix("```")).get(
            "facts", []
        )
        return [str(f).strip() for f in facts if len(str(f).strip()) >= 8][:limit]
    except (json.JSONDecodeError, AttributeError):
        pass
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•").strip()
        line = line[2:].strip() if line[:2].rstrip(".").isdigit() else line
        if not line or len(line) < 8:
            continue
        if line.lower().startswith(("none", "no durable", "no facts", "n/a")):
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return out
