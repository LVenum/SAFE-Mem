"""What kind of claim would answer this query."""

from __future__ import annotations

from dataclasses import dataclass, field

from ams.core.lattice import Rung
from ams.core.prompts import PromptLibrary
from ams.llm import ChatRequest, LLMClient, Message


@dataclass
class UseTypeClassifier:
    """Query -> required rung, memoised by query text."""

    client: LLMClient
    model: str
    prompts: PromptLibrary = field(default_factory=PromptLibrary)
    version: str = "v1"
    _cache: dict[str, Rung] = field(default_factory=dict, repr=False)
    _subjects: dict[str, str] = field(default_factory=dict, repr=False)
    n_unparsed: int = 0

    def required(self, query: str) -> Rung:
        text = (query or "").strip()
        if not text:
            return Rung.DESCRIPTION
        if text in self._cache:
            return self._cache[text]
        rendered = self.prompts.get("classify_use_type", self.version).render(query=text)
        reply = self.client.complete(
            ChatRequest(
                messages=(Message("user", rendered),),
                model=self.model,
                temperature=0.0,
                max_tokens=8 if self.version == "v1" else 24,
            )
        ).text
        kind, _, who = reply.partition("|")
        try:
            rung = Rung.parse(kind)
        except ValueError:
            self.n_unparsed += 1
            rung = Rung.DESCRIPTION
        who = who.strip().strip(".\"'").lower()
        self._subjects[text] = "" if who in ("", "none", "me") else who
        self._cache[text] = rung
        return rung

    def subject(self, query: str) -> str:
        """Who the question is about (v2 only; "" for the asker or unknown). It is
        read off the principal-authored query, which is why a gate may use it."""
        text = (query or "").strip()
        if text and text not in self._subjects:
            self.required(text)
        return self._subjects.get(text, "")

    @staticmethod
    def inherit(turn_required: Rung) -> Rung:
        """The required kind for a sub-query the agent generated itself."""
        return turn_required

    def distribution(self, queries: list[str]) -> dict[str, int]:
        """Counts per rung, for the go/no-go check."""
        out: dict[str, int] = {r.name.lower(): 0 for r in Rung}
        for q in queries:
            out[self.required(q).name.lower()] += 1
        return out


@dataclass(frozen=True)
class FixedUse:
    """Every query requires the same kind."""

    rung: Rung = Rung.PERMISSION_TRUST
    n_unparsed: int = 0

    def required(self, query: str) -> Rung:
        return self.rung

    @staticmethod
    def inherit(turn_required: Rung) -> Rung:
        return turn_required
