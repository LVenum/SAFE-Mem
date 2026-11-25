"""What kind of claim would answer this query."""

from __future__ import annotations

import hashlib
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


def _unit(seed: int, query: str) -> float:
    """A fixed draw in [0, 1) per (seed, query)."""
    h = hashlib.sha256(f"{seed}\x00{query.strip()}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@dataclass
class NoisyUse:
    """Rebuttal E1: shift the classifier's rung by `shift` levels on a fraction `p` of
    queries, clamped to the lattice. The perturbed set is chosen by a hash of
    (seed, query), so a smaller p perturbs a subset of what a larger p does."""

    base: object
    p: float
    shift: int
    seed: int = 0
    shifted: set[str] = field(default_factory=set, repr=False)

    @property
    def label(self) -> str:
        return f"noise={self.shift:+d}@{self.p:g}"

    def required(self, query: str) -> Rung:
        r = self.base.required(query)
        if _unit(self.seed, query) >= self.p:
            return r
        new = Rung(min(max(int(r) + self.shift, Rung.DESCRIPTION), Rung.PERMISSION_TRUST))
        if new != r:
            self.shifted.add((query or "").strip())
        return new

    def subject(self, query: str) -> str:
        return getattr(self.base, "subject", lambda q: "")(query)
