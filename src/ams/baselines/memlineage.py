"""MemLineage, implemented by us from its paper (arXiv 2605.14421, Ouyang & Hou).

What the paper specifies, and this file does as written
-------------------------------------------------------
* Four labels, encoded so larger is less safe: Trusted 0, Derived-Trusted 1,
  Derived-Untrusted 2, External 3 (§3.2).
* Parents of an entry are everything in the LLM context when it was written
  (§3.5, §4.4). Each edge gets an attribution weight w(p, c) in [0, 1].
* **Equation (1):** ``trust(c) = max_p 1[w(p, c) > tau] * trust(p)``, strict
  inequality. **When no parent exceeds tau the result is Trusted** — the
  "no-strong-parent fallback" the paper itself names as defeatable ("an attacker
  who can suppress every attribution edge below tau ... can defeat propagation
  locally"). This is reproduced faithfully and NOT fixed: it is the other
  design's own stated weakness, and the reason our meet runs over the whole
  context with no threshold (spec §5.3). `propagate` is where it lives.
* Two attribution algorithms: **Coarse** (w = 1 for every parent in the context;
  the paper's headline cell, default tau = 0) and **LmSelfEval** (an LLM judge
  scores each parent; we run it at tau = 0.3, the paper's adaptive-test value).
  Fail-open is only reachable under LmSelfEval: Coarse never emits w <= 0.
* Enforcement at the sensitive-action gate with verdict **Deny** — the only
  verdict the paper's ASR results use. Ordinary recall is never gated; untrusted
  entries stay visible there ("derived-untrusted chains are still usable for
  non-sensitive recall", §6.5).
* Sentinel rendering, ``[BEGIN MEMORY entry_id=.. trust=..] .. [END MEMORY]``
  (§3.6), via `ams.core.sentinel`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum

from ams.baselines.base import DefenceResult
from ams.core import sentinel
from ams.core.prompts import PromptLibrary
from ams.llm import ChatRequest, LLMClient, Message
from ams.memory.base import MemoryEntry
from ams.memory.typed import split_artifacts

MAX_CONTENT_CHARS = 2000
MAX_CANDIDATES = 20


class Trust(IntEnum):
    TRUSTED = 0
    DERIVED_TRUSTED = 1
    DERIVED_UNTRUSTED = 2
    EXTERNAL = 3


def propagate(parents: Sequence[tuple[Trust, float]], tau: float) -> Trust:
    """Equation (1), with the paper's no-strong-parent fallback to Trusted."""
    strong = [t for t, w in parents if w > tau]
    return max(strong) if strong else Trust.TRUSTED


def parents_of(entry: MemoryEntry) -> list[tuple[str, Trust]]:
    """The write-time context of an entry as (content, initial label)."""
    md = entry.metadata or {}
    out = []
    if str(md.get("user_query", "")).strip():
        out.append((str(md["user_query"]), Trust.TRUSTED))
    out += [(a, Trust.EXTERNAL) for a in split_artifacts(str(md.get("source_context", "")))]
    return out


@dataclass
class SensitiveAction:
    """Is this request a sensitive action? Cached by query text via the client."""

    client: LLMClient
    model: str
    prompts: PromptLibrary = field(default_factory=PromptLibrary)
    version: str = "v1"

    def __call__(self, query: str) -> bool:
        reply = self.client.complete(
            ChatRequest(
                messages=(
                    Message(
                        "user",
                        self.prompts.get("sensitive_action", self.version).render(query=query),
                    ),
                ),
                model=self.model,
                temperature=0.0,
                max_tokens=4,
            )
        ).text
        return "SENSITIVE" in reply.upper()


@dataclass
class LmSelfEval:
    """The LLM attribution judge."""

    client: LLMClient
    model: str
    prompts: PromptLibrary = field(default_factory=PromptLibrary)
    version: str = "v1"

    def __call__(self, derived: str, candidates: Sequence[str]) -> list[float]:
        cands = list(candidates)[:MAX_CANDIDATES]
        envelope = json.dumps(
            {
                "derived_entry": derived[:MAX_CONTENT_CHARS],
                "candidates": [
                    {"id": f"p{i}", "content": c[:MAX_CONTENT_CHARS]} for i, c in enumerate(cands)
                ],
            },
            ensure_ascii=False,
        )
        reply = self.client.complete(
            ChatRequest(
                messages=(
                    Message(
                        "system", self.prompts.get("memlineage_attribution", self.version).template
                    ),
                    Message("user", envelope),
                ),
                model=self.model,
                temperature=0.0,
                max_tokens=200,
            )
        ).text
        ws = parse_weights(reply, len(cands))
        return ws + [0.0] * (len(candidates) - len(cands))


def parse_weights(reply: str, n: int) -> list[float]:
    """Weights in [0, 1]; anything unreadable is 0 for every parent (choice 3)."""
    body = reply.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        ws = json.loads(body)["weights"]
        out = [min(1.0, max(0.0, float(w))) for w in ws]
    except (ValueError, KeyError, TypeError):
        return [0.0] * n
    return out if len(out) == n else [0.0] * n


def coarse(derived: str, candidates: Sequence[str]) -> list[float]:
    """Coarse attribution: every parent in the context gets weight 1."""
    return [1.0] * len(candidates)


@dataclass
class MemLineage:
    attribution: object
    sensitive: SensitiveAction
    tau: float = 0.0
    label: str = "coarse"
    _trust: dict[str, Trust] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return f"memlineage[{self.label},tau={self.tau:g}]"

    def trust(self, entry: MemoryEntry) -> Trust:
        """Computed once per entry from its write-time context, never from the query."""
        if entry.entry_id not in self._trust:
            parents = parents_of(entry)
            weights = self.attribution(entry.text, [p for p, _ in parents]) if parents else []
            self._trust[entry.entry_id] = propagate(
                [(t, w) for (_, t), w in zip(parents, weights, strict=True)], self.tau
            )
        return self._trust[entry.entry_id]

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        items = list(memories)
        labels = [self.trust(m) for m in items]
        sens = bool(items) and self.sensitive(query)
        verdicts = [{"index": i, "trust": t.name, "sensitive": sens} for i, t in enumerate(labels)]
        if sens and any(t >= Trust.DERIVED_UNTRUSTED for t in labels):
            for v in verdicts:
                v["kept"] = False
            return DefenceResult(
                kept=(),
                flagged=tuple(m for m, t in zip(items, labels, strict=True) if t >= 2),
                reason="Deny: sensitive action with an untrusted ancestor",
                detail={"verdicts": verdicts, "verdict": "deny"},
            )
        for v in verdicts:
            v["kept"] = True
        return DefenceResult(
            kept=tuple(
                replace(m, text=sentinel.render(m.entry_id, m.text, trust=t.name))
                for m, t in zip(items, labels, strict=True)
            ),
            reason="Allow",
            detail={"verdicts": verdicts, "verdict": "allow"},
        )
