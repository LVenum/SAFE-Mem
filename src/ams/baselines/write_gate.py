"""Admission control at write time: does a candidate entry get into memory at all?"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ams.core.prompts import PromptLibrary
from ams.llm import ChatRequest, LLMClient, Message


@dataclass(frozen=True, slots=True)
class GateResult:
    """Which candidate entries may be stored, and why."""

    admitted: tuple[str, ...]
    rejected: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


@runtime_checkable
class WriteGate(Protocol):
    @property
    def name(self) -> str: ...

    def admit(self, *, user_query: str, context: str, facts: Sequence[str]) -> GateResult: ...


class AdmitAll:
    """The control. Stores everything, which is what a memory system does by default."""

    @property
    def name(self) -> str:
        return "admit_all"

    def admit(self, *, user_query: str, context: str, facts: Sequence[str]) -> GateResult:
        return GateResult(admitted=tuple(facts))


_NUM = re.compile(r"(?:0?\.\d+|[01](?:\.0+)?)")


class ConsistencyGate:
    """Self-consistency admission control, per arXiv 2607.22962."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        k: int = 5,
        tau: float = 0.7,
        temperature: float = 0.7,
        prompts: PromptLibrary | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self._client = client
        self._model = model
        self._k = k
        self._tau = tau
        self._temperature = temperature
        self._prompts = prompts or PromptLibrary()
        self._version = prompt_version
        self.n_unparsed = 0

    @property
    def name(self) -> str:
        return f"consistency_gate[k={self._k},tau={self._tau}]"

    def _support(self, context: str, fact: str) -> float:
        prompt = self._prompts.get("write_gate_consistency", self._version)
        rendered = prompt.render(context=context.strip(), fact=fact.strip())
        total = 0.0
        for i in range(self._k):
            text = self._client.complete(
                ChatRequest(
                    messages=(Message("user", rendered),),
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=8,
                    sample_index=i,
                )
            ).text
            m = _NUM.search(text)
            if m is None:
                self.n_unparsed += 1
                total += 0.5
            else:
                total += min(1.0, max(0.0, float(m.group(0))))
        return total / self._k

    def admit(self, *, user_query: str, context: str, facts: Sequence[str]) -> GateResult:
        admitted, rejected, scores = [], [], {}
        for f in facts:
            s = self._support(context, f)
            scores[f] = round(s, 3)
            (admitted if s >= self._tau else rejected).append(f)
        return GateResult(
            admitted=tuple(admitted),
            rejected=tuple(rejected),
            scores=scores,
            detail={"k": self._k, "tau": self._tau, "n_unparsed": self.n_unparsed},
        )
