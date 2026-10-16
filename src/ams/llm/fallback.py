"""Run on a preferred backend; fall back when it refuses, and say how often."""

from __future__ import annotations

from dataclasses import dataclass, field

from ams.llm.base import ContentFiltered, LLMClient, ProviderError
from ams.llm.types import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, Usage


@dataclass
class FallbackCounts:
    """Who served what. Belongs in the run manifest, not in a log line."""

    primary: int = 0
    fallback: int = 0
    refused_by_primary: int = 0
    refused_by_both: int = 0
    failed_both: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.primary + self.fallback

    @property
    def fallback_share(self) -> float:
        return self.fallback / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "fallback": self.fallback,
            "fallback_share": round(self.fallback_share, 4),
            "refused_by_primary": self.refused_by_primary,
            "refused_by_both": self.refused_by_both,
            "failed_both": self.failed_both,
            "reasons": dict(sorted(self.reasons.items())),
        }


class FallbackClient:
    """Two backends, one `LLMClient`."""

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self._primary = primary
        self._fallback = fallback
        self.counts = FallbackCounts()

    @property
    def provider(self) -> str:
        return f"{self._primary.provider}|{self._fallback.provider}"

    def _note(self, exc: Exception) -> None:
        key = f"{type(exc).__name__}:{getattr(exc, 'provider', '?')}"
        self.counts.reasons[key] = self.counts.reasons.get(key, 0) + 1

    def _run(self, call, request, refusal: ChatResponse | None = None):
        try:
            out = call(self._primary, request)
            self.counts.primary += 1
            return out
        except ContentFiltered as exc:
            self.counts.refused_by_primary += 1
            self._note(exc)
            first = exc
        except ProviderError as exc:
            if exc.retryable:
                raise
            self._note(exc)
            first = exc

        try:
            out = call(self._fallback, request)
            self.counts.fallback += 1
            return out
        except ContentFiltered as second:
            if isinstance(first, ContentFiltered) and refusal is not None:
                self.counts.refused_by_both += 1
                self._note(second)
                return refusal
            self.counts.failed_both += 1
            self._note(second)
            raise second from first
        except Exception as second:
            self.counts.failed_both += 1
            self._note(second)
            raise second from first

    def complete(self, request: ChatRequest) -> ChatResponse:
        refusal = ChatResponse(
            text="",
            model=request.model,
            provider=self.provider,
            usage=Usage(),
            raw_finish_reason="content_filter",
        )
        return self._run(lambda c, r: c.complete(r), request, refusal)

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        return self._run(lambda c, r: c.embed(r), request, None)
