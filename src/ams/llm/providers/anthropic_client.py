"""Anthropic adapter. The second and last place allowed to import a vendor SDK."""

from __future__ import annotations

import os
from typing import Any

from ams.llm import pricing
from ams.llm.base import EmbeddingUnsupported, ProviderError
from ams.llm.types import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, Usage

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class AnthropicClient:
    def __init__(self, *, api_key: str | None = None, timeout: float = 120.0) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ProviderError(
                "anthropic", "anthropic SDK not installed; `uv pip install -e '.[anthropic]'`"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("anthropic", "ANTHROPIC_API_KEY is not set")
        self._client = Anthropic(api_key=key, timeout=timeout)

    @property
    def provider(self) -> str:
        return "anthropic"

    def complete(self, request: ChatRequest) -> ChatResponse:
        system = "\n\n".join(m.content for m in request.messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content} for m in request.messages if m.role != "system"
        ]
        if not turns:
            raise ProviderError("anthropic", "request has no user or assistant turns")

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": turns,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if system:
            kwargs["system"] = system
        if request.stop:
            kwargs["stop_sequences"] = list(request.stop)

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise self._wrap(exc) from exc

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        inp = getattr(resp.usage, "input_tokens", 0) or 0
        out = getattr(resp.usage, "output_tokens", 0) or 0
        return ChatResponse(
            text=text,
            model=resp.model,
            provider=self.provider,
            usage=Usage(inp, out, pricing.estimate_cost(self.provider, request.model, inp, out)),
            raw_finish_reason=resp.stop_reason,
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        raise EmbeddingUnsupported(
            "Anthropic serves no embeddings endpoint; point the estimator's encoder at "
            "an OpenAI or local model via its own provider spec."
        )

    def _wrap(self, exc: Exception) -> ProviderError:
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        name = type(exc).__name__.lower()
        retryable = (
            status in _RETRYABLE_STATUS
            if status
            else any(k in name for k in ("timeout", "connection", "ratelimit", "overloaded"))
        )
        return ProviderError("anthropic", f"{type(exc).__name__}: {exc}", retryable=retryable)
