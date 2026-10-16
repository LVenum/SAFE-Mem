"""The seam every other module depends on."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ams.llm.types import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse


@runtime_checkable
class LLMClient(Protocol):
    """A synchronous chat/embedding backend."""

    @property
    def provider(self) -> str:
        """Short provider id used in cache keys and the usage ledger."""
        ...

    def complete(self, request: ChatRequest) -> ChatResponse: ...

    def embed(self, request: EmbedRequest) -> EmbedResponse: ...


class EmbeddingUnsupported(RuntimeError):
    """Raised by chat-only backends when asked to embed."""


class ContentFiltered(RuntimeError):
    """The provider refused the request on content-policy grounds."""

    def __init__(self, provider: str, message: str, categories: dict | None = None) -> None:
        super().__init__(f"[{provider}] content filtered: {message}")
        self.provider = provider
        self.categories = categories or {}
        self.retryable = False


class ProviderError(RuntimeError):
    """Provider call failed after the pool exhausted its retries."""

    def __init__(self, provider: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.retryable = retryable
