"""Provider-agnostic model access."""

from ams.llm.base import ContentFiltered, EmbeddingUnsupported, LLMClient, ProviderError
from ams.llm.cache import NullCache, ResponseCache
from ams.llm.fallback import FallbackClient, FallbackCounts
from ams.llm.ledger import UsageLedger
from ams.llm.pool import ManagedClient, RetryPolicy, map_concurrent, progress_bar
from ams.llm.registry import ClientSpec, build_client, build_raw_client, model_of
from ams.llm.types import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    Message,
    Usage,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ClientSpec",
    "ContentFiltered",
    "EmbedRequest",
    "EmbedResponse",
    "EmbeddingUnsupported",
    "FallbackClient",
    "FallbackCounts",
    "LLMClient",
    "ManagedClient",
    "Message",
    "NullCache",
    "ProviderError",
    "ResponseCache",
    "RetryPolicy",
    "Usage",
    "UsageLedger",
    "build_client",
    "build_raw_client",
    "map_concurrent",
    "progress_bar",
    "model_of",
]
