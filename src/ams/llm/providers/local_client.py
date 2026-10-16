"""Local sentence-transformer embeddings. No API, no key, no per-token cost."""

from __future__ import annotations

import os
import threading

from ams.llm.base import ProviderError
from ams.llm.types import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, Usage

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbeddingClient:
    """Embeddings via sentence-transformers, loaded once and reused."""

    def __init__(self, *, model: str = DEFAULT_MODEL, device: str | None = None) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ProviderError(
                "local",
                "sentence-transformers not installed; "
                "`uv pip install -e '.[local]'` (or use an API embedding model)",
            ) from exc
        self._model_id = model
        self._model = SentenceTransformer(model, device=device)
        self._lock = threading.Lock()

    @property
    def provider(self) -> str:
        return "local"

    def complete(self, request: ChatRequest) -> ChatResponse:
        raise ProviderError(
            "local",
            "the local provider serves embeddings only; point chat roles at "
            "azure:, openai:, anthropic:, or vllm:",
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        with self._lock:
            vectors = self._model.encode(
                list(request.texts), normalize_embeddings=True, show_progress_bar=False
            )
        return EmbedResponse(
            vectors=tuple(tuple(float(x) for x in v) for v in vectors),
            model=self._model_id,
            provider=self.provider,
            usage=Usage(sum(len(t) // 4 for t in request.texts), 0, 0.0),
        )
