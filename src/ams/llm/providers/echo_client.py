"""Deterministic fake backend. Zero cost, no network, no API key."""

from __future__ import annotations

import hashlib
import re

from ams.llm.types import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    Usage,
)

_OPTIONS_RE = re.compile(r"<<OPTIONS>>\s*(.*?)\s*<</OPTIONS>>", re.DOTALL)
_EMBED_DIM = 64


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()[:16], 16)


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class EchoClient:
    """Chat and embedding backend with no external dependency."""

    def __init__(self, *, skew: float = 0.75, dim: int = _EMBED_DIM) -> None:
        if not 0.0 < skew <= 1.0:
            raise ValueError("skew must be in (0, 1]")
        self._skew = skew
        self._dim = dim

    @property
    def provider(self) -> str:
        return "echo"

    def complete(self, request: ChatRequest) -> ChatResponse:
        prompt = "\n".join(m.content for m in request.messages)
        options = self._options(prompt)
        text = (
            self._choose(prompt, options, request.sample_index)
            if options
            else self._summarise(prompt, request.sample_index)
        )
        return ChatResponse(
            text=text,
            model=request.model,
            provider=self.provider,
            usage=Usage(_approx_tokens(prompt), _approx_tokens(text), 0.0),
            raw_finish_reason="stop",
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        vectors = tuple(self._vector(t) for t in request.texts)
        return EmbedResponse(
            vectors=vectors,
            model=request.model,
            provider=self.provider,
            usage=Usage(sum(_approx_tokens(t) for t in request.texts), 0, 0.0),
        )

    @staticmethod
    def _options(prompt: str) -> list[str]:
        match = _OPTIONS_RE.search(prompt)
        if not match:
            return []
        return [line.strip() for line in match.group(1).splitlines() if line.strip()]

    def _choose(self, prompt: str, options: list[str], sample_index: int) -> str:
        """Skewed deterministic draw over `options`."""
        favoured = _seed(prompt) % len(options)
        u = (_seed(prompt, str(sample_index)) % 10_000) / 10_000
        if u < self._skew or len(options) == 1:
            return options[favoured]
        others = [o for i, o in enumerate(options) if i != favoured]
        rank = _seed(prompt, str(sample_index), "alt") % len(others)
        return others[rank]

    @staticmethod
    def _summarise(prompt: str, sample_index: int) -> str:
        """Stand-in for a free-text generation, e.g. consolidation."""
        lines = [ln.strip() for ln in prompt.splitlines() if len(ln.strip()) > 40]
        if not lines:
            return f"[echo:{_seed(prompt, str(sample_index)) % 1000}]"
        picks = [lines[_seed(prompt, str(sample_index), str(i)) % len(lines)] for i in range(3)]
        return " ".join(f"{p[:120].rstrip()}." for p in dict.fromkeys(picks))

    def _vector(self, text: str) -> tuple[float, ...]:
        """Hashed bag-of-words, L2-normalised."""
        acc = [0.0] * self._dim
        for token in re.findall(r"\w+", text.lower()):
            acc[_seed(token) % self._dim] += 1.0
        norm = sum(v * v for v in acc) ** 0.5
        return tuple(v / norm for v in acc) if norm else tuple(acc)
