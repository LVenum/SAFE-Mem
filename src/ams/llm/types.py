"""Provider-neutral request/response types."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One completion request."""

    messages: tuple[Message, ...]
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    stop: tuple[str, ...] = ()
    sample_index: int = 0
    reasoning_effort: str | None = None

    def cache_payload(self, provider: str) -> dict:
        payload = {
            "provider": provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
            "sample_index": self.sample_index,
            "messages": [asdict(m) for m in self.messages],
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def cache_key(self, provider: str) -> str:
        blob = json.dumps(self.cache_payload(provider), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    cached: bool = False
    raw_finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EmbedRequest:
    texts: tuple[str, ...]
    model: str

    def cache_key(self, provider: str) -> str:
        blob = json.dumps(
            {"provider": provider, "model": self.model, "texts": list(self.texts)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbedResponse:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    cached: bool = False
