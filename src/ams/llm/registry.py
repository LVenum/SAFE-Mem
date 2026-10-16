"""Build a client from a string spec."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ams.core.env import load_env
from ams.llm.base import LLMClient
from ams.llm.cache import ResponseCache
from ams.llm.ledger import UsageLedger
from ams.llm.pool import ManagedClient, RetryPolicy

DEFAULT_MODELS = {
    "echo": "echo-1",
    "local": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
}


@dataclass(frozen=True, slots=True)
class ClientSpec:
    provider: str
    model: str
    base_url: str | None = None

    @classmethod
    def parse(cls, spec: str) -> ClientSpec:
        if ":" not in spec:
            raise ValueError(
                f"bad client spec {spec!r}; expected '<provider>:<model>' e.g. 'openai:gpt-4o-mini'"
            )
        provider, _, rest = spec.partition(":")
        provider = provider.strip().lower()
        base_url: str | None = None
        if "@" in rest:
            rest, _, base_url = rest.partition("@")
        model = rest.strip() or DEFAULT_MODELS.get(provider, "")
        if not model:
            raise ValueError(f"client spec {spec!r} names no model and {provider!r} has no default")
        if provider == "vllm" and not base_url:
            raise ValueError(
                f"vllm spec {spec!r} needs a base url, e.g. "
                "'vllm:meta-llama/Llama-3.1-8B-Instruct@http://localhost:8000/v1'"
            )
        if provider == "aiservices" and not (
            base_url
            or os.environ.get("AZURE_AI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        ):
            raise ValueError(
                f"aiservices spec {spec!r} needs an endpoint: either "
                "'aiservices:<deployment>@https://<resource>.services.ai.azure.com/openai/v1' "
                "or AZURE_AI_ENDPOINT in the environment"
            )
        if provider == "azure" and not (base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")):
            raise ValueError(
                f"azure spec {spec!r} needs an endpoint: either "
                "'azure:<deployment>@https://<resource>.openai.azure.com' or "
                "AZURE_OPENAI_ENDPOINT in the environment"
            )
        return cls(provider=provider, model=model, base_url=base_url or None)

    def __str__(self) -> str:
        tail = f"@{self.base_url}" if self.base_url else ""
        return f"{self.provider}:{self.model}{tail}"


def build_raw_client(spec: str | ClientSpec) -> LLMClient:
    """A bare provider adapter: no cache, no retry, no accounting."""
    s = ClientSpec.parse(spec) if isinstance(spec, str) else spec
    if s.provider == "echo":
        from ams.llm.providers.echo_client import EchoClient

        return EchoClient()

    load_env()

    if s.provider == "openai":
        from ams.llm.providers.openai_client import OpenAIClient

        return OpenAIClient(base_url=s.base_url)
    if s.provider == "azure":
        from ams.llm.providers.openai_client import AzureOpenAIClient

        return AzureOpenAIClient(deployment=s.model, endpoint=s.base_url)
    if s.provider == "aiservices":
        from ams.llm.providers.openai_client import AzureAIServicesClient

        return AzureAIServicesClient(deployment=s.model, endpoint=s.base_url)
    if s.provider == "openrouter":
        from ams.llm.providers.openai_client import OpenAIClient

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(f"openrouter spec {s} needs OPENROUTER_API_KEY in the environment")
        return OpenAIClient(
            api_key=key,
            base_url=s.base_url or "https://openrouter.ai/api/v1",
            provider="openrouter",
        )
    if s.provider == "anthropic":
        from ams.llm.providers.anthropic_client import AnthropicClient

        return AnthropicClient()
    if s.provider == "local":
        from ams.llm.providers.local_client import LocalEmbeddingClient

        return LocalEmbeddingClient(model=s.model)
    if s.provider == "vllm":
        from ams.llm.providers.openai_client import VLLMClient

        assert s.base_url is not None
        return VLLMClient(base_url=s.base_url)
    raise ValueError(
        f"unknown provider {s.provider!r}; known: echo, openai, azure, aiservices, "
        "openrouter, anthropic, vllm, local"
    )


def build_client(
    spec: str | ClientSpec,
    *,
    cache: ResponseCache | None = None,
    ledger: UsageLedger | None = None,
    retry: RetryPolicy | None = None,
    stage: str = "unattributed",
    reasoning_effort: str | None = None,
) -> ManagedClient:
    """The normal entry point: a provider wrapped in cache + retry + accounting."""
    return ManagedClient(
        build_raw_client(spec),
        cache=cache,
        ledger=ledger,
        retry=retry,
        stage=stage,
        reasoning_effort=reasoning_effort,
    )


def model_of(spec: str | ClientSpec) -> str:
    return (ClientSpec.parse(spec) if isinstance(spec, str) else spec).model
