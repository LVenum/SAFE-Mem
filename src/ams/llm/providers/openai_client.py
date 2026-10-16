"""OpenAI adapter. One of only two places in the package allowed to import `openai`."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from ams.llm import pricing
from ams.llm.base import ContentFiltered, ProviderError
from ams.llm.types import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    Usage,
)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class OpenAIClient:
    def __init__(
        self,
        *,
        api_key: str | Callable[[], str] | None = None,
        base_url: str | None = None,
        provider: str = "openai",
        timeout: float = 120.0,
        require_key: bool = True,
        sdk_client: Any = None,
    ) -> None:
        """`sdk_client` lets a subclass supply an already-built SDK object."""
        if sdk_client is not None:
            self._provider = provider
            self._client = sdk_client
            return

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ProviderError(
                provider, "openai SDK not installed; `uv pip install -e '.[openai]'`"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if require_key and not key:
            raise ProviderError(provider, "OPENAI_API_KEY is not set")

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None

        if require_key and key and not key.startswith("sk-") and not resolved_base_url:
            raise ProviderError(
                provider,
                "OPENAI_API_KEY does not look like an api.openai.com key (expected an "
                "'sk-' prefix) and no endpoint is set. For an OpenAI-compatible gateway, "
                "give the endpoint: `openai:<model>@https://<host>/openai/v1`, or set "
                "OPENAI_BASE_URL. For classic Azure OpenAI, use the azure provider.",
            )
        self._provider = provider
        self.base_url = resolved_base_url
        self._client = OpenAI(
            api_key=key or "not-needed", base_url=resolved_base_url, timeout=timeout
        )

    @property
    def provider(self) -> str:
        return self._provider

    def complete(self, request: ChatRequest) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.reasoning_effort is not None and self._provider == "openrouter":
            effort = request.reasoning_effort
            kwargs["extra_body"] = {
                "reasoning": {"enabled": False} if effort == "none" else {"effort": effort}
            }
        elif request.reasoning_effort is not None:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.stop:
            kwargs["stop"] = list(request.stop)
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise self._wrap(exc) from exc

        choice = resp.choices[0]
        usage = self._usage(getattr(resp, "usage", None), request.model)
        return ChatResponse(
            text=choice.message.content or "",
            model=resp.model,
            provider=self._provider,
            usage=usage,
            raw_finish_reason=choice.finish_reason,
        )

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        try:
            resp = self._client.embeddings.create(model=request.model, input=list(request.texts))
        except Exception as exc:
            raise self._wrap(exc) from exc
        return EmbedResponse(
            vectors=tuple(tuple(d.embedding) for d in resp.data),
            model=resp.model,
            provider=self._provider,
            usage=self._usage(getattr(resp, "usage", None), request.model),
        )

    def _usage(self, raw: Any, model: str) -> Usage:
        if raw is None:
            return Usage()
        inp = getattr(raw, "prompt_tokens", 0) or 0
        out = getattr(raw, "completion_tokens", 0) or 0
        return Usage(inp, out, pricing.estimate_cost(self._provider, model, inp, out))

    def _wrap(self, exc: Exception) -> Exception:
        text = str(exc)
        if any(
            m in text
            for m in ("content_filter", "ResponsibleAIPolicy", "cyber_policy", "content_policy")
        ):
            body = getattr(exc, "body", None)
            inner = (body or {}).get("innererror", {}) if isinstance(body, dict) else {}
            return ContentFiltered(self._provider, text[:200], inner.get("content_filter_result"))
        status = getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )
        retryable = status in _RETRYABLE_STATUS if status else _looks_transient(exc)
        if "insufficient_quota" in text or "credit_balance_exhausted" in text:
            retryable = False
        return ProviderError(self._provider, f"{type(exc).__name__}: {exc}", retryable=retryable)


def _looks_transient(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(k in name for k in ("timeout", "connection", "ratelimit", "apierror"))


DEFAULT_AZURE_API_VERSION = "2024-10-21"


def entra_token_provider(scope: str = "https://ai.azure.com/.default") -> Callable[[], str]:
    """A bearer-token callable for an Azure AI Services endpoint."""
    try:
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ProviderError(
            "azure_entra",
            "azure-identity not installed; `uv pip install -e '.[azure]'`",
        ) from exc
    return get_bearer_token_provider(DefaultAzureCredential(), scope)


def resource_key_var(endpoint: str) -> str:
    """The env var holding the key for one Azure resource, so several resources
    can be used at once: https://my-res.services.ai.azure.com/... -> AZURE_AI_KEY_MY_RES."""
    from urllib.parse import urlparse

    label = (urlparse(endpoint).hostname or "").split(".")[0]
    return "AZURE_AI_KEY_" + "".join(c if c.isalnum() else "_" for c in label).upper()


class AzureAIServicesClient(OpenAIClient):
    """Azure AI Services v1, reached as a plain OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        deployment: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        resolved = (
            endpoint
            or os.environ.get("AZURE_AI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        if not resolved:
            raise ProviderError(
                "aiservices",
                "no endpoint. Set AZURE_AI_ENDPOINT, or use the spec form "
                "`aiservices:<deployment>@https://<resource>.services.ai.azure.com/openai/v1`",
            )
        resolved = resolved.rstrip("/")
        if not resolved.endswith("/openai/v1"):
            resolved = f"{resolved}/openai/v1"
        key = (
            api_key
            or os.environ.get(resource_key_var(resolved))
            or os.environ.get("AZURE_AI_API_KEY")
            or os.environ.get("AZURE_AI_KEY")
        )
        self.deployment = deployment
        super().__init__(
            api_key=key or entra_token_provider(),
            base_url=resolved,
            provider="aiservices",
            timeout=timeout,
            require_key=False,
        )


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI. Same wire format, different addressing."""

    def __init__(
        self,
        *,
        deployment: str,
        endpoint: str | None = None,
        api_version: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ProviderError(
                "azure", "openai SDK not installed; `uv pip install -e '.[openai]'`"
            ) from exc

        key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("azure", "AZURE_OPENAI_API_KEY (or OPENAI_API_KEY) is not set")

        resolved_endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not resolved_endpoint:
            raise ProviderError(
                "azure",
                "no endpoint. Set AZURE_OPENAI_ENDPOINT, or use the spec form "
                "`azure:<deployment>@https://<resource>.openai.azure.com`",
            )
        version = (
            api_version or os.environ.get("AZURE_OPENAI_API_VERSION") or DEFAULT_AZURE_API_VERSION
        )
        self.deployment = deployment
        super().__init__(
            provider="azure",
            sdk_client=AzureOpenAI(
                api_key=key,
                azure_endpoint=resolved_endpoint.rstrip("/"),
                api_version=version,
                timeout=timeout,
            ),
        )


class VLLMClient(OpenAIClient):
    """Self-hosted open-weight backbone behind an OpenAI-compatible server."""

    def __init__(self, *, base_url: str, timeout: float = 600.0) -> None:
        super().__init__(
            api_key="EMPTY",
            base_url=base_url,
            provider="vllm",
            timeout=timeout,
            require_key=False,
        )
