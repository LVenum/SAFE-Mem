"""Caching, retry, and bounded concurrency — as decorators over `LLMClient`."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import TypeVar

from ams.llm.base import ContentFiltered, LLMClient, ProviderError
from ams.llm.cache import NullCache, ResponseCache
from ams.llm.ledger import UsageLedger
from ams.llm.types import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    Usage,
)

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 10
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter: float = 0.25

    def delay(self, attempt: int) -> float:
        raw = min(self.base_delay * (2**attempt), self.max_delay)
        return raw * (1.0 + random.uniform(-self.jitter, self.jitter))


class ManagedClient:
    """Wraps a provider with cache, retry, and usage accounting."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        cache: ResponseCache | None = None,
        ledger: UsageLedger | None = None,
        retry: RetryPolicy | None = None,
        stage: str = "unattributed",
        sleep: Callable[[float], None] = time.sleep,
        reasoning_effort: str | None = None,
    ) -> None:
        self._inner = inner
        self._effort = reasoning_effort
        self._cache = cache if cache is not None else NullCache()
        self._ledger = ledger
        self._retry = retry or RetryPolicy()
        self._stage = stage
        self._sleep = sleep

    @property
    def provider(self) -> str:
        return self._inner.provider

    def for_stage(self, stage: str) -> ManagedClient:
        """A view of this client whose calls are attributed to `stage`."""
        return ManagedClient(
            self._inner,
            cache=self._cache,
            ledger=self._ledger,
            retry=self._retry,
            stage=stage,
            sleep=self._sleep,
            reasoning_effort=self._effort,
        )

    def complete(self, request: ChatRequest) -> ChatResponse:
        if self._effort is not None and request.reasoning_effort is None:
            request = replace(request, reasoning_effort=self._effort)
        key = request.cache_key(self.provider)
        hit = self._cache.get(key)
        if hit is not None:
            resp = ChatResponse(
                text=hit["text"],
                model=hit["model"],
                provider=hit["provider"],
                usage=Usage(**hit["usage"]),
                cached=True,
                raw_finish_reason=hit.get("raw_finish_reason"),
            )
            self._record(resp.model, resp.usage, cached=True)
            return resp

        resp = self._call(lambda: self._inner.complete(request))
        self._cache.put(
            key,
            provider=resp.provider,
            model=resp.model,
            kind="chat",
            payload={
                "text": resp.text,
                "model": resp.model,
                "provider": resp.provider,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "cost_usd": resp.usage.cost_usd,
                },
                "raw_finish_reason": resp.raw_finish_reason,
            },
        )
        self._record(resp.model, resp.usage, cached=False)
        return resp

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        key = request.cache_key(self.provider)
        hit = self._cache.get(key)
        if hit is not None:
            resp = EmbedResponse(
                vectors=tuple(tuple(v) for v in hit["vectors"]),
                model=hit["model"],
                provider=hit["provider"],
                usage=Usage(**hit["usage"]),
                cached=True,
            )
            self._record(resp.model, resp.usage, cached=True)
            return resp

        resp = self._call(lambda: self._inner.embed(request))
        self._cache.put(
            key,
            provider=resp.provider,
            model=resp.model,
            kind="embed",
            payload={
                "vectors": [list(v) for v in resp.vectors],
                "model": resp.model,
                "provider": resp.provider,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "cost_usd": resp.usage.cost_usd,
                },
            },
        )
        self._record(resp.model, resp.usage, cached=False)
        return resp

    def _call(self, fn: Callable[[], T]) -> T:
        last: Exception | None = None
        for attempt in range(self._retry.max_attempts):
            try:
                return fn()
            except ContentFiltered:
                raise
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt == self._retry.max_attempts - 1:
                    raise
                self._sleep(self._retry.delay(attempt))
        raise last  # pragma: no cover - loop always returns or raises

    def _record(self, model: str, usage: Usage, *, cached: bool) -> None:
        if self._ledger is None:
            return
        from ams.llm import pricing

        self._ledger.record(
            provider=self.provider,
            model=model,
            usage=usage,
            stage=self._stage,
            cached=cached,
            priced=pricing.is_priced(self.provider, model),
        )


def progress_bar(total: int, label: str | None):
    """A tqdm bar (count, rate, ETA), or None when no label is given."""
    if not label:
        return None
    from tqdm import tqdm

    return tqdm(total=total, desc=label, dynamic_ncols=True, mininterval=1.0)


def map_concurrent(
    fn: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    workers: int = 8,
    progress: str | None = None,
) -> list[R]:
    """Run `fn` over `items` with a bounded pool, preserving input order."""
    items = list(items)
    if not items:
        return []
    bar = progress_bar(len(items), progress)
    try:
        if workers <= 1:
            out = []
            for x in items:
                out.append(fn(x))
                if bar:
                    bar.update()
            return out
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fn, x) for x in items]
            for _ in as_completed(futures):
                if bar:
                    bar.update()
            return [f.result() for f in futures]
    finally:
        if bar:
            bar.close()
