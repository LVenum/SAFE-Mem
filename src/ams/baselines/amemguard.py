"""A-MemGuard, run through its authors' own implementation.

Upstream: https://github.com/TangciuYueng/AMemGuard (MIT, (c) 2024 AI Secure),
vendored read-only by `scripts/fetch_third_party.sh` at a pinned commit.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from ams.baselines.base import DefenceResult
from ams.memory.base import MemoryEntry


def upstream_path(root: Path | None = None) -> Path:
    """Where the vendored checker lives."""
    base = root or Path(__file__).resolve().parents[3]
    return base / "third_party" / "AMemGuard" / "ReAct"


def _load_upstream(root: Path | None = None):
    """Import the authors' module from the vendored checkout."""
    path = upstream_path(root)
    if not (path / "consistency.py").exists():
        raise FileNotFoundError(
            f"A-MemGuard not vendored at {path}. Run scripts/fetch_third_party.sh first."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import consistency  # type: ignore[import-not-found]

    return consistency


def _ManagedProvider(base, client, model: str, workers: int):  # noqa: N802 - a class factory
    """An upstream `ModelProvider` backed by our `LLMClient`."""
    from ams.llm import ChatRequest, Message

    system = "You are a helpful and precise assistant for logical analysis and text generation."

    class Provider(base):
        def generate_batch(self, prompts, token_stats=False):
            def one(p):
                r = client.complete(
                    ChatRequest(
                        messages=(Message("system", system), Message("user", p)),
                        model=model,
                        temperature=1.0,
                        max_tokens=2048,
                    )
                )
                return r.text.strip(), r.usage

            out = [one(p) for p in prompts]
            stats = {
                "input_tokens": sum(u.input_tokens for _, u in out),
                "output_tokens": sum(u.output_tokens for _, u in out),
            }
            return [t for t, _ in out], stats

    return Provider()


class AMemGuard:
    """Consensus-based memory validation, delegating to the upstream checker."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        method: str = "llm",
        sentence_transformer_model_id: str = "all-mpnet-base-v2",
        max_workers: int = 5,
        upstream_root: Path | None = None,
        client=None,
    ) -> None:
        if method not in ("llm", "clustering"):
            raise ValueError(f"unknown method {method!r}; upstream supports 'llm' or 'clustering'")
        upstream = _load_upstream(upstream_root)
        self._method = method
        self._model = model

        if client is not None:
            provider = _ManagedProvider(upstream.ModelProvider, client, model, max_workers)
        elif base_url:
            provider = upstream.VLLMModel(
                model_name=model, base_url=base_url, api_key=api_key or "EMPTY"
            )
        else:
            provider = upstream.OpenAIModel(
                model_name=model, api_key=api_key, max_workers=max_workers
            )
        self._checker = upstream.ConsistencyChecker(
            model_provider=provider,
            sentence_transformer_model_id=sentence_transformer_model_id,
        )

    @property
    def name(self) -> str:
        return f"a_memguard[{self._method}]"

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        items = list(memories)
        if not items:
            return DefenceResult(kept=(), reason="empty pool")

        result = self._checker.check(
            query=query,
            memories=[m.text for m in items],
            selected_indexes=list(range(len(items))),
            method=self._method,
        )
        kept_idx = {e["index"] for e in result.get("consistent_memories", [])}
        flagged_idx = {e["index"] for e in result.get("inconsistent_memories", [])}

        kept = tuple(m for i, m in enumerate(items) if i in kept_idx)
        flagged = tuple(m for i, m in enumerate(items) if i in flagged_idx)
        unjudged = tuple(
            m for i, m in enumerate(items) if i not in kept_idx and i not in flagged_idx
        )
        return DefenceResult(
            kept=kept + unjudged,
            flagged=flagged,
            reason=f"consensus validation ({self._method})",
            detail={
                "n_flagged": len(flagged),
                "n_unjudged": len(unjudged),
                "token_usage": result.get("token_usage", {}),
                "verdicts": [
                    {
                        "index": e["index"],
                        "kept": key == "consistent_memories",
                        "chain": e.get("reasoning_chain", ""),
                    }
                    for key in ("consistent_memories", "inconsistent_memories")
                    for e in result.get(key, [])
                ],
            },
        )
