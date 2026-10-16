"""USD per million tokens, by provider and model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, Price] = {
    "openai:gpt-4o-mini": Price(0.15, 0.60),
    "openai:gpt-4o": Price(2.50, 10.00),
    "openai:text-embedding-3-small": Price(0.02, 0.0),
    "openai:text-embedding-3-large": Price(0.13, 0.0),
    "openai:gpt-5": Price(1.25, 10.00),
    "openai:gpt-5.6-terra": Price(2.00, 12.00),
    "anthropic:claude-haiku-4-5-20251001": Price(1.00, 5.00),
    "anthropic:claude-sonnet-5": Price(3.00, 15.00),
    "anthropic:claude-opus-5": Price(5.00, 25.00),
    "azure:gpt-4o-mini": Price(0.15, 0.60),
    "azure:gpt-4o": Price(2.50, 10.00),
    "azure:gpt-5": Price(1.25, 10.00),
    "aiservices:gpt-4o-mini": Price(0.15, 0.60),
    "aiservices:gpt-5": Price(1.25, 10.00),
    "aiservices:gpt-5.6-terra": Price(2.00, 12.00),
    "azure:text-embedding-3-small": Price(0.02, 0.0),
    "azure:text-embedding-3-large": Price(0.13, 0.0),
    "openrouter:anthropic/claude-haiku-5.5": Price(0.10, 0.50),
    "openrouter:qwen/qwen3.8-flash": Price(0.15, 0.47),
    "vllm:*": Price(0.0, 0.0),
    "local:*": Price(0.0, 0.0),
    "echo:*": Price(0.0, 0.0),
}


def lookup(provider: str, model: str) -> Price | None:
    """Exact, then longest priced prefix (Azure returns dated ids like
    gpt-4o-mini-2024-07-18), then the provider wildcard."""
    if f"{provider}:{model}" in PRICES:
        return PRICES[f"{provider}:{model}"]
    prefixes = [
        k for k in PRICES if k.startswith(f"{provider}:") and model.startswith(k.split(":", 1)[1])
    ]
    if prefixes:
        return PRICES[max(prefixes, key=len)]
    return PRICES.get(f"{provider}:*")


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    price = lookup(provider, model)
    if price is None:
        return 0.0
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000


def is_priced(provider: str, model: str) -> bool:
    return lookup(provider, model) is not None
