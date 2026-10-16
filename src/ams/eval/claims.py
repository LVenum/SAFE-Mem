"""What kind of claim a memory entry makes: rule, preference, fact, credential."""

from __future__ import annotations

from collections.abc import Sequence

from ams.core.prompts import PromptLibrary
from ams.llm import ChatRequest, LLMClient, Message, map_concurrent

KINDS = ("rule", "preference", "fact", "credential", "other")


def classify(
    client: LLMClient,
    *,
    model: str,
    payloads: Sequence[str],
    prompts: PromptLibrary | None = None,
    version: str = "v1",
    workers: int = 8,
) -> list[str]:
    prompt = (prompts or PromptLibrary()).get("claim_type", version)

    def one(p: str) -> str:
        text = (
            client.complete(
                ChatRequest(
                    messages=(Message("user", prompt.render(payload=p.strip())),),
                    model=model,
                    temperature=0.0,
                    max_tokens=4,
                )
            )
            .text.strip()
            .lower()
        )
        return next((k for k in KINDS if text.startswith(k)), "other")

    return map_concurrent(one, payloads, workers=workers, progress="claim_type")
