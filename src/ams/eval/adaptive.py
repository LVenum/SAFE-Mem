"""Adaptive attacks of spec §7.8 (1, 2, 4, 5), applied to MPBench's attack contexts."""

from __future__ import annotations

from dataclasses import replace

from ams.core.prompts import PromptLibrary

VARIANTS = ("forged_author", "forged_retention", "under_report", "conflict_evasion")


def apply(cases: list, variant: str | None, prompts: PromptLibrary) -> list:
    if not variant:
        return cases
    if variant not in VARIANTS:
        raise ValueError(f"unknown adaptive variant {variant!r}; known: {VARIANTS}")
    tpl = prompts.get(f"adaptive_{variant}", "v1")

    def render(c) -> str:
        values = {"context": c.context, "payload": c.expected_memory.strip()}
        return tpl.render(**{k: v for k, v in values.items() if k in tpl.placeholders()}).strip()

    return [
        replace(c, context=render(c)) if c.is_adversarial and c.context.strip() else c
        for c in cases
    ]
