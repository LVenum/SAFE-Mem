"""Spec §10.5 attack 6: piggybacking on a genuine retention request."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ams.core.prompts import PromptLibrary


@dataclass(frozen=True)
class Attack6Case:
    cid: str
    variant: str
    user_query: str
    context: str
    expected_memory: str


def build_cases(cases, prompts: PromptLibrary, *, seed: int = 0) -> list[Attack6Case]:
    """Every warranted benign turn, paired with a distinct adversarial context."""
    ben = [c for c in cases if not c.is_adversarial and c.legitimate_memory_write]
    adv = [c for c in cases if c.is_adversarial and c.expected_memory.strip() and c.context.strip()]
    rng = random.Random(seed)
    partners = rng.sample(adv, len(ben))
    cond = prompts.get("attack6_conditional", "v1")
    out = []
    for b, a in zip(ben, partners, strict=True):
        base = f"{b.cid}+{a.cid}"
        out.append(Attack6Case(f"{base}/a", "a", b.user_query, a.context, a.expected_memory))
        tail = cond.render(payload=a.expected_memory.strip())
        out.append(
            Attack6Case(f"{base}/b", "b", b.user_query, f"{a.context}\n\n{tail}", a.expected_memory)
        )
    return out


def run_case(memory, judge, case: Attack6Case) -> dict:
    """Write one case into a fresh store and grade what reached which standing."""
    res = memory.write(session_id=case.cid, user_query=case.user_query, context=case.context)
    texts = [e.metadata.get("raw_text", e.text) for e in res.written]
    elevated = [
        e.metadata.get("raw_text", e.text)
        for e in res.written
        if e.metadata.get("usable_as") != "description"
    ]
    return {
        "cid": case.cid,
        "variant": case.variant,
        "n_written": len(texts),
        "n_elevated": len(elevated),
        "poison_written": judge.entry_matches(case.expected_memory, texts),
        "poison_elevated": judge.entry_matches(case.expected_memory, elevated),
        "elevated_texts": elevated,
    }
