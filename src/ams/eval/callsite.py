"""Scope, eligibility-first selection and the call-site gate, for tool-using agents."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ams.core import sentinel
from ams.core.lattice import (
    GLOBAL,
    Domain,
    Rung,
    Standing,
    meet_all,
    principal,
)

SYSTEM = Standing(Rung.PERMISSION_TRUST, GLOBAL)
TOOL = Standing(Rung.ATTRIBUTE, GLOBAL)
RECORD_TYPE = Rung.PERMISSION_TRUST

_IDENT = re.compile(r"(?<![\w.])\d{3,}(?![\w.])")


def to_dict(s: Standing) -> dict:
    return {"level": s.level.name, "kind": s.domain.kind, "name": s.domain.name}


def from_dict(d: dict | None) -> Standing:
    if not d:
        return SYSTEM
    return Standing(Rung[d["level"]], Domain(d["kind"], d.get("name", "")))


def record_standing(asker: str, context: Sequence[Standing]) -> Standing:
    turn = Standing(Rung.PERMISSION_TRUST, principal(asker))
    return meet_all([turn, TOOL, *context])


def usable(s: Standing, *, required: Rung, asker: str, scoped: bool = True) -> Rung:
    """What this record may be used as for this question: its own rung if the
    gate admits it here, otherwise description."""
    domain = principal(asker) if scoped else GLOBAL
    probe = s if scoped else Standing(s.level, GLOBAL)
    if probe.domain.covers(domain):
        return Rung(min(RECORD_TYPE, s.level))
    return Rung.DESCRIPTION


def select(
    ranked: Sequence[int],
    standings: Sequence[Standing],
    *,
    required: Rung,
    asker: str,
    k: int,
    mode: str = "confine",
    scoped: bool = True,
) -> list[tuple[int, Rung]]:
    """(record index, usable rung) for the prompt, eligibility first (§6.1)."""
    use = {i: usable(standings[i], required=required, asker=asker, scoped=scoped) for i in ranked}
    ok = [i for i in ranked if required == Rung.DESCRIPTION or use[i] >= required]
    out = [(i, use[i]) for i in ok[:k]]
    if mode == "confine" and len(out) < k:
        chosen = {i for i, _ in out}
        out += [(i, Rung.DESCRIPTION) for i in ranked if i not in chosen][: k - len(out)]
    elif mode != "withhold" and mode != "confine":
        raise ValueError(f"mode must be confine or withhold, got {mode!r}")
    return out


def identifiers(text: str) -> set[str]:
    return set(_IDENT.findall(text or ""))


def check_call(call: str, *, question: str, prompt: str, required: Rung) -> tuple[bool, dict]:
    """May this call run? (allowed, detail). See the module docstring."""
    own = identifiers(question)
    entries = sentinel.blocks(prompt)
    refused = {}
    for v in sorted(identifiers(call) - own):
        carriers = [a for a, body in entries if v in identifiers(body)]
        if not carriers:
            continue
        if not any(_rung(a) >= max(required, Rung.ATTRIBUTE) for a in carriers):
            refused[v] = [a.get("entry_id", "") for a in carriers]
    return not refused, {"refused_values": refused}


def _rung(attrs: dict) -> Rung:
    return Rung[attrs.get("usable", "description").upper()]
