"""Entity coreference: owns the graph's connectivity (spec §4)."""

from __future__ import annotations

from ams.memory.graph import canonical_entity_key

_DROP = {"mr", "mrs", "ms", "dr", "the", "a", "an"}


def tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in canonical_entity_key(name).split() if t not in _DROP)


def same_referent(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    if a <= b or b <= a:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) == 1:
        (s,) = short
        return len(s) >= 3 and any(t.startswith(s) for t in long_)
    return False


def resolve(name: str, known: dict[str, frozenset[str]]) -> str | None:
    """The id of the one known entity `name` refers to, or None."""
    key = canonical_entity_key(name)
    if key in known:
        return key
    mine = tokens(name)
    hits = [eid for eid, toks in known.items() if same_referent(mine, toks)]
    return hits[0] if len(hits) == 1 else None
