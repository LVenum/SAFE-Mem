"""Span segmentation and cross-variant alignment."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations

from ams.core.schema import Span

Vector = Sequence[float]
EmbedFn = Callable[[Sequence[str]], Sequence[Vector]]

_SENTENCE_END = re.compile(r"(?<=[.!?])[\s\n]+(?=[A-Z0-9\"'(\[])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_BRUTE_FORCE_LIMIT = 8


def segment(text: str) -> tuple[Span, ...]:
    """Split a derived memory into spans. Sentence-level by default."""
    normalised = _BULLET.sub("\n", text.strip())
    pieces: list[str] = []
    for block in (b.strip() for b in normalised.split("\n")):
        if block:
            pieces.extend(p.strip() for p in _SENTENCE_END.split(block) if p.strip())
    return tuple(Span(index=i, text=p) for i, p in enumerate(pieces))


def cosine(a: Vector, b: Vector) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def align(
    baseline: Sequence[str],
    variant: Sequence[str],
    embed: EmbedFn,
    *,
    min_similarity: float = 0.2,
) -> tuple[int | None, ...]:
    """Map each variant span to a baseline span index, or None."""
    if not baseline or not variant:
        return tuple(None for _ in variant)

    vecs = list(embed([*baseline, *variant]))
    b_vecs, v_vecs = vecs[: len(baseline)], vecs[len(baseline) :]
    sim = [[cosine(v, b) for b in b_vecs] for v in v_vecs]

    pairs = (
        _optimal_assignment(sim)
        if max(len(baseline), len(variant)) <= _BRUTE_FORCE_LIMIT
        else _greedy_assignment(sim)
    )
    out: list[int | None] = [None] * len(variant)
    for vi, bi in pairs.items():
        if sim[vi][bi] >= min_similarity:
            out[vi] = bi
    return tuple(out)


def _optimal_assignment(sim: list[list[float]]) -> dict[int, int]:
    """Exact maximum-weight injective assignment by enumeration."""
    n_v, n_b = len(sim), len(sim[0])
    best_score = float("-inf")
    best: dict[int, int] = {}
    if n_v <= n_b:
        for combo in permutations(range(n_b), n_v):
            score = sum(sim[vi][bi] for vi, bi in enumerate(combo))
            if score > best_score:
                best_score, best = score, {vi: bi for vi, bi in enumerate(combo)}
    else:
        for combo in permutations(range(n_v), n_b):
            score = sum(sim[vi][bi] for bi, vi in enumerate(combo))
            if score > best_score:
                best_score, best = score, {vi: bi for bi, vi in enumerate(combo)}
    return best


def _greedy_assignment(sim: list[list[float]]) -> dict[int, int]:
    """Descending-similarity greedy fallback for large span counts."""
    candidates = sorted(
        ((sim[vi][bi], vi, bi) for vi in range(len(sim)) for bi in range(len(sim[0]))),
        reverse=True,
    )
    out: dict[int, int] = {}
    used_b: set[int] = set()
    for _, vi, bi in candidates:
        if vi not in out and bi not in used_b:
            out[vi] = bi
            used_b.add(bi)
    return out


@dataclass(frozen=True, slots=True)
class VariationReport:
    """Per-baseline-span variation, plus the naive comparison it corrects."""

    aligned: tuple[float, ...]
    naive: tuple[float, ...]
    unmatched_rate: float

    @property
    def concentration(self) -> int:
        """How many baseline spans carry most of the variation."""
        if not self.aligned:
            return 0
        peak = max(self.aligned)
        if peak <= 1e-9:
            return 0
        return sum(1 for v in self.aligned if v >= 0.5 * peak)


def variation_report(
    baseline: Sequence[str],
    variants: Sequence[Sequence[str]],
    embed: EmbedFn,
    *,
    min_similarity: float = 0.2,
) -> VariationReport:
    """How much each baseline span moves across variants."""
    if not baseline:
        return VariationReport((), (), 0.0)

    per_span: list[list[float]] = [[] for _ in baseline]
    naive_per_span: list[list[float]] = [[] for _ in baseline]
    unmatched = 0
    total = 0

    b_vecs = list(embed(baseline))
    for variant in variants:
        if not variant:
            for acc in per_span:
                acc.append(0.0)
            continue
        mapping = align(baseline, variant, embed, min_similarity=min_similarity)
        v_vecs = list(embed(variant))
        for vi, bi in enumerate(mapping):
            total += 1
            if bi is None:
                unmatched += 1
            else:
                per_span[bi].append(cosine(v_vecs[vi], b_vecs[bi]))
        for bi in range(len(baseline)):
            naive_per_span[bi].append(cosine(v_vecs[bi], b_vecs[bi]) if bi < len(v_vecs) else 0.0)

    def _variation(sims: list[list[float]]) -> tuple[float, ...]:
        return tuple(1.0 - (sum(s) / len(s)) if s else 1.0 for s in sims)

    return VariationReport(
        aligned=_variation(per_span),
        naive=_variation(naive_per_span),
        unmatched_rate=unmatched / total if total else 0.0,
    )
