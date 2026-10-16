"""Evidence chunks: the source text itself, retrievable beside claims (spec §10.1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ams.core.lattice import Standing
from ams.core.spans import cosine

MAX_CHUNK_CHARS = 1024


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    text: str
    standing: Standing
    source_ids: tuple[str, ...]
    label: str = ""


def cut(
    sentences: Sequence[str], vectors: Sequence[Sequence[float]], *, cap: int = MAX_CHUNK_CHARS
) -> list[list[int]]:
    """Index runs of `sentences`, broken at below-mean adjacent similarity or the cap."""
    n = len(sentences)
    if n == 0:
        return []
    sims = [cosine(vectors[i], vectors[i + 1]) for i in range(n - 1)]
    mean = sum(sims) / len(sims) if sims else 0.0
    runs: list[list[int]] = [[0]]
    size = len(sentences[0])
    for i in range(1, n):
        if sims[i - 1] < mean or size + len(sentences[i]) > cap:
            runs.append([i])
            size = len(sentences[i])
        else:
            runs[-1].append(i)
            size += len(sentences[i]) + 1
    return runs
