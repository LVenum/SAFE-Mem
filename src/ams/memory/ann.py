"""Retrieval that scales: one index per gate partition (spec §11)."""

from __future__ import annotations

import threading
from collections.abc import Hashable, Iterable, Sequence

import numpy as np

from ams.memory.index import VectorIndex


class _Hnsw:
    """One HNSW graph (hnswlib) over cosine similarity, grown as entries arrive."""

    def __init__(self, dim: int, *, m: int = 16, ef_construction: int = 200, ef: int = 64):
        import hnswlib

        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(max_elements=1024, ef_construction=ef_construction, M=m)
        self._ef = ef
        self._ids: list = []

    def add(self, ids: Sequence, vectors: Sequence[Sequence[float]]) -> None:
        need = len(self._ids) + len(ids)
        if need > self._index.get_max_elements():
            self._index.resize_index(max(need, 2 * self._index.get_max_elements()))
        labels = np.arange(len(self._ids), need)
        self._index.add_items(np.asarray(vectors, dtype=np.float32), labels)
        self._ids.extend(ids)

    def top(self, query: Sequence[float], k: int) -> list[tuple]:
        n = len(self._ids)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)
        self._index.set_ef(max(self._ef, k))
        labels, dists = self._index.knn_query(np.asarray(query, dtype=np.float32), k=k)
        return [
            (self._ids[int(i)], 1.0 - float(d)) for i, d in zip(labels[0], dists[0], strict=True)
        ]


class _Exact:
    def __init__(self, dim: int):
        self._index = VectorIndex()

    def add(self, ids, vectors) -> None:
        self._index.add(ids, vectors)

    def top(self, query, k: int) -> list[tuple]:
        sims = self._index.scores(query)
        return sorted(sims.items(), key=lambda kv: kv[1], reverse=True)[:k]


class PartitionedIndex:
    def __init__(self, backend: str = "hnsw") -> None:
        if backend not in ("hnsw", "exact"):
            raise ValueError(f"unknown index backend {backend!r}; use 'hnsw' or 'exact'")
        self.backend = backend
        self._parts: dict[Hashable, _Hnsw | _Exact] = {}
        self._vectors: dict = {}
        self._seq: dict = {}
        self._lock = threading.Lock()

    def keys(self) -> list[Hashable]:
        return list(self._parts)

    def add(self, key: Hashable, ids: Sequence, vectors: Sequence[Sequence[float]]) -> None:
        if not ids:
            return
        with self._lock:
            part = self._parts.get(key)
            if part is None:
                cls = _Hnsw if self.backend == "hnsw" else _Exact
                part = self._parts[key] = cls(len(vectors[0]))
            part.add(ids, vectors)
            for i, v in zip(ids, vectors, strict=True):
                self._vectors[i] = np.asarray(v, dtype=np.float64)
                self._seq.setdefault(i, len(self._seq))

    def search(self, query: Sequence[float], k: int, keys: Iterable[Hashable]) -> list[tuple]:
        """Top `k` (id, cosine) over the given partitions, best first."""
        out: list[tuple] = []
        for key in keys:
            part = self._parts.get(key)
            if part is not None:
                out += part.top(query, k)
        out.sort(key=lambda kv: (-kv[1], self._seq[kv[0]]))
        return out[:k]

    def score(self, query: Sequence[float], item) -> float:
        """Exact cosine of one stored entry, for entries reached by traversal."""
        v = self._vectors.get(item)
        if v is None:
            return -1.0
        q = np.asarray(query, dtype=np.float64)
        d = np.linalg.norm(q) * np.linalg.norm(v)
        return float(q @ v / d) if d else 0.0
