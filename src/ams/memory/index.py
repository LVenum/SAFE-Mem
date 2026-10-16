"""Append-only vector index with numpy scoring."""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np


class VectorIndex:
    def __init__(self) -> None:
        self._ids: list = []
        self._rows: list = []
        self._matrix: np.ndarray | None = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence, vectors: Sequence[Sequence[float]]) -> None:
        with self._lock:
            self._ids.extend(ids)
            self._rows.extend(vectors)
            self._matrix = None

    def scores(self, query: Sequence[float]) -> dict:
        """Cosine similarity of `query` with every stored vector, by id."""
        with self._lock:
            if not self._ids:
                return {}
            if self._matrix is None:
                m = np.asarray(self._rows, dtype=np.float64).reshape(len(self._rows), -1)
                norms = np.linalg.norm(m, axis=1, keepdims=True)
                self._matrix = m / np.where(norms == 0, 1.0, norms)
            matrix, ids = self._matrix, list(self._ids)
        if not ids:
            return {}
        q = np.asarray(query, dtype=np.float64)
        n = np.linalg.norm(q)
        sims = matrix @ (q / n if n else q)
        return dict(zip(ids, sims.tolist(), strict=True))
