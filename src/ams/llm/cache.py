"""Content-addressed response cache."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL DEFAULT (julianday('now'))
);
CREATE INDEX IF NOT EXISTS responses_provider_model ON responses (provider, model);
"""


class ResponseCache:
    """SQLite-backed cache, safe to share across the worker pool."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        with self._lock:
            self._connect().executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Callers hold `self._lock`."""
        if self._conn is None:
            conn = sqlite3.connect(
                self.path, timeout=30.0, isolation_level=None, check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = (
                self._connect()
                .execute("SELECT payload FROM responses WHERE key = ?", (key,))
                .fetchone()
            )
        return json.loads(row[0]) if row else None

    def put(self, key: str, *, provider: str, model: str, kind: str, payload: dict) -> None:
        blob = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            self._connect().execute(
                "INSERT OR IGNORE INTO responses (key, provider, model, kind, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, provider, model, kind, blob),
            )

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = (
                self._connect()
                .execute("SELECT provider, model, COUNT(*) FROM responses GROUP BY provider, model")
                .fetchall()
            )
        return {f"{p}:{m}": n for p, m, n in rows}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class NullCache(ResponseCache):
    """Disabled cache, for tests that need to count real calls."""

    def __init__(self) -> None:  # noqa: D107 - deliberately does not call super
        pass

    def get(self, key: str) -> dict | None:
        return None

    def put(self, key: str, **kwargs: object) -> None:
        return None

    def stats(self) -> dict[str, int]:
        return {}

    def close(self) -> None:
        return None
