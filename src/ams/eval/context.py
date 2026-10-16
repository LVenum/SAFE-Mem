"""Run context: config in, self-describing result directory out."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ams.core.prompts import PromptLibrary
from ams.llm import ClientSpec, ManagedClient, ResponseCache, UsageLedger, build_client


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except Exception:  # pragma: no cover - git absent or not a repo
        return "unknown"


@dataclass
class Config:
    """A thin typed view over the experiment YAML."""

    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls(raw=data)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"config is missing required key {dotted!r}")
        return value

    def override(self, dotted: str, value: Any) -> None:
        node = self.raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


class RunContext:
    """Owns the result directory, the shared cache, and the usage ledger."""

    def __init__(
        self,
        config: Config,
        *,
        results_root: Path | str | None = None,
        cache_path: Path | str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        name = config.get("name", "run")
        self.run_id = run_id or f"{name}-{stamp}"
        root = Path(results_root or os.environ.get("AMS_RESULTS_ROOT") or repo_root() / "results")

        candidate, n = root / self.run_id, 1
        while candidate.exists():
            n += 1
            candidate = root / f"{self.run_id}-{n}"
        if n > 1:
            self.run_id = f"{self.run_id}-{n}"
        self.dir = candidate
        self.dir.mkdir(parents=True)

        self.cache = ResponseCache(
            Path(
                cache_path
                or os.environ.get("AMS_CACHE_PATH")
                or repo_root() / ".llm_cache" / "responses.sqlite"
            )
        )
        self.ledger = UsageLedger(self.dir / "usage.jsonl")
        self.prompts = PromptLibrary()

    def client(self, spec: str, *, stage: str = "unattributed") -> ManagedClient:
        """A managed client for `spec`, or a two-backend one when the config asks."""
        s = ClientSpec.parse(spec)
        effort = (self.config.get("reasoning_effort") or {}).get(s.model)
        primary = build_client(
            spec, cache=self.cache, ledger=self.ledger, stage=stage, reasoning_effort=effort
        )
        fb_spec = (self.config.get("fallbacks") or {}).get(spec)
        fb = self.config.get("fallback_provider")
        if not fb_spec:
            if not fb or s.provider in (fb, "local", "echo"):
                return primary
            fb_spec = f"{fb}:{s.model}"
        from ams.llm import FallbackClient

        secondary = build_client(
            fb_spec,
            cache=self.cache,
            ledger=self.ledger,
            stage=f"{stage}_fallback",
            reasoning_effort=(self.config.get("reasoning_effort") or {}).get(
                ClientSpec.parse(fb_spec).model
            ),
        )
        client = FallbackClient(primary, secondary)
        self._fallbacks[stage] = client
        return client

    @property
    def _fallbacks(self) -> dict:
        if not hasattr(self, "_fb_registry"):
            self._fb_registry: dict = {}
        return self._fb_registry

    def fallback_summary(self) -> dict | None:
        """Per-stage counts of who served what, or None when no fallback is set."""
        fbs = getattr(self, "_fb_registry", {})
        if not fbs:
            return None
        out = {stage: c.counts.to_dict() for stage, c in fbs.items()}
        served = sum(c.counts.total for c in fbs.values())
        fell_back = sum(c.counts.fallback for c in fbs.values())
        out["_overall"] = {
            "calls": served,
            "served_by_fallback": fell_back,
            "fallback_share": round(fell_back / served, 4) if served else 0.0,
        }
        return out

    def path(self, filename: str) -> Path:
        return self.dir / filename

    def write_json(self, filename: str, payload: Any) -> Path:
        p = self.path(filename)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return p

    def finalise(self, extra: dict[str, Any] | None = None) -> Path:
        """Write the manifest. Call once, at the end of a run."""
        manifest = {
            "run_id": self.run_id,
            "finished_utc": datetime.now(UTC).isoformat(),
            "git_sha": git_sha(),
            "prompts": self.prompts.fingerprint(),
            "usage": self.ledger.summary(),
            "config": self.config.raw,
        }
        fb = self.fallback_summary()
        if fb:
            manifest["provider_fallback"] = fb
        if extra:
            manifest.update(extra)
        path = self.write_json("manifest.json", manifest)
        self.ledger.close()
        return path
