"""Versioned prompt loading."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def default_prompt_root() -> Path:
    """`<repo>/prompts`, located relative to this file."""
    return Path(__file__).resolve().parents[3] / "prompts"


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: str
    template: str

    @property
    def id(self) -> str:
        return f"{self.name}/{self.version}"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:16]

    def placeholders(self) -> frozenset[str]:
        return frozenset(_PLACEHOLDER.findall(self.template))

    def render(self, **values: object) -> str:
        missing = self.placeholders() - set(values)
        if missing:
            raise KeyError(f"{self.id} is missing values for {sorted(missing)}")
        extra = set(values) - self.placeholders()
        if extra:
            raise KeyError(f"{self.id} has no placeholders for {sorted(extra)}")
        return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), self.template)


class PromptLibrary:
    """Reads `prompts/` and remembers what was used."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else default_prompt_root()
        self._used: dict[str, str] = {}

    def get(self, name: str, version: str = "v1") -> Prompt:
        path = self.root / name / f"{version}.txt"
        if not path.exists():
            available = sorted(p.stem for p in (self.root / name).glob("*.txt"))
            raise FileNotFoundError(
                f"no prompt {name}/{version} under {self.root}; available: {available}"
            )
        prompt = Prompt(name=name, version=version, template=_read(path))
        self._used[prompt.id] = prompt.sha256
        return prompt

    def fingerprint(self) -> dict[str, str]:
        """Prompt id -> content hash, for the result directory's manifest."""
        return dict(sorted(self._used.items()))


@lru_cache(maxsize=256)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
