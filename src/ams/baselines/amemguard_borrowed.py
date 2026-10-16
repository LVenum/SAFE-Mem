"""Two of A-MemGuard's own comparison baselines, run from its vendored tree.

Upstream: https://github.com/TangciuYueng/AMemGuard (MIT, (c) 2024 AI Secure),
`ReAct/{classifier,auditor}.py`, which its `local_wikienv.py` selects with
`--audit_method distil | ro`. Their third, the perplexity filter (`ppl`), is not
run: it needs prompt-token probabilities, which the gpt-4o-mini API every other
component uses does not expose, and the user chose to drop it rather than run it
on a different model (2026-10-02).
"""

from __future__ import annotations

import ast
import contextlib
import io
import random
import re
import threading
import zlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from ams.baselines.base import DefenceResult
from ams.core.prompts import PromptLibrary
from ams.llm import ChatRequest, LLMClient, Message
from ams.memory.base import MemoryEntry


def upstream_dir(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / "third_party" / "AMemGuard"


def load_definitions(path: Path, **inject) -> ModuleType:
    """Execute only the imports, literal constants and def/class statements of `path`."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/fetch_third_party.sh first.")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, keep) or (isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant))
    ]
    mod = ModuleType(f"amemguard_upstream_{path.stem}")
    mod.__dict__.update(inject)
    exec(compile(tree, str(path), "exec"), mod.__dict__)  # noqa: S102 - vendored, pinned
    return mod


_TORCH_LOCK = threading.Lock()


def _verdicts(items: Sequence[MemoryEntry], dropped: set[int]) -> list[dict]:
    return [{"index": i, "kept": i not in dropped} for i in range(len(items))]


class RandECClassifier:
    """`--audit_method distil`: drop every entry erase-and-check calls harmful."""

    def __init__(
        self,
        *,
        weights: str | Path | None = None,
        mode: str = "suffix",
        upstream_root: Path | None = None,
    ):
        root = upstream_dir(upstream_root)
        weights = Path(
            weights or root / "certified-llm-safety" / "models" / f"distilbert_{mode}.pt"
        )
        if not weights.exists():
            raise FileNotFoundError(
                f"RandEC weights not found at {weights}; see scripts/fetch_third_party.sh"
            )
        mod = load_definitions(root / "ReAct" / "classifier.py")
        self._detector = mod.RandECDetector(model_path=str(weights))
        self._mode = mode

    @property
    def name(self) -> str:
        return "randec_distil"

    def harmful(self, text: str) -> bool:
        with _TORCH_LOCK, contextlib.redirect_stdout(io.StringIO()):
            random.seed(zlib.crc32(text.encode("utf-8")))
            return bool(self._detector.detect_query(text, mode=self._mode))

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        items = list(memories)
        dropped = {i for i, m in enumerate(items) if self.harmful(m.text + "\n")}
        return DefenceResult(
            kept=tuple(m for i, m in enumerate(items) if i not in dropped),
            flagged=tuple(m for i, m in enumerate(items) if i in dropped),
            reason=f"RandEC/DistilBERT ({self._mode})",
            detail={"verdicts": _verdicts(items, dropped)},
        )


class LLMSanitiser:
    """`--audit_method ro`: audit, and rewrite what the auditor calls harmful."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        prompts: PromptLibrary | None = None,
        version: str = "v1",
    ) -> None:
        self._client, self._model = client, model
        p = prompts or PromptLibrary()
        self._audit = p.get("sanitiser_audit", version)
        self._clean = p.get("sanitiser_clean", version)
        self._system = p.get("sanitiser_system", version).template

    @property
    def name(self) -> str:
        return "llm_sanitiser"

    def _ask(self, rendered: str) -> str:
        return self._client.complete(
            ChatRequest(
                messages=(Message("system", self._system), Message("user", rendered)),
                model=self._model,
                temperature=0.0,
                max_tokens=1024,
            )
        ).text

    def status(self, text: str) -> str:
        """Upstream's parse: harmful | safe | error. Only 'harmful' triggers a rewrite."""
        return parse_status(self._ask(self._audit.render(memory=text)))

    def sanitise(self, text: str) -> str:
        return parse_clean(self._ask(self._clean.render(memory=text)))

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        kept, flagged, verdicts = [], [], []
        for i, m in enumerate(memories):
            st = self.status(m.text)
            v = {"index": i, "status": st, "kept": True}
            if st == "harmful":
                flagged.append(m)
                clean = self.sanitise(m.text).strip()
                v["rewrite"] = clean
                if clean:
                    kept.append(replace(m, text=clean))
                else:
                    v["kept"] = False
            else:
                kept.append(m)
            verdicts.append(v)
        return DefenceResult(
            kept=tuple(kept),
            flagged=tuple(flagged),
            reason="LLM audit + sanitise",
            detail={"verdicts": verdicts, "n_rewritten": len(flagged)},
        )


def parse_status(output: str) -> str:
    m = re.search(r"\[STATUS\]:\s*(harmful|safe|HARMFUL|SAFE)", output, re.IGNORECASE)
    return m.group(1).lower() if m else "error"


def parse_clean(output: str) -> str:
    return output.replace("[Santized Memory Log]: ", "", 1)
