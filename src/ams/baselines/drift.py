"""DRIFT's Injection Isolator, and only that component.

Upstream: https://github.com/SaFo-Lab/DRIFT (MIT, (c) 2026 Hao Li), commit
6fd3df4763fad77398c3c273c2a4fe14dd50f7ea — `prompts.py` INJECTION_DETECTION_PROMPT,
`DRIFTLLM.query` (the call site) and `DRIFTLLM.injection_isolate` (the masking).
Its other two components, the Secure Planner (`--build_constraints`) and the
Dynamic Validator (`--dynamic_validation`), are ablated; upstream switches the
three on independently, so `--injection_isolation` alone is a configuration
their code supports. That must be stated wherever this row appears.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from ams.core.prompts import PromptLibrary
from ams.llm import ChatRequest, LLMClient, Message
from ams.memory.base import MemorySystem, RetrievalResult, WriteResult

_DETECTED = re.compile(r"<detected_instructions>(.*?)</detected_instructions>", re.DOTALL)
MASK_LIMITATION = 1


def remove_sentence(p: str, t) -> str:
    """Upstream's mask function, verbatim in behaviour."""
    if type(t) is not str:
        t = ""
    words = t.split()
    escaped_words = [re.escape(word) for word in words]
    pattern = r"[\s\\]+".join(escaped_words)
    pattern = r"\s*" + pattern + r"\s*"
    return re.sub(str(pattern), " ", str(p), flags=re.DOTALL).strip()


def parse_detected(reply: str) -> list | None:
    """None when the reply has no tags (upstream then re-asks); else the list."""
    if "<detected_instructions>" not in reply:
        return None
    m = _DETECTED.search(reply)
    content = m.group(1).strip() if m else "[]"
    try:
        out = ast.literal_eval(content)
    except Exception:
        return []
    return out if isinstance(out, list) else []


def isolate(content: str, detected: list | None) -> tuple[bool, str, list[str]]:
    """Upstream `injection_isolate` for one tool message: (done, content, removed)."""
    if detected is None:
        return False, content, []
    if not detected:
        return True, content, []
    length = len(content)
    out, removed = content, []
    for _ in range(2):
        for item in detected:
            before, out = out, remove_sentence(out, item)
            if out != before:
                removed.append(str(item))
        if len(out) != length:
            break
    return len(out) != length, out, removed


@dataclass
class InjectionIsolator:
    client: LLMClient
    model: str
    prompts: PromptLibrary = field(default_factory=PromptLibrary)
    version: str = "v1"

    def _detect(self, query: str, content: str) -> list | None:
        obs = repr({"role": "tool", "content": content})
        reply = self.client.complete(
            ChatRequest(
                messages=(
                    Message(
                        "system", self.prompts.get("drift_isolator_system", self.version).template
                    ),
                    Message(
                        "user",
                        self.prompts.get("drift_isolator_user", self.version).render(
                            query=query, obs=obs
                        ),
                    ),
                ),
                model=self.model,
                temperature=0.0,
                max_tokens=10000,
            )
        ).text
        return parse_detected(reply)

    def __call__(self, query: str, content: str) -> tuple[str, list[str]]:
        """Masked content and the spans removed from it."""
        done, out, removed = isolate(content, self._detect(query, content))
        for _ in range(MASK_LIMITATION):
            if done:
                break
            done, out, more = isolate(out, self._detect(query, out))
            removed += more
        return out, removed


@dataclass
class IsolatedMemory:
    """Any `MemorySystem`, with the isolator in front of its write path."""

    inner: MemorySystem
    isolator: InjectionIsolator

    @property
    def name(self) -> str:
        return f"drift_isolated[{self.inner.name}]"

    def write(
        self, *, session_id: str, user_query: str, context: str, principal: str = "user"
    ) -> WriteResult:
        if not (context or "").strip():
            return self.inner.write(
                session_id=session_id, user_query=user_query, context=context, principal=principal
            )
        masked, removed = self.isolator(user_query or "", context)
        res = self.inner.write(
            session_id=session_id, user_query=user_query, context=masked, principal=principal
        )
        return WriteResult(written=res.written, discarded=tuple(removed) + res.discarded)

    def prefetch(self, *, user_query: str, context: str) -> None:
        """The same model calls `write` will make, without storing (cache warm-up)."""
        if (context or "").strip():
            context, _ = self.isolator(user_query or "", context)
        pf = getattr(self.inner, "prefetch", None)
        if pf:
            pf(user_query=user_query, context=context)

    def retrieve(
        self, *, session_id: str, query: str, k: int = 4, principal: str = "user"
    ) -> RetrievalResult:
        return self.inner.retrieve(session_id=session_id, query=query, k=k, principal=principal)

    def reset(self, session_id: str | None = None) -> None:
        self.inner.reset(session_id)
