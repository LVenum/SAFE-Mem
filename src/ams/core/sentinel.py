"""Sentinel rendering of retrieved memory, after MemLineage (arXiv 2605.14421 §3.6)."""

from __future__ import annotations

import re

_FORGE = re.compile(r"\[\s*(BEGIN|END)\s+MEMORY", re.IGNORECASE)
_HEADER = re.compile(r"\[BEGIN MEMORY ((?:\w+=[^\s\]]+ ?)+)\]")
_ATTR = re.compile(r"(\w+)=([^\s\]]+)")


def defang(text: str) -> str:
    """Make sentinel literals in content unparseable as sentinels."""
    return _FORGE.sub(lambda m: f"(memory-{m.group(1).lower()})", text)


def render(entry_id: str, text: str, **attrs: str) -> str:
    for k, v in attrs.items():
        if not re.fullmatch(r"[^\s\]]+", str(v)):
            raise ValueError(f"sentinel attribute {k}={v!r} must be one token without ']'")
    head = " ".join([f"entry_id={entry_id}", *(f"{k}={v}" for k, v in attrs.items())])
    return f"[BEGIN MEMORY {head}]\n{defang(text)}\n[END MEMORY]"


def read(prompt: str) -> list[dict[str, str]]:
    """Every sentinel header in `prompt`, in order, as attribute dicts."""
    return [dict(_ATTR.findall(m.group(1))) for m in _HEADER.finditer(prompt)]


_BLOCK = re.compile(r"\[BEGIN MEMORY ((?:\w+=[^\s\]]+ ?)+)\]\n(.*?)\n\[END MEMORY\]", re.DOTALL)


def blocks(prompt: str) -> list[tuple[dict[str, str], str]]:
    """Every rendered entry in `prompt` as (attributes, body)."""
    return [(dict(_ATTR.findall(m.group(1))), m.group(2)) for m in _BLOCK.finditer(prompt)]
