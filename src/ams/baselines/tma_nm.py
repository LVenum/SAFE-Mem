"""TMA-NM, run through its authors' own monitor.

Upstream: https://github.com/yedidel/mem-inv-bench (MIT), `v2/code/tma.py`,
vendored read-only by `scripts/fetch_third_party.sh` at a pinned commit. v2 is
the artifact for their current paper and supersedes the root directory.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from ams.baselines.base import DefenceResult
from ams.memory.base import MemoryEntry


def upstream_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / "third_party" / "mem-inv-bench" / "v2" / "code" / "tma.py"


def _load_upstream(root: Path | None = None):
    """Import their `tma.py` by path. Pure stdlib, so nothing heavy comes with it."""
    path = upstream_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"mem-inv-bench not vendored at {path}. Run scripts/fetch_third_party.sh first."
        )
    if "tma_upstream" in sys.modules:
        return sys.modules["tma_upstream"]
    spec = importlib.util.spec_from_file_location("tma_upstream", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tma_upstream"] = mod
    spec.loader.exec_module(mod)
    return mod


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(
    r"https?://[^\s)\]>\"']+|\b[\w-]+\.(?:com|org|net|io|edu|gov|dev|ai)\b(?:/[^\s)\]>\"']*)?"
)
_IDENT = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{5,}\b")


def security_values(text: str) -> list[str]:
    """Emails, URLs/domains and identifier-like tokens in `text`, in order."""
    out: list[str] = []
    for rx in (_EMAIL, _URL, _IDENT):
        for m in rx.finditer(text):
            v = m.group(0).rstrip(".,;:")
            if v.isdigit() or any(v in seen for seen in out):
                continue
            out.append(v)
    return out


class TmaNm:
    """Origin-bound authority over retrieved entries, delegating to their Monitor."""

    def __init__(self, *, mode: str = "values", upstream_root: Path | None = None) -> None:
        if mode not in ("values", "strict", "full"):
            raise ValueError(f"mode must be 'values', 'strict' or 'full', got {mode!r}")
        self._mode = mode
        self._tma = _load_upstream(upstream_root)

    @property
    def name(self) -> str:
        return f"tma_nm[{self._mode}]"

    def _monitor(self):
        t = self._tma
        chans = {
            "user": t.Channel("user", t.Origin.USER, "principal", b"k-user"),
            "external": t.Channel("external", t.Origin.UNTRUSTED, "public-content", b"k-ext"),
        }
        reg = t.Registry()
        for c in chans.values():
            reg.register(c)
        return t.Monitor(reg, k=2), chans

    def _filter_full(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        """Their mechanism with M3 elevation reachable, not only the act class."""
        t = self._tma
        items = list(memories)
        if not items:
            return DefenceResult(kept=(), reason="empty pool")

        reg = t.Registry()
        user = t.Channel("user", t.Origin.USER, "principal", b"k-user")
        reg.register(user)
        chans: dict[str, object] = {}
        for m in items:
            key = str(m.metadata.get("source_id") or m.metadata.get("source_context", m.text))[:64]
            if key not in chans:
                trusted = bool(m.metadata.get("authenticated_tool"))
                ch = t.Channel(
                    f"src{len(chans)}",
                    t.Origin.TRUSTED if trusted else t.Origin.UNTRUSTED,
                    f"domain{len(chans)}",
                    f"k{len(chans)}".encode(),
                )
                reg.register(ch)
                chans[key] = ch
        mon = t.Monitor(reg, k=2)

        for ut in {query} | {
            str(m.metadata.get("user_query", "")) for m in items if m.metadata.get("user_query")
        }:
            mon.ingest(user.send(ut))
            for v in security_values(ut):
                mon.endorse(v, "principal")
        mon.tick_epoch()

        tma_items = []
        for m in items:
            key = str(m.metadata.get("source_id") or m.metadata.get("source_context", m.text))[:64]
            tma_items.append(mon.ingest(chans[key].send(m.text)))

        kept, flagged, verdicts, n_vouched = [], [], [], 0
        for m, it in zip(items, tma_items, strict=True):
            denied: list[str] = []
            for v in security_values(m.text):
                if mon.attribute(v)[0]:
                    continue
                vouch = mon.independent_vouchers(v, tma_items)
                allowed, _why = mon.authorize(v, tma_items)
                if allowed and vouch:
                    n_vouched += 1
                if not allowed:
                    denied.append(v)
            ok = not denied and int(it.act) >= int(t.ActClass.INFORM)
            (kept if ok else flagged).append(m)
            verdicts.append(
                {
                    "index": len(verdicts),
                    "kept": ok,
                    "origin": t.Origin(it.origin).name,
                    "why": "ok" if ok else "uncorroborated: " + ", ".join(denied),
                    "denied_values": denied,
                }
            )
        return DefenceResult(
            kept=tuple(kept),
            flagged=tuple(flagged),
            reason="tma_nm full (M1-M3)",
            detail={
                "n_flagged": len(flagged),
                "verdicts": verdicts,
                "n_vouched": n_vouched,
                "n_channels": len(chans),
                "log_ok": mon.verify_log(),
            },
        )

    def filter(self, query: str, memories: Sequence[MemoryEntry]) -> DefenceResult:
        if self._mode == "full":
            return self._filter_full(query, memories)
        items = list(memories)
        if not items:
            return DefenceResult(kept=(), reason="empty pool")
        t = self._tma
        mon, chans = self._monitor()

        user_texts = {query} | {
            str(m.metadata.get("user_query", "")) for m in items if m.metadata.get("user_query")
        }
        for ut in user_texts:
            mon.ingest(chans["user"].send(ut))
            for v in security_values(ut):
                mon.endorse(v, "principal")
        mon.tick_epoch()

        tma_items = []
        for m in items:
            src = mon.ingest(chans["external"].send(str(m.metadata.get("source_context", m.text))))
            tma_items.append(mon.derive(m.text, [src], via="agent"))

        kept, flagged, verdicts = [], [], []
        for m, it in zip(items, tma_items, strict=True):
            if self._mode == "strict":
                ok = int(it.act) >= int(t.ActClass.INFORM)
                why = f"origin={t.Origin(it.origin).name} act={t.ActClass(it.act).name}"
                denied: list[str] = []
            else:
                denied = []
                for v in security_values(m.text):
                    permitted, _ = mon.attribute(v)
                    if not permitted:
                        allowed, _why = mon.authorize(v, [it])
                        if not allowed:
                            denied.append(v)
                ok = not denied
                why = (
                    "no unendorsed security-relevant value"
                    if ok
                    else "untrusted-labelled value, uncorroborated: " + ", ".join(denied)
                )
            (kept if ok else flagged).append(m)
            verdicts.append(
                {
                    "index": len(verdicts),
                    "kept": ok,
                    "origin": t.Origin(it.origin).name,
                    "why": why,
                    "denied_values": denied,
                }
            )
        return DefenceResult(
            kept=tuple(kept),
            flagged=tuple(flagged),
            reason=f"tma_nm {self._mode}",
            detail={"n_flagged": len(flagged), "verdicts": verdicts, "log_ok": mon.verify_log()},
        )
