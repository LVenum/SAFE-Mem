"""Checking that the dataset cannot be solved without reading the text."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ams.data.mpbench import MPBenchCase


@dataclass(frozen=True, slots=True)
class FeatureLeak:
    feature: str
    separability: float
    verdict: str
    detail: str

    def line(self) -> str:
        return f"[{self.verdict:^7}] {self.feature:<22} {self.separability:.3f}  {self.detail}"


@dataclass
class LeakReport:
    features: list[FeatureLeak]
    n_poisoned: int
    n_clean: int

    @property
    def leaking(self) -> list[FeatureLeak]:
        return [f for f in self.features if f.verdict == "LEAK"]

    @property
    def ok(self) -> bool:
        return not self.leaking

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_adversarial_cases": self.n_poisoned,
            "n_benign_cases": self.n_clean,
            "features": [
                {
                    "feature": f.feature,
                    "separability": round(f.separability, 4),
                    "verdict": f.verdict,
                    "detail": f.detail,
                }
                for f in self.features
            ],
        }


def _balanced(tp: int, n_pos: int, tn: int, n_neg: int) -> float:
    """Mean of the true-positive and true-negative rates."""
    if not n_pos or not n_neg:
        return 0.5
    return 0.5 * (tp / n_pos + tn / n_neg)


def _separability(values_pos: list, values_neg: list) -> tuple[float, str]:
    """Best balanced accuracy of a single equality test over the value set."""
    if not values_pos or not values_neg:
        return 0.5, "one class is empty"
    n_pos, n_neg = len(values_pos), len(values_neg)
    best, best_val = 0.5, None
    for candidate in set(values_pos) | set(values_neg):
        tp = sum(1 for v in values_pos if v == candidate)
        tn = sum(1 for v in values_neg if v != candidate)
        acc = max(_balanced(tp, n_pos, tn, n_neg), _balanced(n_pos - tp, n_pos, n_neg - tn, n_neg))
        if acc > best:
            best, best_val = acc, candidate
    if best_val is None:
        return best, "no single test beats chance"
    return best, f"best single test: value == {best_val!r}"


def _numeric_separability(pos: list[float], neg: list[float]) -> tuple[float, str]:
    """Best balanced accuracy of a single threshold, in either direction."""
    if not pos or not neg:
        return 0.5, "one class is empty"
    n_pos, n_neg = len(pos), len(neg)
    best, best_t = 0.5, None
    for t in sorted(set(pos) | set(neg)):
        above = sum(1 for v in pos if v >= t)
        below_neg = sum(1 for v in neg if v < t)
        acc = max(
            _balanced(above, n_pos, below_neg, n_neg),
            _balanced(n_pos - above, n_pos, n_neg - below_neg, n_neg),
        )
        if acc > best:
            best, best_t = acc, t
    return best, f"best threshold: {best_t}"


def _identifier_leak(pos: list[MPBenchCase], neg: list[MPBenchCase]) -> tuple[float, str]:
    """Does any token in the identifiers mark out one class?"""

    def tokens(items: list[MPBenchCase]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for cid in {c.cid for c in items}:
            for tok in re.split(r"[^A-Za-z]+", cid):
                if len(tok) >= 4:
                    counts[tok] = counts.get(tok, 0) + 1
        return counts

    tp, tn = tokens(pos), tokens(neg)
    marks_pos = {t for t, c in tp.items() if c >= 2 and t not in tn}
    marks_neg = {t for t, c in tn.items() if c >= 2 and t not in tp}
    marks = marks_pos | marks_neg
    if marks:
        return 1.0, f"class-identifying token(s) in identifiers: {sorted(marks)[:5]}"
    return 0.5, "identifiers are opaque; no token marks out a class"


FEATURES: dict[str, Callable[[MPBenchCase], object]] = {
    "domain": lambda c: c.domain,
    "attack_signal": lambda c: c.attack_signal,
    "n_context_lines": lambda c: len(c.context.splitlines()),
    "has_expected_memory": lambda c: bool(c.expected_memory),
    "query_len_bucket": lambda c: len(c.user_query) // 50,
}


def audit_leakage(cases: Sequence[MPBenchCase], *, leak_threshold: float = 0.85) -> LeakReport:
    """Can any observable field separate adversarial cases from benign ones?"""
    pos = [c for c in cases if c.is_adversarial]
    neg = [c for c in cases if not c.is_adversarial]

    features: list[FeatureLeak] = []
    for name, fn in FEATURES.items():
        if name == "cid":
            sep, detail = _identifier_leak(pos, neg)
        else:
            sep, detail = _separability([fn(m) for m in pos], [fn(m) for m in neg])
        features.append(
            FeatureLeak(
                feature=name,
                separability=sep,
                verdict="LEAK" if sep >= leak_threshold else "ok",
                detail=detail,
            )
        )

    sep, detail = _numeric_separability(
        [float(len(c.context)) for c in pos], [float(len(c.context)) for c in neg]
    )
    features.append(
        FeatureLeak(
            feature="text_length",
            separability=sep,
            verdict="watch" if sep >= leak_threshold else "ok",
            detail=detail + " (content property, not bookkeeping)",
        )
    )

    return LeakReport(features=features, n_poisoned=len(pos), n_clean=len(neg))
