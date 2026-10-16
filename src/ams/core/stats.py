"""Vote aggregation, margins, and intervals."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vote:
    """The outcome of N samples over a discrete action space."""

    action: str | None
    counts: dict[str, int]
    n_valid: int
    n_failed: int

    @property
    def n(self) -> int:
        return self.n_valid + self.n_failed

    @property
    def top_freq(self) -> float:
        if not self.n_valid or self.action is None:
            return 0.0
        return self.counts[self.action] / self.n_valid

    @property
    def margin(self) -> float:
        """Top frequency minus runner-up, over valid samples."""
        if self.n_valid == 0:
            return 0.0
        ordered = sorted(self.counts.values(), reverse=True)
        top = ordered[0]
        second = ordered[1] if len(ordered) > 1 else 0
        return (top - second) / self.n_valid


def majority(samples: list[str | None]) -> Vote:
    """Majority action over samples, with unparseable replies excluded."""
    valid = [s for s in samples if s is not None]
    counts = Counter(valid)
    if not counts:
        return Vote(None, {}, 0, len(samples))
    top = max(sorted(counts), key=lambda a: counts[a])
    return Vote(top, dict(counts), len(valid), len(samples) - len(valid))


def deviated(baseline: Vote, other: Vote) -> bool:
    """The deviation test at the heart of the definition: a(v) != a_0."""
    if baseline.action is None or other.action is None:
        return False
    return other.action != baseline.action


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(successes: int, n: int) -> float:
    return successes / n if n else 0.0
