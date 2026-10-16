"""Uncertainty and breakdowns for MPBench ASR, from per-case outcome CSVs."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


def attack_rows(csv_path: Path, defence: str) -> list[dict]:
    """Scored attack cases of one defence (cases without a query are not in the CSV)."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            r for r in csv.DictReader(f) if r["defence"] == defence and r["is_adversarial"] == "1"
        ]


def asr_ci(rows: list[dict], query_of: dict[str, str], *, b: int = 2000, seed: int = 0):
    """(point, case CI, query-clustered CI), percentages, 95%."""
    y = [int(r["attack_succeeded"]) for r in rows]
    point = 100 * sum(y) / len(y)
    rng = random.Random(seed)

    def pct(v):
        v = sorted(v)
        return round(v[int(0.025 * len(v))], 1), round(v[int(0.975 * len(v)) - 1], 1)

    case = [100 * sum(rng.choices(y, k=len(y))) / len(y) for _ in range(b)]
    groups: dict[str, list[int]] = defaultdict(list)
    for r, v in zip(rows, y, strict=True):
        groups[query_of.get(r["cid"], r["cid"])].append(v)
    g = list(groups.values())
    clus = []
    for _ in range(b):
        pick = rng.choices(g, k=len(g))
        n = sum(len(x) for x in pick)
        clus.append(100 * sum(sum(x) for x in pick) / n)
    return round(point, 1), pct(case), pct(clus), len(y), len(groups)


def by_field(rows: list[dict], field: str) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        out[r[field]][0] += int(r["attack_succeeded"])
        out[r[field]][1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


def locomo_scores(csv_path: Path, system_prefix: str) -> dict[str, tuple[str, float]]:
    """qid -> (conversation, judge correct) over categories 1-4 for one system."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {
            r["qid"]: (r["sample_id"], float(r["judge_correct"] in ("1", "True", "true")))
            for r in csv.DictReader(f)
            if r["system"].startswith(system_prefix) and r["category"] in ("1", "2", "3", "4")
        }


def paired_conversation_ci(base: dict, ours: dict, *, b: int = 2000, seed: int = 0):
    """Gain in percentage points with a 95% CI that resamples whole conversations."""
    conv: dict[str, list[float]] = defaultdict(list)
    for q, (c, y) in ours.items():
        if q in base:
            conv[c].append(y - base[q][1])
    groups = list(conv.values())
    point = 100 * sum(sum(g) for g in groups) / sum(len(g) for g in groups)
    rng = random.Random(seed)
    boot = []
    for _ in range(b):
        pick = rng.choices(groups, k=len(groups))
        boot.append(100 * sum(sum(g) for g in pick) / sum(len(g) for g in pick))
    boot.sort()
    return (
        round(point, 1),
        (round(boot[int(0.025 * b)], 1), round(boot[int(0.975 * b) - 1], 1)),
        len(groups),
    )
