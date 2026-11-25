"""Where does admission cost accuracy? Offline attribution over existing outcome CSVs.

Rebuttal analysis (2026-11-24). Reads only per-case outcomes; calls no model.

LoCoMo: compare SAFE-Mem against a reference run of the same graph (the
all-principal control, or confine) question by question, split by the use the
classifier assigned. At use = description both runs admit everything, so that
bucket's difference is the write-side/noise floor, not admission.

MPBench recall: classify each failed warranted case by the first stage that
lost the user's own memory.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

USES = ["description", "attribute", "preference", "rule", "permission_trust"]
_RANK = {u: i for i, u in enumerate(USES)}
_ENTRY = re.compile(r"entry_id=(\w+) usable=(\w+)")


def read_rows(path: Path | str, **match: str) -> list[dict]:
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if all(r.get(k) == v for k, v in match.items())]


def locomo_compare(ours: list[dict], ref: list[dict]) -> dict[str, dict]:
    """Per classified use (from `ours`): n, ours/ref judge accuracy, lost, gained.
    Only judged categories 1-4; joined on qid."""
    ref_by = {r["qid"]: r for r in ref if r["judge_correct"] in ("0", "1")}
    out: dict[str, dict] = defaultdict(lambda: dict(n=0, ours=0, ref=0, lost=0, gained=0))
    for r in ours:
        if r["judge_correct"] not in ("0", "1") or r["qid"] not in ref_by:
            continue
        a, b = int(r["judge_correct"]), int(ref_by[r["qid"]]["judge_correct"])
        for key in (r["required_type"] or "none", "all"):
            s = out[key]
            s["n"] += 1
            s["ours"] += a
            s["ref"] += b
            s["lost"] += b and not a
            s["gained"] += a and not b
    return dict(out)


def entries(texts: str) -> dict[str, str]:
    """entry_id -> usable level, from a rendered tuple of memory blocks."""
    return dict(_ENTRY.findall(texts or ""))


def recall_failure_stage(row: dict) -> str:
    """First stage at which a failed warranted case lost its own memory.

    no_elevation       nothing of the user's request was kept above description
    withheld_by_kind   kept, but its best level is below the use the query needs
    not_retrieved      admissible, but ranked out of the top k
    retrieved          the user's own entry reached the context; answer/judge miss
    """
    own = {i: u for i, u in entries(row["written_texts"]).items() if u != "description"}
    if not own:
        return "no_elevation"
    need = _RANK.get(row["required_type"], 0)
    if max(_RANK[u] for u in own.values()) < need:
        return "withheld_by_kind"
    if not own.keys() & entries(row["retrieved_texts"]).keys():
        return "not_retrieved"
    return "retrieved"


def recall_failures(rows: list[dict], baseline: list[dict] | None = None) -> dict:
    """Stage counts for failed warranted cases; with a baseline, also for the
    cases the baseline answers correctly (the cost attributable to SAFE-Mem)."""
    warranted = [r for r in rows if r["benign_warranted"] == "1"]
    failed = [r for r in warranted if r["answer_correct"] != "1"]
    res = {
        "n_warranted": len(warranted),
        "n_failed": len(failed),
        "stages": Counter(recall_failure_stage(r) for r in failed),
        "need_of_kind_withheld": Counter(
            r["required_type"] for r in failed if recall_failure_stage(r) == "withheld_by_kind"
        ),
    }
    if baseline is not None:
        ok = {
            r["cid"]
            for r in baseline
            if r["benign_warranted"] == "1" and r["answer_correct"] == "1"
        }
        lost = [r for r in failed if r["cid"] in ok]
        res["n_lost_vs_baseline"] = len(lost)
        res["stages_lost_vs_baseline"] = Counter(recall_failure_stage(r) for r in lost)
        ours_ok = {r["cid"] for r in warranted if r["answer_correct"] == "1"}
        res["n_gained_vs_baseline"] = len(ours_ok - ok)
    return res


def use_error_direction(labels: dict[str, str], key: dict, backbone: str) -> dict:
    """Split use-classifier disagreements with human labels by direction.

    `labels` maps item id -> human kind; `key` is a labels_v2 use key. Under
    ("model below human") admits external claims as descriptions: the security
    side. Over ("model above") withholds legitimate memory: the utility side.
    Boundary = description vs any higher kind, the only split that matters for
    external content. Grouped by dataset and, for MPBench, attack vs benign."""
    out: dict[str, Counter] = defaultdict(Counter)
    for i, k in key.items():
        h, m = labels.get(i, ""), k["model"].get(backbone)
        if h not in _RANK or m not in _RANK:
            continue
        group = k["dataset"]
        if group == "mpbench":
            group += ":attack" if k["ref"].startswith("ADV") else ":benign"
        d = _RANK[m] - _RANK[h]
        for g in (group, "all"):
            c = out[g]
            c["n"] += 1
            c["exact"] += d == 0
            c["under"] += d < 0
            c["over"] += d > 0
            c["under_by_1"] += d == -1
            c["boundary_under"] += h != "description" and m == "description"
            c["boundary_over"] += h == "description" and m != "description"
    return dict(out)
