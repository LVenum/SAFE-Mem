"""Export the data behind the paper's figures as CSV, straight from run outcomes."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

KINDS = ("description", "attribute", "preference", "rule", "permission_trust")

GOALS = {
    "recommendation_poisoning": "Recommendation",
    "trust_hijacking": "Trust hijacking",
    "bypass_verification": "Verification bypass",
    "unauthorized_action": "Unauthorized action",
    "data_exfiltration": "Data exfiltration",
    "policy_override": "Policy override",
    "credential_harvesting": "Credential harvesting",
}


def adversarial_rows(path: Path, system: str | None = None) -> list[dict]:
    """Attack-case outcome rows of one run, optionally of one `defence|memory` system."""
    with open(path) as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["is_adversarial"] == "1"
            and (system is None or f"{r['defence']}|{r['memory']}" == system)
        ]


def goal_by_kind(rows: list[dict]) -> list[dict]:
    """Share (%) of each goal's attack queries that the use classifier assigns to each kind."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["goal"]][r["required_type"]] += 1
    out = []
    for gi, goal in enumerate(GOALS):
        n = sum(counts[goal].values())
        for ki, kind in enumerate(KINDS):
            out.append({"goal": gi, "kind": ki, "share": round(100 * counts[goal][kind] / n, 1)})
    return out


def asr_by_goal(systems: dict[str, list[dict]]) -> list[dict]:
    """ASR (%) per goal for each named system; one row per goal, one column per system."""
    out = []
    for gi, (goal, label) in enumerate(GOALS.items()):
        row = {"goal": gi, "label": label}
        for name, rows in systems.items():
            rs = [r for r in rows if r["goal"] == goal]
            row["n"] = len(rs)
            row[name] = round(
                100 * sum(r["attack_succeeded"] in ("True", "1") for r in rs) / len(rs), 1
            )
        out.append(row)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stage_asr(summary: Path, systems: dict[str, str]) -> dict[str, list[float]]:
    """Write / retrieve / end-to-end ASR (%) of each named `defence|memory` system."""
    data = json.loads(summary.read_text())
    reports = data if isinstance(data, list) else data.get("reports", [data])
    by_name = {f"{r['defence']}|{r['memory']}": r for r in reports if "asr_end_to_end" in r}
    keys = ("asr_write", "asr_retrieve", "asr_end_to_end")
    return {label: [round(100 * by_name[s][k], 1) for k in keys] for label, s in systems.items()}


def stages_wide(stages: dict[str, list[float]]) -> list[dict]:
    """One row per stage, one column per system, as pgfplots reads it."""
    return [{"stage": i, **{name: v[i] for name, v in stages.items()}} for i in range(3)]


def use_shares(splits: dict[str, list[dict]]) -> list[dict]:
    """Share (%) of each split's queries per required kind; one row per split, a column per kind."""
    out = []
    for i, (label, rows) in enumerate(splits.items()):
        c = Counter(r["required_type"] for r in rows)
        n = sum(c.values())
        shares = {k: round(100 * c[k] / n, 1) for k in KINDS}
        labels = {f"{k}_lab": f"{v:.0f}" if v >= 10 else "" for k, v in shares.items()}
        mids, start = {}, 0.0
        for k, v in shares.items():
            mids[f"{k}_mid"] = round(start + v / 2, 2)
            start += v
        out.append({"y": i, "label": label, **shares, **labels, **mids})
    return out


def outcome_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


SPLITS = {
    "attack": "MPBench attacks",
    "warranted": "Requests to remember",
    "unwarranted": "Other benign cases",
}


def split_of(r: dict) -> str:
    """Which MPBench split an outcome row belongs to."""
    if r["is_adversarial"] == "1":
        return "attack"
    return "warranted" if r["benign_warranted"] in ("True", "1") else "unwarranted"


HEATMAP_ROWS = {
    "no_defence": "No defense",
    "a_memguard": "A-MemGuard",
    "drift": "DRIFT",
    "randec_distil": "RandEC",
    "llm_sanitiser": "LLM sanitizer",
    "memlineage[coarse": "MemLineage",
    "integrity_gate[all]": "TMA-NM",
    "safe_mem": "SAFE-Mem",
}


def goal_heatmap(runs: dict[str, Path]) -> list[dict]:
    """Long-format ASR (%) per defence and goal. `runs` maps a row key of
    HEATMAP_ROWS (matched as a prefix of the CSV's defence column) to an outcomes
    CSV; "drift" and "safe_mem" name whole runs whose defence column is no_defence."""
    csv.field_size_limit(10**9)
    out = []
    for key, path in runs.items():
        rows = adversarial_rows(path)
        if key not in ("drift", "safe_mem"):
            rows = [r for r in rows if r["defence"].startswith(key)]
        for gi, goal in enumerate(GOALS):
            rs = [r for r in rows if r["goal"] == goal]
            asr = 100 * sum(r["attack_succeeded"] in ("True", "1") for r in rs) / len(rs)
            row = list(HEATMAP_ROWS).index(key)
            out.append({"defense": HEATMAP_ROWS[key], "row": row, "goal": gi, "asr": round(asr, 1)})
    return out
