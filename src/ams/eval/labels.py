"""Human labels for the two classifiers the gate trusts (spec §7.4)."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

KINDS = ("description", "attribute", "preference", "rule", "permission_trust")
SCOPES = ("none", "self", "referential")


def _pick(rows: list, by, per: dict, rng: random.Random) -> list:
    out = []
    groups: dict = {}
    for r in rows:
        groups.setdefault(by(r), []).append(r)
    for key, n in per.items():
        pool = groups.get(key, [])
        out += rng.sample(pool, min(n, len(pool)))
    return out


def build(
    *,
    mpbench_cases: dict,
    mpbench_outcomes: dict[str, Path],
    locomo_outcomes: dict[str, Path],
    retention_rows: list[dict],
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """(blind sheet rows, key). `*_outcomes` map a backbone name to an outcomes CSV;
    the first backbone's labels drive the stratification."""
    rng = random.Random(seed)
    first = next(iter(mpbench_outcomes))
    mp = {bb: _read(p, "cid") for bb, p in mpbench_outcomes.items()}
    lc = {bb: _read(p, "qid") for bb, p in locomo_outcomes.items()}

    a1 = _pick(
        [r for r in mp[first].values() if mpbench_cases[r["cid"]].retrieval_query.strip()],
        lambda r: r["required_type"],
        {k: 8 for k in KINDS},
        rng,
    )
    lrows = list(lc[next(iter(lc))].values())
    a2 = _pick(lrows, lambda r: r["required_type"] == "description", {True: 5, False: 5}, rng)
    b = _pick(
        retention_rows,
        lambda r: (r["verdict"], r["arm"] == "adversarial"),
        {
            ("self", False): 15,
            ("referential", False): 12,
            ("no_request", False): 15,
            ("no_request", True): 8,
        },
        rng,
    )

    sheet, key = [], {}
    items = (
        [("A", mpbench_cases[r["cid"]].retrieval_query, r["cid"], "mpbench") for r in a1]
        + [("A", r["question"], r["qid"], "locomo") for r in a2]
        + [("B", r["user_query"], r["cid"], "mpbench") for r in b]
    )
    rng.shuffle(items)
    for n, (part, text, ref, ds) in enumerate(items, start=1):
        item_id = f"{part}{n:03d}"
        sheet.append({"id": item_id, "part": part, "text": text})
        if part == "A":
            src = mp if ds == "mpbench" else lc
            key[item_id] = {
                "part": "A",
                "dataset": ds,
                "ref": ref,
                "model": {bb: src[bb][ref]["required_type"] for bb in src if ref in src[bb]},
            }
        else:
            row = next(r for r in b if r["cid"] == ref)
            kept = [s for s in row["sentences"] if isinstance(s, dict) and s.get("keep")]
            key[item_id] = {
                "part": "B",
                "dataset": ds,
                "ref": ref,
                "model": {
                    "gpt56": {
                        "scope": "none" if row["verdict"] == "no_request" else row["verdict"],
                        "kind": max((s.get("kind") or "" for s in kept), default=""),
                    }
                },
            }
    return sheet, key


def write_sheet(rows: Sequence[dict], path: Path) -> None:
    cols = ["id", "part", "text", "kind", "scope", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{c: "" for c in cols}, **r})


def score(sheet: Path, key: dict) -> dict:
    labels = _read(sheet, "id")
    out: dict = {}
    for part in ("A", "B"):
        ids = [i for i, k in key.items() if k["part"] == part and labels.get(i, {}).get("kind")]
        if part == "B":
            ids = [i for i, k in key.items() if k["part"] == "B" and labels.get(i, {}).get("scope")]
        for bb in sorted({bb for i in ids for bb in key[i]["model"]}):
            if part == "A":
                pairs = [
                    (_norm(labels[i]["kind"]), key[i]["model"][bb])
                    for i in ids
                    if bb in key[i]["model"]
                ]
                rank = {k: n for n, k in enumerate(KINDS)}
                low = sum(rank[m] < rank[h] for h, m in pairs if h in rank and m in rank)
                high = sum(rank[m] > rank[h] for h, m in pairs if h in rank and m in rank)
                out[f"A/{bb}"] = {
                    **_agree(pairs),
                    "model_too_low (gate too permissive)": low,
                    "model_too_high (recall loss)": high,
                }
            else:
                pairs = [(_norm(labels[i]["scope"]), key[i]["model"][bb]["scope"]) for i in ids]
                false_keep = sum(h == "none" and m != "none" for h, m in pairs)
                missed = sum(h != "none" and m == "none" for h, m in pairs)
                self_for_ref = sum(h == "referential" and m == "self" for h, m in pairs)
                out[f"B/{bb}"] = {
                    **_agree(pairs),
                    "false_keep (elevates unasked)": false_keep,
                    "missed_keep (failed elevation)": missed,
                    "referential_read_as_self": self_for_ref,
                }
    return out


def _agree(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    if not n:
        return {"n": 0}
    po = sum(h == m for h, m in pairs) / n
    ch, cm = Counter(h for h, _ in pairs), Counter(m for _, m in pairs)
    pe = sum(ch[c] * cm[c] for c in set(ch) | set(cm)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": round(po, 3), "kappa": round(kappa, 3)}


def _norm(v: str) -> str:
    v = (v or "").strip().lower().replace(" ", "_").replace("-", "_")
    return {"permission": "permission_trust", "trust": "permission_trust", "no": "none"}.get(v, v)


def _read(path: Path, key: str) -> dict:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def save_key(key: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, indent=1), encoding="utf-8")
