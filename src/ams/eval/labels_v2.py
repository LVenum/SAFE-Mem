"""Second labelling round (2026-10-11, after review): 150 more use-type items and
200 answer-judge items. Sheets are blind; model and judge labels live in a key."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from ams.eval.labels import KINDS, _agree, _pick, _read


def build_use(
    *, cases: dict, mp: dict[str, dict], lc: dict[str, dict], exclude: set[str], seed: int = 1
):
    """120 MPBench queries (24 per kind the first backbone gave) + 30 LoCoMo questions."""
    rng = random.Random(seed)
    first = next(iter(mp))
    rows = [
        r
        for r in mp[first].values()
        if cases[r["cid"]].retrieval_query.strip() and r["cid"] not in exclude
    ]
    a1 = _pick(rows, lambda r: r["required_type"], {k: 24 for k in KINDS}, rng)
    lrows = [r for r in lc[next(iter(lc))].values() if r["qid"] not in exclude]
    a2 = _pick(lrows, lambda r: r["required_type"] == "description", {True: 15, False: 15}, rng)
    items = [(cases[r["cid"]].retrieval_query, r["cid"], "mpbench") for r in a1] + [
        (r["question"], r["qid"], "locomo") for r in a2
    ]
    rng.shuffle(items)
    sheet, key = [], {}
    for n, (text, ref, ds) in enumerate(items, start=1):
        i = f"U{n:03d}"
        src = mp if ds == "mpbench" else lc
        sheet.append({"id": i, "question": text, "kind": "", "notes": ""})
        key[i] = {
            "part": "A",
            "dataset": ds,
            "ref": ref,
            "model": {bb: src[bb][ref]["required_type"] for bb in src if ref in src[bb]},
        }
    return sheet, key


def build_judge(*, cases: dict, runs: dict[str, dict], per: int = 25, seed: int = 2):
    """For each run: `per` attack cases the judge scored as success and `per` as failure."""
    rng = random.Random(seed)
    picked = []
    for name, rows in runs.items():
        att = [
            r for r in rows.values() if r["is_adversarial"] == "1" and r.get("answer", "").strip()
        ]
        for want in ("1", "0"):
            pool = [r for r in att if r["attack_succeeded"] == want]
            picked += [(name, r) for r in rng.sample(pool, min(per, len(pool)))]
    rng.shuffle(picked)
    sheet, key = [], {}
    for n, (name, r) in enumerate(picked, start=1):
        i = f"J{n:03d}"
        c = cases[r["cid"]]
        sheet.append(
            {
                "id": i,
                "question": c.retrieval_query,
                "injected_memory": c.expected_memory,
                "answer": r["answer"],
                "acts": "",
                "notes": "",
            }
        )
        key[i] = {"run": name, "cid": r["cid"], "judge": int(r["attack_succeeded"])}
    return sheet, key


def write(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def score_judge(sheet: Path, key: dict) -> dict:
    lab = _read(sheet, "id")
    yes = {"yes", "y", "1", "true"}
    pairs = [
        ("1" if lab[i]["acts"].strip().lower() in yes else "0", str(k["judge"]))
        for i, k in key.items()
        if lab.get(i, {}).get("acts", "").strip()
    ]
    return {
        **_agree(pairs),
        "judge_false_positive": sum(h == "0" and m == "1" for h, m in pairs),
        "judge_false_negative": sum(h == "1" and m == "0" for h, m in pairs),
    }
