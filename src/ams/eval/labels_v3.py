"""Third labelling round (rebuttal, 2026-11-24): claim kind and question target.

Part A, 50 claims: the kind the extractor gave each claim, which the earlier rounds
never checked. 30 LoCoMo claims (half from the principal's own turns, half from the
other speaker) and 20 claims from MPBench requests to remember. Only principal and
counterpart claims are sampled: an external claim's kind cannot change what it may
serve (min with description authority), so its errors matter to nobody.

Part B, 30 LoCoMo questions: who the question is about, t(q), as read by the use
classifier. 20 questions classified above description (where t(q) reaches the gate)
and 10 at description.

Sheets are blind; model labels live in a key.
"""

from __future__ import annotations

import random

from ams.core.lattice import Rung
from ams.core.standing import SpanRole
from ams.eval.labels import KINDS, _agree, _norm, _read


def claims_of(memory, session_id: str) -> list[dict]:
    """Every claim of one store with the text of the source it was extracted from."""
    g = memory._graph(session_id)
    out = []
    for c in g.claims():
        src = g.source(c.source_id)
        out.append(
            {
                "claim": c.text,
                "kind": c.claim_type.name.lower(),
                "role": str(src.role),
                "source": src.text,
            }
        )
    return out


def build_claims(locomo: list[dict], mpbench: list[dict], *, seed: int = 3):
    """`locomo` / `mpbench`: claims_of() rows, each with a `ref` added by the caller."""
    rng = random.Random(seed)
    mine = [r for r in locomo if r["role"] == SpanRole.PRINCIPAL_TURN]
    theirs = [r for r in locomo if r["role"] == SpanRole.COUNTERPART_TURN]
    asked = [r for r in mpbench if r["role"] == SpanRole.PRINCIPAL_TURN]
    picked = [
        ("locomo", r)
        for r in rng.sample(mine, min(15, len(mine))) + rng.sample(theirs, min(15, len(theirs)))
    ] + [("mpbench", r) for r in rng.sample(asked, min(20, len(asked)))]
    rng.shuffle(picked)
    sheet, key = [], {}
    for n, (ds, r) in enumerate(picked, start=1):
        i = f"K{n:03d}"
        sheet.append({"id": i, "source": r["source"], "claim": r["claim"], "kind": "", "notes": ""})
        key[i] = {"dataset": ds, "ref": r["ref"], "role": r["role"], "model": r["kind"]}
    return sheet, key


def build_targets(questions: list[dict], about, *, seed: int = 4):
    """`questions`: dicts with qid, question, speakers, principal, required.
    `about(question)` gives the classifier's target; called only for sampled rows."""
    rng = random.Random(seed)
    hi = [q for q in questions if q["required"] != "description"]
    lo = [q for q in questions if q["required"] == "description"]
    picked = rng.sample(hi, min(20, len(hi))) + rng.sample(lo, min(10, len(lo)))
    rng.shuffle(picked)
    sheet, key = [], {}
    for n, q in enumerate(picked, start=1):
        i = f"T{n:03d}"
        sheet.append(
            {
                "id": i,
                "speakers": " & ".join(q["speakers"]),
                "asker": q["principal"],
                "question": q["question"],
                "about": "",
                "notes": "",
            }
        )
        key[i] = {
            **{k: q[k] for k in ("qid", "principal", "required")},
            "about": about(q["question"]),
        }
    return sheet, key


def _person(v: str, asker: str) -> str:
    v = (v or "").strip().lower()
    return asker.lower() if v in ("", "me", "asker", "none") else v


def score_claims(sheet, key: dict) -> dict:
    lab = _read(sheet, "id")
    rank = {k: n for n, k in enumerate(KINDS)}
    pairs, under, over = [], 0, 0
    by_role: dict[str, list] = {}
    for i, k in key.items():
        h = _norm(lab.get(i, {}).get("kind", ""))
        if h not in rank:
            continue
        pairs.append((h, k["model"]))
        by_role.setdefault(f"{k['dataset']}:{k['role']}", []).append((h, k["model"]))
        under += rank[k["model"]] < rank[h]
        over += rank[k["model"]] > rank[h]
    return {
        **_agree(pairs),
        "model_below_human": under,
        "model_above_human": over,
        "by_source": {g: _agree(p) for g, p in by_role.items()},
    }


def score_targets(sheet, key: dict) -> dict:
    lab = _read(sheet, "id")
    pairs, hi = [], []
    for i, k in key.items():
        h = lab.get(i, {}).get("about", "").strip()
        if h:
            a = k["principal"]
            pair = (_person(h, a), _person(k["about"], a))
            pairs.append(pair)
            if k["required"] != "description":
                hi.append(pair)
    return {**_agree(pairs), "above_description": _agree(hi)}


assert Rung.DESCRIPTION.name.lower() == KINDS[0]
