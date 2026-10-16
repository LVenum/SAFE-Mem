#!/usr/bin/env python
"""Does a trigger actually pull the poisoned records into the top-k?  No LLM, no cost."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, DPRContextEncoder

ROOT = Path(__file__).resolve().parents[1]
AM = ROOT / "third_party" / "AMemGuard"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "facebook/dpr-ctx_encoder-single-nq-base"


def embed(texts: list[str], model, tok) -> torch.Tensor:
    out = []
    for t in texts:
        x = tok(t, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        with torch.no_grad():
            out.append(model(x["input_ids"].to(DEV), x["attention_mask"].to(DEV)).pooler_output)
    return torch.cat(out, 0)


def top_hits(queries: torch.Tensor, db: torch.Tensor, n_clean: int, k: int) -> list[int]:
    """Per query: how many of the top-k rows are poisoned (rows >= n_clean)."""
    sims = torch.nn.functional.cosine_similarity(queries.unsqueeze(1), db.unsqueeze(0), dim=2)
    top = sims.topk(k, dim=1).indices
    return [(row >= n_clean).sum().item() for row in top]


def load_react(model, tok):
    """9,251 passages (cached matrix, dict order is deterministic) + 20 poisoned demos."""
    base = AM / "ReAct" / "database"
    cache = (
        base / "embeddings" / "strategyqa_database_embeddings_dpr-ctx_encoder-single-nq-base.pkl"
    )
    if cache.exists():
        db = torch.tensor(pickle.load(cache.open("rb")), dtype=torch.float32).to(DEV)
        db = db.squeeze(1) if db.dim() == 3 else db
    else:
        paras = json.load((base / "strategyqa_train_paragraphs.json").open())
        db = embed([paras[p]["content"] for p in paras], model, tok)
    train = json.load((base / "strategyqa_train.json").open())[:20]
    dev = json.load((base / "strategyqa" / "strategyqa_dev.json").open())
    return db, [t["question"] for t in train], [d["question"] for d in dev], 5


def load_ehr(model, tok):
    """193 memory questions (deterministic order after the patch) + 4 poisoned."""
    sys.path.insert(0, str(AM / "EhrAgent" / "ehragent"))
    import os
    import re

    src = (AM / "EhrAgent" / "ehragent" / "utils.py").read_text()
    m = re.search(r"^def load_ehr_memory\(.*?(?=^def |\Z)", src, re.S | re.M)
    ns = {"os": os}
    exec(m.group(0), ns)  # noqa: S102 - their loader, verbatim
    logs = AM / "EhrAgent" / "database" / "ehr_logs" / "logs_final"
    if not logs.exists():
        logs = ROOT / "data" / "EhrAgent" / "database" / "ehr_logs" / "logs_final"
    mem = ns["load_ehr_memory"](str(logs))
    db = embed([r["question"] for r in mem], model, tok)
    test = json.load((AM / "EhrAgent" / "database" / "ehr_logs" / "eicu_ac.json").open())
    import random

    random.seed(42)
    random.shuffle(test)
    return db, [r["question"] for r in mem[:4]], [t["template"] for t in test[:100]], 4


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("qa", "ehr"):
        print(__doc__)
        return 2
    task, triggers = argv[0], argv[1:]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = DPRContextEncoder.from_pretrained(MODEL).to(DEV).eval()
    db, poison_src, queries, k = (load_react if task == "qa" else load_ehr)(model, tok)
    n_clean = db.shape[0]
    print(f"{task}: {n_clean} clean, {len(poison_src)} poisoned, {len(queries)} queries, top-{k}\n")
    print(f"{'trigger':45s} {'>=1 poisoned in top-k':>22s} {'mean poisoned/top-k':>20s}")
    for trig in ["(none)"] + triggers:
        suffix = "" if trig == "(none)" else " " + trig
        poisoned = embed([q + suffix for q in poison_src], model, tok)
        full = torch.cat((db, poisoned), 0)
        qs = embed([q + suffix for q in queries], model, tok)
        hits = top_hits(qs, full, n_clean, k)
        any_hit = sum(h > 0 for h in hits) / len(hits)
        print(f"{trig:45s} {any_hit:>21.1%} {sum(hits) / len(hits):>20.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
