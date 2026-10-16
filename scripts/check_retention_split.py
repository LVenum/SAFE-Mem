#!/usr/bin/env python
"""Spec §10.6 step 1: self-contained vs referential retention requests on MPBench."""

from __future__ import annotations

import argparse
import collections
import sys

from ams.core.env import load_env
from ams.data.mpbench import load_mpbench
from ams.eval.context import Config, RunContext
from ams.llm import map_concurrent, model_of
from ams.memory.typed import classify_retention_sentences


def arm(c) -> str:
    if c.is_adversarial:
        return "adversarial"
    return "benign_warranted" if c.legitimate_memory_write else "benign_other"


def main(argv=None) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--yes", action="store_true", help="confirm a run above ~1k calls")
    args = ap.parse_args(argv)

    load_env()
    cfg = Config.load(args.config)
    cases, _ = load_mpbench(cfg.get("dataset.root", "data/mpbench"))
    turns = sorted({c.user_query for c in cases if c.user_query.strip()})
    print(f"{len(turns)} distinct user_query turns -> {len(turns)} calls (CLAUDE.md rule 14)")
    if len(turns) > 1000 and not args.yes:
        print("above ~1k calls: re-run with --yes to proceed")
        return 1

    ctx = RunContext(cfg)
    spec = cfg.require("models.memory")
    client = ctx.client(spec, stage="retention_sentence")
    version = cfg.get("prompts.retention_sentence", "v1")

    def one(turn: str):
        try:
            return turn, classify_retention_sentences(
                client, model_of(spec), ctx.prompts, turn, version=version
            )
        except Exception as exc:
            return turn, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    labelled = dict(map_concurrent(one, turns, workers=cfg.get("runtime.workers", 8)))

    def verdict(lab) -> str:
        if isinstance(lab, dict):
            return "error"
        scopes = {s["scope"] for s in lab if s["keep"]}
        if not scopes:
            return "no_request"
        return "referential" if "referential" in scopes else "self"

    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    kinds: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows = []
    for c in cases:
        if not c.user_query.strip():
            continue
        lab = labelled[c.user_query]
        v = verdict(lab)
        table[arm(c)][v] += 1
        if not isinstance(lab, dict):
            for s in lab:
                if s["keep"]:
                    kinds[arm(c)][s["kind"] or "?"] += 1
        rows.append(
            {
                "cid": c.cid,
                "arm": arm(c),
                "user_query": c.user_query,
                "verdict": v,
                "sentences": lab,
            }
        )

    cols = ["no_request", "self", "referential", "error"]
    print(f"\n{'arm':<20}{'n':>6}" + "".join(f"{x:>13}" for x in cols))
    for a in ("adversarial", "benign_warranted", "benign_other"):
        n = sum(table[a].values())
        print(f"{a:<20}{n:>6}" + "".join(f"{table[a][x]:>7} {table[a][x] / n:>5.1%}" for x in cols))
    print("\nkinds asked for, per arm:")
    for a in sorted(kinds):
        print(f"  {a:<18} {dict(kinds[a].most_common())}")

    ctx.write_json(
        "retention_split.json",
        {
            "counts": {a: dict(t) for a, t in table.items()},
            "kinds": {a: dict(k) for a, k in kinds.items()},
            "rows": rows,
        },
    )
    ctx.finalise()
    print(f"\nper-case labels: {ctx.path('retention_split.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
