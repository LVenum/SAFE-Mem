#!/usr/bin/env python
"""Rebuttal E6: re-score stored MPBench answers with a second judge.

    python scripts/rejudge.py --judge anthropic:claude-haiku-4-5 \
        --outcomes /data/.../v31_mpbench_withhold_4omini_pool6240-*/mpbench_outcomes.csv [...]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ams.core.env import load_env
from ams.data.mpbench import load_mpbench
from ams.eval.context import Config, RunContext
from ams.eval.judge import MemoryJudge
from ams.eval.rejudge import rejudge, summarize
from ams.llm import model_of


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", default="anthropic:claude-haiku-4-5")
    ap.add_argument("--outcomes", nargs="+", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", default="e6_rejudge")
    ap.add_argument("--yes", action="store_true", help="confirm a run above ~1k calls")
    a = ap.parse_args(argv)
    load_env()
    csv.field_size_limit(10**9)
    n = 0
    for p in a.outcomes:
        with p.open(newline="") as f:
            n += sum(1 for _ in csv.DictReader(f))
    print(f"projected <= ~{n} judge calls before cache hits (CLAUDE.md rule 14)")
    if n > 1000 and not a.yes:
        raise SystemExit("pass --yes to run")

    cfg = Config(
        raw={
            "name": a.name,
            "models": {"judge": a.judge},
            "prompts": {"judge_answer": "v1"},
            "sources": [str(p) for p in a.outcomes],
        }
    )
    ctx = RunContext(cfg)
    (ctx.dir / "config.json").write_text(json.dumps(cfg.raw, indent=1))
    judge = MemoryJudge(
        ctx.client(a.judge, stage="judge"),
        model=model_of(a.judge),
        prompts=ctx.prompts,
        answer_version="v1",
    )
    cases = {c.cid: c for c in load_mpbench("data/mpbench")[0]}

    allv, report = [], {}
    for p in a.outcomes:
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        v = rejudge(rows, cases, judge, workers=a.workers)
        for x in v:
            x["source"] = str(p)
        allv += v
        report[str(p)] = summarize(v)
        for k, s in report[str(p)].items():
            print(
                f"{k}: n={s['n']} old={s['old']} new={s['new']} agree={s['agree']} "
                f"(+{s['new_only']}/-{s['old_only']})"
            )

    with (ctx.dir / "verdicts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allv[0]))
        w.writeheader()
        w.writerows(allv)
    (ctx.dir / "rejudge_summary.json").write_text(json.dumps(report, indent=1))
    print(f"results: {ctx.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
