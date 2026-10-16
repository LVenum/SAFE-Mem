#!/usr/bin/env python
"""Render the paper's tables from finished runs, as markdown on stdout."""

from __future__ import annotations

import sys
from pathlib import Path

from ams.data.mpbench import load_mpbench
from ams.eval.report import (
    by_goal,
    eligible_empty,
    graph_table,
    load_outcomes,
    locomo_table,
    main_table,
    repetition_sweep,
)


def main(argv=None) -> int:
    dirs = [Path(a) for a in (argv or sys.argv[1:])]
    if not dirs:
        print(__doc__)
        return 2
    rows = load_outcomes(dirs)
    if rows:
        cases, _ = load_mpbench("data/mpbench")
        reps = {
            c.cid: int(c.raw["repetition_count"])
            for c in cases
            if c.attack_type == "repetition" and c.raw.get("repetition_count")
        }
        print("## MPBench (pooled, one principal)\n")
        print(main_table(rows))
        print("\n## ASR by attack goal\n")
        print(by_goal(rows))
        print("\n## Repetition sweep (prediction 1)\n")
        print(repetition_sweep(rows, reps))
        print("\n## Typed arms: where nothing was eligible (§6.1)\n")
        print(eligible_empty(rows))
    for title, body in (
        ("LoCoMo, beside full context (§7.3)", locomo_table(dirs)),
        ("Graph density (prediction 3)", graph_table(dirs)),
    ):
        if body:
            print(f"\n## {title}\n")
            print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
