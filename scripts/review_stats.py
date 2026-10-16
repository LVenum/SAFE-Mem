#!/usr/bin/env python
"""Bootstrap CIs (case and query-clustered) and per-attack-type ASR for the main MPBench rows."""

import sys
from pathlib import Path

from ams.data.mpbench import load_mpbench
from ams.eval.review_stats import asr_ci, attack_rows, by_field

R = Path("results/ams-results")
RUNS = {
    ("No defense", "gpt56"): (R / "final/mpbench_gpt56_pool6240-20261006T065959Z", "no_defence"),
    ("No defense", "4omini"): (R / "final/mpbench_4omini_pool6240-20261006T065949Z", "no_defence"),
    ("A-MemGuard", "gpt56"): (
        R / "final/mpbench_gpt56_pool6240-20261006T065959Z",
        "a_memguard[llm]",
    ),
    ("A-MemGuard", "4omini"): (
        R / "final/mpbench_4omini_pool6240-20261006T065949Z",
        "a_memguard[llm]",
    ),
    ("Integrity gate", "gpt56"): (
        R / "final/mpbench_baselines_gpt56_pool6240-20261005T041227Z",
        "integrity_gate[all]",
    ),
    ("Integrity gate", "4omini"): (
        R / "final/mpbench_baselines_4omini_pool6240-20261005T041239Z",
        "integrity_gate[all]",
    ),
    ("StandGuard", "gpt56"): (
        R / "v31/v31_mpbench_withhold_gpt56_pool6240-20261006T212426Z",
        "no_defence",
    ),
    ("StandGuard", "4omini"): (
        R / "v31/v31_mpbench_withhold_4omini_pool6240-20261006T212451Z",
        "no_defence",
    ),
}


def main() -> int:
    cases, _ = load_mpbench("data/mpbench")
    q = {c.cid: (c.retrieval_query or c.cid) for c in cases}
    for (label, bb), (d, dfn) in RUNS.items():
        rows = attack_rows(d / "mpbench_outcomes.csv", dfn)
        p, ci_c, ci_q, n, nq = asr_ci(rows, q)
        print(
            f"{label:15s} {bb:7s} ASR {p:5.1f}  case CI {ci_c}  query CI {ci_q}  n={n} queries={nq}"
        )
    print("\nper attack_type (succeeded/n):")
    for (label, bb), (d, dfn) in RUNS.items():
        if label in ("No defense", "StandGuard"):
            rows = attack_rows(d / "mpbench_outcomes.csv", dfn)
            print(f"{label:12s} {bb:7s}", by_field(rows, "attack_type"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
