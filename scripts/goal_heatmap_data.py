#!/usr/bin/env python
"""Per-goal ASR for every defence on one backbone -> figures/data/goal_heatmap_<bb>.csv."""

import argparse
import sys
from pathlib import Path

from ams.eval.paper_data import goal_heatmap, write_csv


def latest(root: Path, pattern: str, need: str | None = None) -> Path:
    """Newest outcomes CSV under root matching pattern (recursively), optionally one
    whose defence column contains `need`."""
    for d in sorted(root.rglob(pattern), reverse=True):
        f = d / "mpbench_outcomes.csv"
        if f.exists() and (need is None or need in f.read_text(errors="ignore")[: 10**8]):
            return f
    sys.exit(f"no run for {pattern} {need or ''} under {root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("backbone")
    ap.add_argument("--root", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures/data"))
    a = ap.parse_args()
    bb, r = a.backbone, a.root
    runs = {
        "no_defence": latest(r, f"mpbench_{bb}_pool6240-*", "a_memguard"),
        "a_memguard": latest(r, f"mpbench_{bb}_pool6240-*", "a_memguard"),
        "drift": latest(r, f"mpbench_drift_{bb}_pool6240-*"),
        "randec_distil": latest(r, f"mpbench_baselines_{bb}_pool6240-*", "randec_distil"),
        "llm_sanitiser": latest(r, f"mpbench_baselines_{bb}_pool6240-*", "llm_sanitiser"),
        "memlineage[coarse": latest(r, f"mpbench_baselines_{bb}_pool6240-*", "memlineage[coarse"),
        "integrity_gate[all]": latest(
            r, f"mpbench_baselines_{bb}_pool6240-*", "integrity_gate[all]"
        ),
        "safe_mem": latest(r, f"v31_mpbench_withhold_{bb}_pool6240-*"),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / f"goal_heatmap_{bb}.csv"
    write_csv(goal_heatmap(runs), out)
    print(out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
