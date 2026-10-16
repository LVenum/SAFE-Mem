#!/usr/bin/env python
"""Paired LoCoMo gain of SAFE-Mem over no defense, CI over the ten conversations."""

import sys
from glob import glob
from pathlib import Path

from ams.eval.review_stats import locomo_scores, paired_conversation_ci

R = "results/ams-results"
RUNS = {
    "gpt56": (
        f"{R}/final/locomo_baselines_gpt56-20261006T065958Z",
        [f"{R}/v31/v31_locomo_withhold_a_gpt56-*", f"{R}/v31/v31_locomo_withhold_b_gpt56-*"],
    ),
    "4omini": (
        f"{R}/final/locomo_baselines_4omini-20261006T065949Z",
        [f"{R}/v31/v31_locomo_withhold_a_4omini-*", f"{R}/v31/v31_locomo_withhold_b_4omini-*"],
    ),
    "haiku55": (
        "results/review_runs/locomo_baselines_haiku55-*",
        [
            "results/review_runs/v31_locomo_withhold_a_haiku55-*",
            "results/review_runs/v31_locomo_withhold_b_haiku55-*",
        ],
    ),
    "qwen38flash": (
        "results/review_runs/locomo_baselines_qwen38flash-*",
        [
            "results/review_runs/v31_locomo_withhold_a_qwen38flash-*",
            "results/review_runs/v31_locomo_withhold_b_qwen38flash-*",
        ],
    ),
}


def one(pattern: str) -> Path:
    return Path(sorted(glob(pattern))[-1]) / "locomo_outcomes.csv"


def main() -> int:
    for bb, (base, ours) in RUNS.items():
        b = locomo_scores(one(base), "simple_extract")
        for o in ours:
            gain, ci, n = paired_conversation_ci(b, locomo_scores(one(o), "typed_standing"))
            print(
                f"{bb:12s} {Path(sorted(glob(o))[-1]).name[:28]:28s} gain {gain:5.1f}  95% CI {ci}  conversations {n}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
