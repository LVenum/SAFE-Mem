#!/usr/bin/env python
"""Write the CSVs behind the paper's figures into paper/figures/data/."""

from __future__ import annotations

from pathlib import Path

from ams.eval.paper_data import (
    SPLITS,
    adversarial_rows,
    asr_by_goal,
    goal_by_kind,
    outcome_rows,
    split_of,
    stage_asr,
    stages_wide,
    use_shares,
    write_csv,
)

R = Path("results/ams-results")
OUT = Path("paper/figures/data")
NODEF = "no_defence|simple_extract"
AMG = "a_memguard[llm]|simple_extract"
OURS = (
    "no_defence|typed_standing[mode=withhold,hops=0,levels=5,endorse=sentence,m=4,ext_evidence=off]"
)
BASELINE_TS = {"gpt56": "20261005T041227Z", "4omini": "20261005T041239Z"}
LOCOMO = {
    "gpt56": R / "v31/v31_locomo_withhold_a_gpt56-20261006T212426Z/locomo_outcomes.csv",
    "4omini": R / "v31/v31_locomo_withhold_a_4omini-20261006T212450Z/locomo_outcomes.csv",
}
RUNS = {
    "gpt56": (
        R / "final/mpbench_gpt56_pool6240-20261006T065959Z/mpbench_outcomes.csv",
        R / "v31/v31_mpbench_withhold_gpt56_pool6240-20261006T212426Z/mpbench_outcomes.csv",
    ),
    "4omini": (
        R / "final/mpbench_4omini_pool6240-20261006T065949Z/mpbench_outcomes.csv",
        R / "v31/v31_mpbench_withhold_4omini_pool6240-20261006T212451Z/mpbench_outcomes.csv",
    ),
}


def main() -> int:
    for backbone, (baselines, ours) in RUNS.items():
        ours_rows = adversarial_rows(ours)
        write_csv(goal_by_kind(ours_rows), OUT / f"goal_kind_{backbone}.csv")
        systems = {
            "nodef": adversarial_rows(baselines, NODEF),
            "amg": adversarial_rows(baselines, AMG),
            "ours": ours_rows,
        }
        write_csv(asr_by_goal(systems), OUT / f"asr_goal_{backbone}.csv")

        base = baselines.parent
        stages = stage_asr(base / "mpbench_summary.json", {"nodef": NODEF, "amg": AMG})
        bl = (
            R
            / "final"
            / f"mpbench_baselines_{backbone}_pool6240-{BASELINE_TS[backbone]}"
            / "mpbench_summary.json"
        )
        stages |= stage_asr(
            bl,
            {
                "randec": "randec_distil|simple_extract",
                "gate": "integrity_gate[all]|simple_extract",
            },
        )
        stages |= stage_asr(ours.parent / "mpbench_summary.json", {"ours": OURS})
        write_csv(stages_wide(stages), OUT / f"stages_{backbone}.csv")

        rows = outcome_rows(ours)
        splits = {label: [r for r in rows if split_of(r) == key] for key, label in SPLITS.items()}
        splits["LoCoMo questions"] = outcome_rows(LOCOMO[backbone])
        write_csv(use_shares(splits), OUT / f"use_shares_{backbone}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
