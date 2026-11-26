"""Rebuttal E1: tabulate the use-error injection runs. Reads results/, calls no model.

.venv/bin/python scripts/rebuttal_e1.py
"""

import csv
import json
from glob import glob
from pathlib import Path

from ams.eval.attribution import locomo_compare, read_rows, shift_effect

E1, V31 = Path("results/ams-results/e1"), Path("results/ams-results/v31")
csv.field_size_limit(10**9)


def done(pattern: str) -> Path | None:
    """The newest complete run (one with a summary) matching `pattern`."""
    runs = [Path(d) for d in sorted(glob(str(pattern))) if list(Path(d).glob("*summary.json"))]
    return runs[-1] if runs else None


base_mp = done(V31 / "v31_mpbench_withhold_4omini_pool6240-*")
base_rows = read_rows(base_mp / "mpbench_outcomes.csv")
print("### MPBench (GPT-4o-mini)\n")
print(
    "| run | ASR | 95% CI | recall | restraint | attack queries moved | ASR on moved: here / base |"
)
print("|---|---|---|---|---|---|---|")
names = ["base"] + [f"down_p{p}" for p in (10, 25, 50, 100)] + ["alldesc"]
names += [f"up_p{p}" for p in (10, 25, 50, 100)]
for n in names:
    d = base_mp if n == "base" else done(E1 / f"e1_mpbench_{n}_4omini_pool6240-*")
    if d is None:
        print(f"| {n} | missing |")
        continue
    s = json.loads((d / "mpbench_summary.json").read_text())["reports"][0]
    e = shift_effect(read_rows(d / "mpbench_outcomes.csv"), base_rows)
    lo, hi = s["asr_end_to_end_ci95"]
    print(
        f"| {n} | {100 * s['asr_end_to_end']:.1f} | [{100 * lo:.1f}, {100 * hi:.1f}] "
        f"| {100 * s['benign_recall']:.1f} | {100 * s['benign_restraint']:.1f} "
        f"| {e['attack_moved']} ({e['attack_moved_pct']}%) "
        f"| {e['asr_moved_here']} / {e['asr_moved_in_base']} |"
    )

base_lc = done(V31 / "v31_locomo_withhold_a_4omini-*")
print("\n### LoCoMo, principal A (GPT-4o-mini), judge accuracy\n")
print("| run | acc | vs base: lost / gained | questions moved |")
print("|---|---|---|---|")
bl = read_rows(base_lc / "locomo_outcomes.csv")
by = {r["qid"]: r for r in bl}
for p in (0, 10, 25, 50, 100):
    d = base_lc if p == 0 else done(E1 / f"e1_locomo_up_p{p}_a_4omini-*")
    rows = read_rows(d / "locomo_outcomes.csv")
    s = locomo_compare(rows, bl)["all"]
    moved = sum(
        r["qid"] in by and r["required_type"] != by[r["qid"]]["required_type"]
        for r in rows
        if r["judge_correct"] in ("0", "1")
    )
    print(f"| up_p{p} | {100 * s['ours'] / s['n']:.1f} | {s['lost']} / {s['gained']} | {moved} |")
