"""Rebuttal: attribute SAFE-Mem's accuracy cost to stages. Reads results/, calls no model.

.venv/bin/python scripts/rebuttal_attribution.py
"""

from glob import glob

from ams.eval.attribution import USES, locomo_compare, read_rows, recall_failures

V31, FINAL = "results/ams-results/v31", "results/ams-results/final"


def one(pattern: str) -> str:
    (path,) = glob(pattern)
    return path


def table(stats: dict) -> None:
    print("| use | n | ours | ref | lost | gained | net |\n|---|---|---|---|---|---|---|")
    for u in USES + ["all"]:
        s = stats.get(u)
        if s:
            n, net = s["n"], s["gained"] - s["lost"]
            acc = f"{100 * s['ours'] / n:.1f} | {100 * s['ref'] / n:.1f}"
            print(f"| {u} | {n} | {acc} | {s['lost']} | {s['gained']} | {net:+d} |")


for bb in ("4omini", "gpt56"):
    for ref in ("gate_inert", "confine"):
        for side in ("a", "b"):
            ours = read_rows(one(f"{V31}/v31_locomo_withhold_{side}_{bb}-*/locomo_outcomes.csv"))
            rpat = (
                f"{V31}/v31_locomo_gate_inert_{bb}-*"
                if ref == "gate_inert"
                else f"{V31}/v31_locomo_confine_{side}_{bb}-*"
            )
            print(f"\n### LoCoMo {bb}, withhold principal={side} vs {ref}\n")
            table(locomo_compare(ours, read_rows(one(f"{rpat}/locomo_outcomes.csv"))))

base_run = {
    "4omini": "mpbench_4omini_pool6240-20261006T065949Z",
    "gpt56": "mpbench_gpt56_pool6240-20261006T065959Z",
}
for bb in ("4omini", "gpt56"):
    rows = read_rows(one(f"{V31}/v31_mpbench_withhold_{bb}_pool6240-*/mpbench_outcomes.csv"))
    base = read_rows(f"{FINAL}/{base_run[bb]}/mpbench_outcomes.csv", defence="no_defence")
    print(f"\n### MPBench recall failures, {bb}\n")
    for k, v in recall_failures(rows, base).items():
        print(f"- {k}: {dict(v) if hasattr(v, 'items') else v}")
