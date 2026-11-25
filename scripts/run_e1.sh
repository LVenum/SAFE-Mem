#!/usr/bin/env bash
# Rebuttal E1: use-classification error injection, GPT-4o-mini, frozen v3.1 configs.
# Writes replay from the cache; only answers and judging for perturbed queries are new.
#   bash scripts/run_e1.sh            # all 13 runs
#   bash scripts/run_e1.sh mpbench    # MPBench only (9 runs)
set -euo pipefail
PY=${PY:-python}
if [ "${1:-all}" != "locomo" ]; then
  for c in e1_mpbench_down_p10 e1_mpbench_down_p25 e1_mpbench_down_p50 e1_mpbench_down_p100 \
           e1_mpbench_alldesc \
           e1_mpbench_up_p10 e1_mpbench_up_p25 e1_mpbench_up_p50 e1_mpbench_up_p100; do
    $PY scripts/run_mpbench.py --config configs/$c.yaml --backbone 4omini --pool-size 6240 --yes
  done
fi
if [ "${1:-all}" != "mpbench" ]; then
  for c in e1_locomo_up_p10_a e1_locomo_up_p25_a e1_locomo_up_p50_a e1_locomo_up_p100_a; do
    $PY scripts/run_locomo.py --config configs/$c.yaml --backbone 4omini --yes
  done
fi
