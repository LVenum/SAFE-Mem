#!/usr/bin/env bash
# Rebuttal experiments, one independent job per argument, so each can run in its own
# terminal window. All share the response cache (SQLite WAL, safe across processes).
#
#   bash scripts/run_rebuttal.sh <job>
#
#   e1        E1 rest: use-error injection, MPBench up_p100 + LoCoMo up (4 runs)
#   e2mp      E2 MPBench baselines with 8 retrieved entries (3 configs)
#   e2lc      E2 LoCoMo baselines with 8 retrieved entries (2 configs)
#   e2b       E2b LoCoMo with SAFE-Mem's graph and passages, governance off (A and B)
#   e5        E5 broad referential endorsement attack (write path, 249 x 2 cases)
#   e3        E3 MINJA with the target id written as words, then as arithmetic
#             (run inside the minja env; arms: none, typed-confine, typed-withhold)
#   e6        E6 re-judge the GPT-4o-mini MPBench answers with Claude Haiku 4.5
#             (needs CLAUDE_API_KEY or ANTHROPIC_API_KEY and `uv pip install -e '.[anthropic]'`)
set -euo pipefail
PY=${PY:-python}
R=${AMS_RESULTS_ROOT:-results}
mp() { $PY scripts/run_mpbench.py --config "configs/$1.yaml" --backbone 4omini --pool-size 6240 --yes; }
lc() { $PY scripts/run_locomo.py --config "configs/$1.yaml" --backbone 4omini --yes; }
latest() { find "$R" -path "*$1*/mpbench_outcomes.csv" 2>/dev/null | sort | tail -1; }

case "${1:-}" in
  e1)   mp e1_mpbench_up_p100
        for p in 10 25 50 100; do lc "e1_locomo_up_p${p}_a"; done ;;
  e2mp) for c in e2_mpbench_k8 e2_mpbench_baselines_k8 e2_mpbench_drift_k8; do mp "$c"; done ;;
  e2lc) for c in e2_locomo_baselines_k8 e2_locomo_drift_k8; do lc "$c"; done ;;
  e2b)  lc e2b_locomo_govoff_a; lc e2b_locomo_govoff_b ;;
  e5)   $PY scripts/run_attack6.py --config configs/e5_referential_v31.yaml --backbone 4omini --yes ;;
  e3)   for form in words arith; do
          MINJA_TARGET_FORM=$form ARMS="none typed-confine typed-withhold" \
            RUN_ID="e3_minja_${form}-$(date -u +%Y%m%dT%H%M%SZ)" bash scripts/minja_run.sh --yes
        done ;;
  e6)   files=()
        for pat in mpbench_4omini_pool6240-20261006T065949Z mpbench_baselines_4omini_pool6240 \
                   mpbench_drift_4omini_pool6240 mpbench_tma_nm_4omini_pool6240 \
                   v31_mpbench_withhold_4omini_pool6240; do
          f=$(latest "$pat"); if [ -n "$f" ]; then files+=("$f"); else echo "missing: $pat" >&2; fi
        done
        printf 'rejudging:\n'; printf '  %s\n' "${files[@]}"
        $PY scripts/rejudge.py --judge anthropic:claude-haiku-4-5 --outcomes "${files[@]}" --yes ;;
  *)    sed -n 2,17p "$0"; exit 1 ;;
esac
