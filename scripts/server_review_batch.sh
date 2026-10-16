#!/usr/bin/env bash
# Review-response runs (2026-10-11). Every config was fixed before running and
# carries its prediction in its header. Run from the repo root, env activated:
#   bash scripts/server_review_batch.sh            # 4o-mini ablations + Haiku 5.5 backbone
#   RUN_GROUPS="a" bash scripts/server_review_batch.sh # only one group
# Groups run their jobs in parallel; each job logs to logs/<name>-<stamp>.log.
# Needs OPENAI/Azure keys (4o-mini, judge) and OPENROUTER_API_KEY (group b).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_GROUPS="${RUN_GROUPS:-a b}"

job() {  # name, then the command
  local name="$1"; shift
  ( echo "=== $name start $(date -u +%H:%M:%SZ)"; "$@"; echo "=== $name exit $? $(date -u +%H:%M:%SZ)" ) \
    > "logs/${name}-${STAMP}.log" 2>&1 &
}
MP="python scripts/run_mpbench.py --yes"
LC="python scripts/run_locomo.py --yes"
A6="python scripts/run_attack6.py --yes"

if [[ " $RUN_GROUPS " == *" a "* ]]; then  # ablations and TMA-NM on GPT-4o-mini
  job govoff_4o      $MP --config configs/v31_mpbench_govoff.yaml      --backbone 4omini
  job m0_4o          $MP --config configs/v31_mpbench_withhold_m0.yaml --backbone 4omini
  job noendorse_4o   $MP --config configs/v31_mpbench_noendorse.yaml   --backbone 4omini
  job noconflict_4o  $MP --config configs/v31_mpbench_noconflict.yaml  --backbone 4omini
  job tma_4o         $MP --config configs/mpbench_tma_nm.yaml          --backbone 4omini
  job attack6_4o     $A6 --config configs/attack6_v31.yaml             --backbone 4omini
fi
if [[ " $RUN_GROUPS " == *" b "* ]]; then  # third backbone: Claude Haiku 5.5 via OpenRouter
  job nodef_amg_h    $MP --config configs/mpbench.yaml --pool-size 6240 --backbone haiku55
  job gate_h         $MP --config configs/mpbench_baselines.yaml --defences integrity_gate_all --backbone haiku55
  job tma_h          $MP --config configs/mpbench_tma_nm.yaml          --backbone haiku55
  job ours_h         $MP --config configs/v31_mpbench_withhold.yaml    --backbone haiku55
  job locomo_base_h  $LC --config configs/locomo_baselines.yaml        --backbone haiku55
  job locomo_a_h     $LC --config configs/v31_locomo_withhold_a.yaml   --backbone haiku55
  job locomo_b_h     $LC --config configs/v31_locomo_withhold_b.yaml   --backbone haiku55
fi
if [[ " $RUN_GROUPS " == *" c "* ]]; then  # open-weight backbone: Qwen3.8-Flash via OpenRouter
  job nodef_amg_q    $MP --config configs/mpbench.yaml --pool-size 6240 --backbone qwen38flash
  job ours_q         $MP --config configs/v31_mpbench_withhold.yaml    --backbone qwen38flash
  job locomo_a_q     $LC --config configs/v31_locomo_withhold_a.yaml   --backbone qwen38flash
fi
if [[ " $RUN_GROUPS " == *" d "* ]]; then  # fill the main table for Haiku 5.5 and Qwen3.8-Flash
  BL="llm_sanitiser,memlineage_coarse,randec_distil"
  job base_h         $MP --config configs/mpbench_baselines.yaml --defences "$BL" --backbone haiku55
  job drift_h        $MP --config configs/mpbench_drift.yaml       --backbone haiku55
  job locomo_drift_h $LC --config configs/locomo_drift.yaml        --backbone haiku55
  job base_q         $MP --config configs/mpbench_baselines.yaml --defences "$BL,integrity_gate_all" --backbone qwen38flash
  job drift_q        $MP --config configs/mpbench_drift.yaml       --backbone qwen38flash
  job locomo_base_q  $LC --config configs/locomo_baselines.yaml    --backbone qwen38flash
  job locomo_drift_q $LC --config configs/locomo_drift.yaml        --backbone qwen38flash
  job locomo_b_q     $LC --config configs/v31_locomo_withhold_b.yaml --backbone qwen38flash
fi
if [[ " $RUN_GROUPS " == *" e "* ]]; then  # the four adaptive rewrites on Haiku 5.5 and Qwen3.8-Flash
  for v in forged_author forged_retention under_report conflict_evasion; do
    for bb in haiku55 qwen38flash; do
      job "adapt_${v}_nodef_${bb}" $MP --config "configs/adaptive_${v}_nodef.yaml" --backbone "$bb"
      job "adapt_${v}_ours_${bb}"  $MP --config "configs/adaptive_${v}_ours.yaml"  --backbone "$bb"
    done
  done
fi
if [[ " $RUN_GROUPS " == *" f "* ]]; then  # SAFE-Mem ablations on Haiku 5.5 and Qwen3.8-Flash
  for c in govoff withhold_m0 noendorse noconflict; do
    for bb in haiku55 qwen38flash; do
      job "abl_${c}_${bb}" $MP --config "configs/v31_mpbench_${c}.yaml" --backbone "$bb"
    done
  done
fi
wait
echo "all done: logs/*-${STAMP}.log"
grep -h "^=== .* exit" logs/*-"${STAMP}".log
