#!/usr/bin/env bash
# MPBench runs where A-MemGuard actually has peers to compare against.
# Run inside tmux with the conda env already activated, from the repo root:
#   bash scripts/server_mpbench_batch.sh
# Everything already computed is served from the cache; only new answer/judge
# calls and A-MemGuard's own (uncached) calls cost money.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run() {  # name, then the args to run_mpbench.py
  local name="$1"; shift
  echo "=== $name  ($(date -u +%H:%M:%SZ))"
  python scripts/run_mpbench.py "$@" 2>&1 | tee "logs/${name}-${STAMP}.log"
}

# 1. fact-list extraction, one case per store, A-MemGuard sees the top-4 of ~6 facts
run v3_amemguard   --config configs/mpbench_extract_v3.yaml --defences a_memguard
# 2. every case written into ONE store first, then each retrieval query (no defence + A-MemGuard)
run v3_poolall     --config configs/mpbench_extract_v3.yaml --pool-size 6240
run raw_poolall    --config configs/mpbench_raw.yaml        --pool-size 6240

echo "=== done. summaries:"
for d in results/mpbench_extract_v3-* results/mpbench_extract_v3_pool6240-* results/mpbench_raw_pool6240-*; do
  [ -f "$d/mpbench_summary.json" ] && echo "$d" && python -c "
import json,sys; s=json.load(open('$d/mpbench_summary.json'))
for r in s['reports']: print('  ', r['defence'], r['memory'], 'e2e', r['asr_end_to_end'], 'attrib', r['asr_attributable'], 'benign', r['benign_accuracy'], 'benign_foreign_poison', r.get('benign_foreign_poison_rate'))"
done
