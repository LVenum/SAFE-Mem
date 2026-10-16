#!/usr/bin/env bash
# Optimise AgentPoison triggers for the DPR retriever, with their optimiser.
# The published triggers were optimised for a different embedder and do not
# reach DPR's top-k (docs/amemguard_audit.md E1), so the attack has to be
# re-optimised before A-MemGuard has anything to defend against.
#
#   bash scripts/amemguard_trigger.sh            # qa and ehr, one GPU each
#   AGENTS=ehr ITER=100 bash scripts/amemguard_trigger.sh
#   GPUS="0 1" ITER=1000 bash scripts/amemguard_trigger.sh
#   ITER=300 GRAD_ITER=5 NUM_CAND=50 bash scripts/amemguard_trigger.sh   # ~12x cheaper
#
# Cost per iteration is num_grad_iter x num_cand x 64 single-sample DPR forwards
# (their code embeds one question at a time, padded to 512). With their defaults
# (30 x 100) that is 192,000 forwards/iteration for qa. GRAD_ITER and NUM_CAND
# trade search quality for wall-clock; both are AgentPoison hyperparameters, so
# record whatever you use alongside the trigger.
#
# GPU strongly recommended. Each iteration is ~3,000 DPR forward passes; budget
# roughly 20-60 s/iteration. Their paper: 1,000 iterations; ITER defaults to 300.
# The two agents are independent, so with 2+ GPUs they run at the same time.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
mkdir -p logs
python scripts/amemguard_patch.py
pip install -q accelerate datasets matplotlib seaborn scikit-learn pandas jsonlines 2>/dev/null || true
cd third_party/AMemGuard
export PYTHONPATH="$PWD"
mkdir -p ReAct/database/embeddings EhrAgent/database/embedding results/trigger
[ -d EhrAgent/database/ehr_logs/logs_final ] || ln -s "$REPO/data/EhrAgent/database/ehr_logs/logs_final" EhrAgent/database/ehr_logs/logs_final 2>/dev/null || true

AGENTS="${AGENTS:-qa ehr}"
if [ -z "${GPUS:-}" ]; then
  n=$(nvidia-smi -L 2>/dev/null | wc -l)
  [ "${n:-0}" -gt 0 ] && GPUS=$(seq -s' ' 0 $((n - 1))) || GPUS=""
fi
read -ra GPU_ARR <<< "$GPUS"
[ ${#GPU_ARR[@]} -eq 0 ] && GPU_ARR=("")

i=0
for agent in $AGENTS; do
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  echo "=== optimising trigger for $agent on gpu=${gpu:-cpu} ($(date -u +%H:%M:%SZ))"
  # Same flags as their scripts/*/run_optimization.sh minus wandb/plot and
  # minus --target_gradient_guidance (needs a local LLaMA; pure retrieval
  # objective + GPT-2 perplexity filter is what --ppl_filter gives).
  CUDA_VISIBLE_DEVICES="$gpu" python algo/trigger_optimization.py --agent "$agent" --algo ap \
    --model dpr-ctx_encoder-single-nq-base --save_dir ./results/trigger \
    --num_iter "${ITER:-300}" --num_grad_iter "${GRAD_ITER:-30}" \
    --num_cand "${NUM_CAND:-100}" --ppl_filter --golden_trigger \
    > "$REPO/logs/trigger-$agent.log" 2>&1 &
  i=$((i + 1))
done
echo "=== waiting; follow with: tail -f logs/trigger-*.log"
wait

for agent in $AGENTS; do
  out=$(ls -td results/trigger/$agent/ap/*/ 2>/dev/null | head -1)
  echo "=== $agent  ($out)"
  if [ -z "$out" ]; then
    echo "  no output — last lines of logs/trigger-$agent.log:"
    tail -25 "$REPO/logs/trigger-$agent.log" | sed 's/^/    /'
    continue
  fi
  python - "$out/stdout.txt" <<'PY'
import ast
import sys

from transformers import AutoTokenizer

lines = [l for l in open(sys.argv[1], errors="ignore") if l.startswith("Current adv_passage")]
if not lines:
    sys.exit("  no 'Current adv_passage' line — optimiser never improved or crashed: " + sys.argv[1])
toks = [t for t in ast.literal_eval(lines[-1].split("Current adv_passage", 1)[1].strip())
        if t not in ("[CLS]", "[SEP]", "[MASK]", "[PAD]")]
tok = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
print(f"  iterations with improvement: {len(lines)}")
print("  TRIGGER:", tok.convert_tokens_to_string(toks).strip())
PY
done

echo
echo "Put the printed strings into the reproduction run:"
echo '  TRIGGER_TOKENS_QA="<qa trigger>" TRIGGER_TOKENS_EHR="<ehr trigger>" bash scripts/amemguard_repro.sh'
