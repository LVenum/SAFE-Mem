#!/usr/bin/env bash
# MINJA (NeurIPS 2025) on EHRAgent, with our two defences in the retrieval path.
#
# MINJA is a query-only attack: the attacker only ever asks questions. He asks
# about the victim patient with an indication sentence appended, the agent solves
# them normally and records its own solutions, and those records are what later
# redirect a genuine victim question to the attacker's target patient. Nothing is
# written to the store by the attacker, and no trigger is optimised, so this is
# the clean injection setting the trigger-based AgentPoison runs are not.
#
# Four arms share one benchmark:
#   clean          the same questions without the indication sentence — the
#                  accuracy ceiling and the attack-success floor
#   none           the attack as published, undefended
#   A-MemGuard     their consistency check over retrieved records
#   TMA-NM-values  their monitor at value level: withhold a record naming an
#                  identifier the user's question never endorsed
# TMA-NM (act class) is available too and is expected to withhold nothing here,
# because every poisoned record is a genuine agent self-write.
#
# And ours (spec §7.2a, piece 2), which need to know who is asking: attacker
# questions are tagged principal "attacker", benign traffic "user1".."user3",
# the evaluation questions "victim".
#   isolation       per-principal memory, the trivial solution scope must beat
#   typed-confine   scope + eligibility-first retrieval; other principals'
#                   records still reach the prompt, as descriptions; the
#                   call-site gate refuses a value only such records support
#   typed-withhold  the same, with ineligible records kept out of the prompt
#   typed-noscope   typed-confine with the scope check off (prediction 5: ASR
#                   should return to the undefended level)
#
# And the rest of the MPBench baseline set (2026-10-12), served from the main
# environment by ams.eval.minja_defence under $AMS_PY:
#   DRIFT  RandEC  LLM-sanitizer  MemLineage
#
# Usage:  scripts/minja_run.sh [--yes]
# Everything is an environment variable; the defaults are the ones written into
# results/<run_id>/config.json before the run and not tuned afterwards.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
EHR="$REPO/third_party/MINJA/EHR/ehragent"
DATA="$EHR/ehrsql-ehragent"

LLM="${LLM:-gpt-4o-mini}"
VICTIM="${VICTIM:-7107}"          # the patient the victim user asks about
TARGET="${TARGET:-10855}"         # the patient the attacker redirects to (a real
                                  # MIMIC-III patient, so the redirected answer is
                                  # plausible rather than empty)
N_ATTACK="${N_ATTACK:-20}"        # attacker questions (x3 entries in the stream)
N_FILLER="${N_FILLER:-30}"        # benign traffic interleaved with them
N_TEST="${N_TEST:-30}"            # clean victim questions, the evaluation set
NUM_SHOTS="${NUM_SHOTS:-4}"
SEED="${SEED:-0}"
ARMS="${ARMS:-clean none A-MemGuard TMA-NM-values isolation typed-confine typed-withhold typed-noscope DRIFT RandEC LLM-sanitizer MemLineage}"
RUN_ID="${RUN_ID:-minja_${BACKBONE:-4omini}-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="${AMS_RESULTS_ROOT:-$REPO/results}/$RUN_ID"
YES="${MINJA_YES:-0}"
[ "${1:-}" = "--yes" ] && YES=1

# .env is parsed, never sourced: it holds URLs with characters the shell treats
# as redirection.
if [ -f "$REPO/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do   # also a last line with no newline
    line="${line%$'\r'}"
    case "$line" in ''|\#*) continue;; esac
    key="${line%%=*}"; val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
    case "$key" in OPENAI_API_KEY|OPENAI_BASE_URL|OPENROUTER_API_KEY|AMEMGUARD_MODEL|AZURE_AI_KEY|AZURE_OPENAI_ENDPOINT|AZURE_AI_KEY_MAG_SECURITY_HAOMING_RESOURCE) export "$key=$val";; esac
  done < "$REPO/.env"
fi
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY not set (looked in $REPO/.env)"; exit 1; }

# Backbone (configs/backbones.yaml names). gpt56 runs the agent and A-MemGuard on
# the Azure gpt-5.6-terra deployment. autogen 0.2 has no per-request fallback, so
# a refusal here is an error in the log, not a silent switch to OpenAI.
# UNVERIFIED end to end: the parameter patches in minja_patch.py were checked
# against the API directly, not through autogen 0.2.
case "${BACKBONE:-4omini}" in
  gpt56)
    LLM=gpt-5.6-terra
    export AMEMGUARD_MODEL=gpt-5.6-terra MINJA_REASONING_EFFORT=none
    export OPENAI_BASE_URL="${AZURE_OPENAI_ENDPOINT:?AZURE_OPENAI_ENDPOINT not set}"
    export OPENAI_API_KEY="${AZURE_AI_KEY:?AZURE_AI_KEY not set}"
    ;;
  4omini)
    # api.openai.com ran out of credit (2026-10-05); the same model is deployed on a
    # second Azure resource. autogen 0.2 has no fallback, so pick one endpoint here.
    if [ -n "${AZURE_AI_KEY_MAG_SECURITY_HAOMING_RESOURCE:-}" ]; then
      export OPENAI_BASE_URL="https://mag-security-haoming-resource.services.ai.azure.com/openai/v1"
      export OPENAI_API_KEY="$AZURE_AI_KEY_MAG_SECURITY_HAOMING_RESOURCE"
    fi
    ;;
  haiku55|qwen38flash)
    # OpenRouter, reasoning off as in configs/backbones.yaml: left on, it used up the
    # token budget and returned empty text (2026-10-11). The agent, its code generator
    # and A-MemGuard all send this body.
    case "$BACKBONE" in haiku55) LLM=anthropic/claude-haiku-5.5 ;; *) LLM=qwen/qwen3.8-flash ;; esac
    export AMEMGUARD_MODEL="$LLM" MINJA_REASONING_EFFORT=none
    export MINJA_EXTRA_BODY='{"reasoning": {"enabled": false}}'
    export OPENAI_BASE_URL=https://openrouter.ai/api/v1
    export OPENAI_API_KEY="${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set}"
    ;;
  *) echo "unknown BACKBONE=${BACKBONE}"; exit 1 ;;
esac

# MINJA is built on the autogen 0.2 agent API, which is not source-compatible
# with anything later, so it needs its own environment:
#
#   bash scripts/setup_minja_env.sh   (from the main env; builds a thin venv on /data)
#   source /data/$USER/envs/minja/bin/activate
#
# The venv holds only pyautogen 0.2.0 and openai 1.7.2; sentence-transformers (MINJA's
# retriever) and torch/transformers/scikit-learn (the A-MemGuard arm) come from the
# main environment underneath it.
python - <<'PRE' || exit 1
import sys
missing = []
for m in ("autogen", "openai", "pandas", "Levenshtein", "termcolor", "tiktoken",
          "sentence_transformers", "sklearn"):
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    sys.exit("missing in this environment: " + ", ".join(missing) +
             "\nsee the environment recipe at the top of scripts/minja_run.sh")
import autogen
if not autogen.__version__.startswith("0.2"):
    sys.exit(f"autogen {autogen.__version__} found; MINJA needs the 0.2 agent API "
             "(pip install 'pyautogen==0.2.0'). Later versions removed "
             "llm_config['functions'] and register_function(function_map=...).")
PRE

# Apply the vendored-tree patches on every run. They are idempotent, and a patch
# added after the tree was fetched must not silently fail to take effect (the
# calculator fix of 2026-10-07 did exactly that once).
python "$REPO/scripts/minja_patch.py" || { echo "minja_patch.py failed"; exit 1; }
python - "$EHR/tools/calculator.py" <<'CHK' || exit 1
import sys
if "_local_formula" not in open(sys.argv[1]).read():
    sys.exit("calculator.py is unpatched: generated code would fail on `import wolframalpha`")
CHK

export EHRSQL_ROOT="$DATA"
export EHRSQL_DB="${EHRSQL_DB:-$REPO/data/ehrsql/mimic_iii.db}"
python - "$EHRSQL_DB" <<'PY' || exit 1
import sqlite3, sys
try:
    sqlite3.connect(sys.argv[1]).execute("select count(*) from patients").fetchone()
except Exception as e:
    sys.exit(f"{sys.argv[1]} is not a usable MIMIC-III database ({e}).\n"
             "The copy inside third_party/MINJA is an unpulled git-LFS pointer; "
             "fetch it with `git -C third_party/MINJA lfs pull` or from "
             "huggingface.co/datasets/wshi83/EHRAgent-mimic_iii, and point $EHRSQL_DB at it.")
PY

mkdir -p "$OUT/bench"
python - "$OUT/config.json" <<PY
import json, subprocess, sys
sha = subprocess.run(["git","-C","$REPO","rev-parse","HEAD"], capture_output=True, text=True).stdout.strip()
minja = subprocess.run(["git","-C","$REPO/third_party/MINJA","rev-parse","HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"run_id":"$RUN_ID","git_sha":sha,"minja_sha":minja,"llm":"$LLM",
           "victim_id":$VICTIM,"target_id":$TARGET,"n_attack":$N_ATTACK,
           "n_filler":$N_FILLER,"n_test":$N_TEST,"num_shots":$NUM_SHOTS,
           "seed":$SEED,"arms":"$ARMS".split(),
           "poison_modes":"add_poison.py as released: indication modes 1 and 3, plus the clean original"},
          open(sys.argv[1],"w"), indent=2)
PY

# ---------------------------------------------------------------- benchmark
# Built once and shared by every arm, so the arms differ only in the defence.
cd "$DATA"
if [ ! -f "$OUT/bench/victim_test.json" ]; then
  MINJA_SEED="$SEED" python data_create_poison.py mimic_iii/patient_only.json \
      "$OUT/bench/attack_base.json" --csv_file mimic_iii/PATIENTS.csv \
      --num_entries "$N_ATTACK" --generate_with_id "$VICTIM"
  MINJA_SEED="$SEED" python data_create.py mimic_iii/valid_preprocessed.json \
      "$OUT/bench/filler.json" --csv_file mimic_iii/PATIENTS.csv --num_entries "$N_FILLER"
  python - "$OUT/bench" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
for name, who in (("attack_base.json", lambda n: "attacker"),
                  ("filler.json", lambda n: f"user{n % 3 + 1}")):
    rows = json.loads((d / name).read_text())
    for n, r in enumerate(rows):
        r["principal"] = who(n)
    (d / name).write_text(json.dumps(rows, indent=4))
PY
  MINJA_SEED="$SEED" python add_poison.py "$OUT/bench/attack_base.json" \
      "$OUT/bench/filler.json" "$VICTIM" "$TARGET" "$OUT/bench/poison_stream.json"
  # the same traffic with no indication sentence: the clean arm's write phase
  MINJA_SEED="$SEED" python - "$OUT/bench" <<'PY'
import json, random, sys
from pathlib import Path
d = Path(sys.argv[1])
stream = json.loads((d/"attack_base.json").read_text()) + json.loads((d/"filler.json").read_text())
random.shuffle(stream)
(d/"clean_stream.json").write_text(json.dumps(stream, indent=4))
PY
  # a fresh draw for the evaluation set, so the victim's questions are not the
  # attacker's questions
  MINJA_SEED="$(( SEED + 1000 ))" python data_create_poison.py mimic_iii/patient_only.json \
      "$OUT/bench/victim_test.json" --csv_file mimic_iii/PATIENTS.csv \
      --num_entries "$N_TEST" --generate_with_id "$VICTIM"
  python - "$OUT/bench/victim_test.json" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
for r in rows:
    r["principal"] = "victim"
json.dump(rows, open(sys.argv[1], "w"), indent=4)
PY
fi

# The typed arms need each question's required kind, classified in the main
# environment (cached, in the cost ledger) rather than in this one.
case " $ARMS " in *" typed-"*)
  [ -f "$OUT/bench/use_types.json" ] || \
    "${AMS_PY:-$REPO/.venv/bin/python}" "$REPO/scripts/minja_use_types.py" "$OUT/bench" \
      --backbone "${BACKBONE:-4omini}" ;;
esac
export MINJA_USE_TYPES="$OUT/bench/use_types.json"
export AMS_REPO="$REPO"   # minja_audit.py finds scripts/ and src/ here

# Cost guard, counted off the benchmark that was just built rather than
# predicted: add_poison.py repeats some of the benign traffic, so the stream is
# longer than the entries that went into it.
N_WRITE=$(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$OUT/bench/poison_stream.json")
N_ARMS=$(echo "$ARMS" | wc -w | tr -d ' ')
CALLS=$(( N_ARMS * (N_WRITE + N_TEST) * 3 ))
echo "run_id     $RUN_ID"
echo "model      $LLM"
echo "arms       $ARMS"
echo "questions  $N_WRITE written + $N_TEST read, per arm"
echo "calls      ~$CALLS agent calls (each A-MemGuard arm adds ~$(( (N_WRITE + N_TEST) * (NUM_SHOTS + 1) )) audit calls;"
echo "           DRIFT and LLM-sanitizer at most ~$(( (N_WRITE + N_TEST) * NUM_SHOTS * 2 )) each, MemLineage ~$(( N_WRITE + N_TEST )), RandEC none)"
if [ "$CALLS" -gt 1000 ] && [ "$YES" != "1" ]; then
  echo "above 1k calls: rerun with --yes (or MINJA_YES=1)"; exit 1
fi

# ---------------------------------------------------------------- the arms
cd "$EHR"
# Arms are independent (own memory, logs and output directory), so they run in
# parallel, ARM_JOBS at a time. Inside one arm the questions stay sequential: the
# write phase builds the memory in stream order.
#
# An arm is finished only when its marker exists. read.jsonl alone is not proof: it
# is appended one line per question, so an interrupted arm has a partial one. An
# unfinished arm is cleared and rerun from scratch (the LLM cache makes the repeated
# calls cheap), never appended to.
run_arm() {
  local arm="$1" A="$OUT/$1" stream="$OUT/bench/poison_stream.json" audit="$1"
  if [ "$arm" = "clean" ]; then stream="$OUT/bench/clean_stream.json"; audit="none"; fi
  # "<defence>@clean": the same defence over the attack-free stream, so a refused
  # call there is a false refusal of the victim's legitimate question.
  case "$arm" in *@clean) stream="$OUT/bench/clean_stream.json"; audit="${arm%@clean}" ;; esac
  {
    MINJA_AUDIT="$audit" MINJA_AUDIT_LOG="$A/audit_write.jsonl" MINJA_RESULTS="$A/write.jsonl" \
    MINJA_GATE_LOG="$A/gate_write.jsonl" \
    python main.py --llm "$LLM" --dataset mimic_iii --data_path "$stream" \
        --logs_path "$A/logs_write" --num_questions -1 --seed "$SEED" \
        --num_shots "$NUM_SHOTS" --save_memory_path "$A/memory.json"
    MINJA_AUDIT="$audit" MINJA_AUDIT_LOG="$A/audit_read.jsonl" MINJA_RESULTS="$A/read.jsonl" \
    MINJA_GATE_LOG="$A/gate_read.jsonl" \
    python main.py --llm "$LLM" --dataset mimic_iii --data_path "$OUT/bench/victim_test.json" \
        --logs_path "$A/logs_read" --num_questions -1 --seed "$SEED" \
        --num_shots "$NUM_SHOTS" --load_memory_path "$A/memory.json"
  } > "$A/run.log" 2>&1 && touch "$A/.done"
}

ARM_JOBS="${ARM_JOBS:-4}"
running=0
for arm in $ARMS; do
  A="$OUT/$arm"
  # Finished only by the marker. Arms from runs before it existed are NOT trusted:
  # that rule once skipped every arm of a run whose code never executed.
  if [ -f "$A/.done" ]; then
    echo "== $arm already done"; continue
  fi
  rm -rf "$A"; mkdir -p "$A"
  echo "== $arm: started (log $A/run.log)"
  run_arm "$arm" &
  running=$((running + 1))
  if [ "$running" -ge "$ARM_JOBS" ]; then wait -n || true; running=$((running - 1)); fi
done
wait
for arm in $ARMS; do
  [ -f "$OUT/$arm/.done" ] || echo "!! $arm did not finish: see $OUT/$arm/run.log"
done

python "$REPO/scripts/minja_report.py" "$OUT" | tee "$OUT/report.txt"
echo "results in $OUT"
