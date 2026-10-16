#!/usr/bin/env bash
# The whole experiment, in the order the design document fixes.
#
# Stage 0 costs nothing and must pass before anything else runs. Stage 1 is the
# go/no-go check of spec §7.5 and this script STOPS if it fails, because the
# answer there is to redesign, not to tune. Every paid runner prints its
# projected call count first and refuses above ~1k calls without YES=1
# (CLAUDE.md rule 14), so a bare run is a cost preview that stops at the first
# paid step.
#
#   YES=1 scripts/run_all.sh                 # stages 0-5 and 8, stopping at a NO-GO
#                                            # finished arms are skipped; it resumes
#                                            # at the one that broke. FORCE=1 to redo.
#   YES=1 scripts/run_all.sh --stage 2       # one stage
#   LIMIT=40 YES=1 scripts/run_all.sh        # same shape on 40 MPBench cases / 1 LoCoMo conversation
#
# Opt-in stages, never part of a bare run:
#   final  THE FINAL SWEEP (spec §10): every experiment on both backbones,
#          gpt-5.6-terra (Azure first) and then gpt-4o-mini. Resumable.
#   final_minja  MINJA on both backbones; run it inside the minja conda env
#   scale_check  spec §11 at the current scale (HNSW vs exact, entity hops); opt-in
#   final31  v3.1 rerun of our rows (baselines unchanged, reused from `final`)
#   adaptive adaptive attacks 1, 2, 4, 5 (spec §7.8, §10.8): ours vs no defence
#   v2   round v2 on gpt-4o-mini: answer_typed v2 + retention check (OpenAI)
#   v2g  round v2 on gpt-5.6-terra, Azure first, same-model OpenAI fallback
#   6   MINJA / EHRAgent-MIMIC-III            its own conda env (autogen 0.2), see scripts/minja_run.sh
# AgentPoison is not run (decided 2026-10-02): its trigger optimisation needs a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

# The project .venv if there is one, otherwise the active environment (conda).
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
fi
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"
YES_ARG=""
[ "${YES:-0}" = "1" ] && YES_ARG="--yes"
ONLY=""
[ "${1:-}" = "--stage" ] && ONLY="${2:-}"
want() { case "$ONLY" in final|final31|adaptive|final_minja|scale_check|v2|v2g|6) return 1 ;; esac; [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
only() { [ "$ONLY" = "$1" ]; }
run() { echo; echo "=== $* "; "$@"; }
# -- resume ---------------------------------------------------------------------
# `finalise()` writes manifest.json last, so its presence marks a run as
# finished. A bare sweep skips the arms that completed and picks up at the one
# that broke. Re-running a finished arm would be free -- the response cache is
# content-addressed and shared, so it replays at $0.00 -- but it still costs
# wall-clock and leaves a duplicate directory.
#
# The decision is delegated to scripts/run_done.py, which compares the RESOLVED
# CONFIG against each finished run's manifest. Content, not timestamps: `git
# pull` and `rsync` both refresh a config's mtime, so a timestamp rule would
# declare every prior run stale at exactly the moment the code is synced to the
# server and a sweep is about to be resumed.
#
#   FORCE=1   run everything regardless of what is already there.
config_of() { while [ $# -gt 0 ]; do [ "$1" = "--config" ] && { echo "$2"; return; }; shift; done; }
done_already() {
  [ "${FORCE:-0}" = "1" ] && return 1
  $PY scripts/run_done.py "$@" 2>/dev/null
}

mp() {
  local c; c=$(config_of "$@")
  # The full argument list goes to the guard, so an arm that differs only by
  # --defences or --pool-size is correctly treated as a different arm.
  [ -n "$c" ] && done_already "$c" "$@" $LIMIT_ARG && return 0
  run $PY scripts/run_mpbench.py $YES_ARG $LIMIT_ARG "$@"
}
a6() {
  local c; c=$(config_of "$@")
  [ -n "$c" ] && done_already "$c" "$@" && return 0
  run $PY scripts/run_attack6.py $YES_ARG "$@"
}
lc() {
  local c; c=$(config_of "$@")
  local l=""; [ -n "${LIMIT:-}" ] && l="--limit 1"
  [ -n "$c" ] && done_already "$c" "$@" $l && return 0
  run $PY scripts/run_locomo.py $YES_ARG $l "$@"
}

# -- where results go ----------------------------------------------------------
# A full sweep writes hundreds of megabytes of self-describing run directories
# and the repo usually sits on a small home partition. Two independent knobs:
#
#   RESULTS_ROOT  where runs are WRITTEN. Exported as $AMS_RESULTS_ROOT, which
#                 every script honours via RunContext, so this redirects the whole
#                 pipeline and not just this driver.
#   ARCHIVE_DIR   where finished runs are COPIED after each paid stage, as a
#                 second copy on a volume that outlives the working tree.
#                 ARCHIVE_DIR=none turns it off.
#
# Both default to a large volume when one exists and is writable, and to the repo
# otherwise, so the same command works on a laptop and on the server unflagged.
# A big volume is commonly root-owned with a per-user directory inside it
# (/data drwxr-xr-x root, /data/$USER yours), so test the per-user path as well as
# the mount point itself. Testing only the mount point is what made this fall back
# to a 97%-full home partition on the first server it met.
BIG=""
for cand in /data /mnt/data /scratch /work; do
  [ -d "$cand" ] || continue
  for probe in "$cand/${USER:-ams}" "$cand"; do
    if [ -d "$probe" ] && [ -w "$probe" ]; then BIG="$cand/${USER:-ams}/ams-results"; break 2; fi
  done
done
RESULTS_ROOT="${RESULTS_ROOT:-${BIG:-$PWD/results}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${BIG:-none}}"
# Falling back is better than aborting a sweep over a permission error, but it
# must be loud: silently writing hundreds of megabytes to the wrong volume is how
# a disk fills up mid-run.
if ! mkdir -p "$RESULTS_ROOT" 2>/dev/null; then
  echo "WARNING: cannot create $RESULTS_ROOT -- falling back to $PWD/results"
  RESULTS_ROOT="$PWD/results"; ARCHIVE_DIR=none
  mkdir -p "$RESULTS_ROOT"
fi
export AMS_RESULTS_ROOT="$RESULTS_ROOT"
# Keep the response cache on the same volume as the results unless told otherwise.
# It is shared across runs and grows without bound, so it is the other thing that
# fills a small home partition.
if [ -z "${AMS_CACHE_PATH:-}" ] && [ "$RESULTS_ROOT" != "$PWD/results" ]; then
  export AMS_CACHE_PATH="$(dirname "$RESULTS_ROOT")/llm_cache/responses.sqlite"
  mkdir -p "$(dirname "$AMS_CACHE_PATH")" 2>/dev/null || unset AMS_CACHE_PATH
fi
RES="$RESULTS_ROOT"

echo "results  -> $RESULTS_ROOT"
[ -n "${AMS_CACHE_PATH:-}" ] && echo "cache    -> $AMS_CACHE_PATH"
# Free space where the results are about to go. A sweep writes hundreds of MB and
# the cache keeps growing, so a nearly-full volume is worth knowing about before
# the first paid call rather than after.
if command -v df >/dev/null 2>&1; then
  echo "space    -> $(df -h "$RESULTS_ROOT" 2>/dev/null | awk 'NR==2 {print $4" free of "$2" ("$5" used)"}')"
fi
if [ "$ARCHIVE_DIR" = "none" ] || [ "$ARCHIVE_DIR" = "$RESULTS_ROOT" ]; then
  ARCHIVE_DIR=none
  echo "archive  -> (none; results already sit on their final volume)"
else
  echo "archive  -> $ARCHIVE_DIR"
fi
# Only a note when the defaults actually fell back, not when a path was given.
if [ -z "$BIG" ] && [ "$RESULTS_ROOT" = "$PWD/results" ]; then
  echo "note: found no writable volume (tried /data, /mnt/data, /scratch, /work and a"
  echo "      \$USER subdirectory of each), so results stay in the repo. To force it:"
  echo "        RESULTS_ROOT=/data/\$USER/ams-results scripts/run_all.sh"
fi

# Run directories are never edited once written (CLAUDE.md rule 10), so an
# incremental rsync is both correct and cheap. A failed copy is reported and does
# NOT abort the sweep: the primary copy is already on disk, and losing an hour of
# paid compute to a full archive volume would be the worse outcome.
archive() {
  [ "$ARCHIVE_DIR" = "none" ] && return 0
  mkdir -p "$ARCHIVE_DIR" 2>/dev/null || { echo "archive: cannot create $ARCHIVE_DIR, skipped"; return 0; }
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "$RESULTS_ROOT"/ "$ARCHIVE_DIR"/ && echo "archived -> $ARCHIVE_DIR" \
      || echo "archive: rsync failed; the primary copy in $RESULTS_ROOT is intact"
  else
    cp -R "$RESULTS_ROOT"/. "$ARCHIVE_DIR"/ && echo "archived -> $ARCHIVE_DIR" \
      || echo "archive: cp failed; the primary copy in $RESULTS_ROOT is intact"
  fi
}
trap archive EXIT

# -- 0. structural guarantees, offline, free -----------------------------------
if want 0; then
  # `tests/` and `docs/` are gitignored as local-only working material, so a
  # machine that got the code through git has neither. The pipeline check below
  # still runs; the guarantees the suite pins simply cannot be verified there,
  # and that is said out loud rather than passing silently.
  LINT_TARGETS="src/ scripts/"
  if [ -d tests ]; then
    run $PY -m pytest tests/ -q
    LINT_TARGETS="src/ tests/ scripts/"
  else
    echo
    echo "=== tests/ is absent on this machine (it is gitignored), so the structural"
    echo "    guarantees are NOT verified here. They are what pins the adaptive attacks"
    echo "    of spec §7.8 that must fail by construction, the call-site gate, and the"
    echo "    MemLineage fail-open. Run the suite wherever tests/ exists before"
    echo "    trusting a result from this machine."
  fi
  run $PY -m ruff check $LINT_TARGETS
  run $PY scripts/check_repetition_sources.py
  run $PY scripts/run_mpbench.py --config configs/smoke_typed.yaml --limit 8
  echo
  if [ -d tests ]; then
    echo "Stage 0 green: the adaptive attacks of spec §7.8 that are predicted to fail"
    echo "by construction do fail, the call-site gate and the MemLineage fail-open are"
    echo "pinned, and the pipeline runs end to end on the echo provider. Nothing here"
    echo "has cost anything."
  else
    echo "Stage 0 green as far as it goes: lint is clean and the pipeline runs end to"
    echo "end on the echo provider, at no cost. The test suite was skipped (see above)."
  fi
fi

# -- 1. go / no-go (passed 2026-10-02: 4.1% description vs a 25% threshold) -----
if want 1; then
  echo; echo "=== go/no-go (spec §7.5): does MPBench need anything above description?"
  if ! $PY scripts/check_use_type.py --config configs/mpbench_typed.yaml --sample 200; then
    echo
    echo "NO-GO. Most adversarial queries need only a description, so the"
    echo "use-dependent gate cannot carry MPBench. Stop here and revisit the design"
    echo "(spec §2.1). Do not tune the classifier to get past this."
    exit 2
  fi
fi

# -- 2. the main MPBench table -------------------------------------------------
# TMA-NM is cited, not run (superseded by MemLineage); its adapter still works:
# add tma_nm_strict,tma_nm_values,tma_nm_full to --defences to bring it back.
if want 2; then
  mp --config configs/mpbench_typed.yaml
  mp --config configs/mpbench_typed_withhold.yaml
  mp --config configs/mpbench.yaml --defences no_defence,a_memguard --pool-size 6240
  archive
fi

# -- 2b. the remaining §7.6 baselines ------------------------------------------
# randec_distil (DistilBERT) is a local CPU model; its
# weights come from scripts/fetch_third_party.sh.
if want 2b; then
  mp --config configs/mpbench_baselines.yaml
  mp --config configs/mpbench_drift.yaml
  archive
fi

# -- 3. ablation == the neighbour comparison (spec §7.7) -----------------------
if want 3; then
  for cell in g5_query g5_fixed g2_query g2_fixed; do
    mp --config "configs/ablation_${cell}.yaml"
  done
  mp --config configs/mpbench_typed_cap1.yaml
  mp --config configs/mpbench_typed_noscope.yaml   # prediction 5, MPBench half
  archive
fi

# -- final. Every experiment on both backbones (user, 2026-10-05): gpt-5.6-terra
#    first, then gpt-4o-mini. Within a backbone, independent CHAINS run in
#    parallel; inside a chain, arms run in order. A chain groups the arms that
#    share their writes, so the first arm pays for the writes and the rest replay
#    them from the cache instead of racing to pay twice. Each chain logs to
#    $RES/logs/<backbone>_<chain>.log. Finished arms are skipped, so rerunning the
#    same command resumes. LongMemEval is held out and NOT run here (spec §10.4).
#    BACKBONES=gpt56 / BACKBONES=4omini runs one backbone (e.g. two terminals).
chain_ours_mp() {   # v3 main row pays the shared writes; the rest then fan out
  mp --config configs/v3_mpbench_withhold.yaml --backbone "$1"
  local pids="" c rc=0
  for c in v3_mpbench_confine v3_mpbench_withhold_m2 v3_mpbench_withhold_m8 \
           v3_mpbench_g2_query v3_mpbench_g5_fixed v3_mpbench_g2_fixed \
           v3_mpbench_noscope v3_mpbench_cap1; do
    mp --config "configs/$c.yaml" --backbone "$1" > "$RES/logs/${1}_ours_mp_$c.log" 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || rc=1; done
  return $rc
}
chain_base_mp() {   # simple_extract writes, shared by every content baseline
  mp --config configs/mpbench_baselines.yaml --backbone "$1"
  mp --config configs/mpbench.yaml --defences no_defence,a_memguard --pool-size 6240 --backbone "$1"
}
chain_drift_mp() { mp --config configs/mpbench_drift.yaml --backbone "$1"; }
chain_lc_a() {
  lc --config configs/v3_locomo_withhold_a.yaml --backbone "$1"
  lc --config configs/v3_locomo_confine_a.yaml --backbone "$1"
}
chain_lc_b() {
  lc --config configs/v3_locomo_withhold_b.yaml --backbone "$1"
  lc --config configs/v3_locomo_confine_b.yaml --backbone "$1"
}
chain_lc_inert() { lc --config configs/v3_locomo_gate_inert.yaml --backbone "$1"; }
chain_lc_base() {   # carries the full-context column
  lc --config configs/locomo_baselines.yaml --backbone "$1"
  lc --config configs/locomo_drift.yaml --backbone "$1"
}
chain_attack6() {
  a6 --config configs/attack6_v2.yaml --backbone "$1"
  a6 --config configs/attack6_v3.yaml --backbone "$1"
}
CHAINS="${CHAINS:-ours_mp base_mp drift_mp lc_a lc_b lc_inert lc_base attack6}"

if only final; then
  mkdir -p "$RES/logs"
  failed=""
  for bb in ${BACKBONES:-gpt56 4omini}; do
    echo; echo "################ backbone: $bb  (chains in parallel: $CHAINS) ################"
    pids=""
    for ch in $CHAINS; do
      ( "chain_$ch" "$bb" ) > "$RES/logs/${bb}_${ch}.log" 2>&1 &
      pids="$pids $!:$ch"
      echo "  started $ch  -> $RES/logs/${bb}_${ch}.log"
    done
    for pc in $pids; do
      if wait "${pc%%:*}"; then echo "  done    ${pc#*:} ($bb)"
      else echo "  FAILED  ${pc#*:} ($bb) -- see $RES/logs/${bb}_${pc#*:}.log"; failed="$failed ${bb}_${pc#*:}"
      fi
    done
    archive
  done
  [ -z "$failed" ] || { echo; echo "chains that failed:$failed  (rerun the same command to resume)"; exit 1; }
fi

# -- final31. v3.1 (spec §10.7): our MPBench and LoCoMo rows again, with external
#    text kept out of the evidence slots, and the corrected LoCoMo control row.
#    Writes, and every answer whose prompt did not change, replay from the cache.
chain_ours_mp31() {
  mp --config configs/v31_mpbench_withhold.yaml --backbone "$1"
  local pids="" c rc=0
  for c in v31_mpbench_confine v31_mpbench_withhold_m2 v31_mpbench_withhold_m8 \
           v31_mpbench_g2_query v31_mpbench_g5_fixed v31_mpbench_g2_fixed \
           v31_mpbench_noscope v31_mpbench_cap1; do
    mp --config "configs/$c.yaml" --backbone "$1" > "$RES/logs/${1}_ours_mp31_$c.log" 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || rc=1; done
  return $rc
}
chain_lc31_a() {
  lc --config configs/v31_locomo_withhold_a.yaml --backbone "$1"
  lc --config configs/v31_locomo_confine_a.yaml --backbone "$1"
}
chain_lc31_b() {
  lc --config configs/v31_locomo_withhold_b.yaml --backbone "$1"
  lc --config configs/v31_locomo_confine_b.yaml --backbone "$1"
}
chain_lc31_inert() { lc --config configs/v31_locomo_gate_inert.yaml --backbone "$1"; }

if only final31; then
  mkdir -p "$RES/logs"
  failed=""
  for bb in ${BACKBONES:-gpt56 4omini}; do
    echo; echo "################ v3.1, backbone: $bb ################"
    pids=""
    for ch in ${CHAINS31:-ours_mp31 lc31_a lc31_b lc31_inert}; do
      ( "chain_$ch" "$bb" ) > "$RES/logs/${bb}_${ch}.log" 2>&1 &
      pids="$pids $!:$ch"
      echo "  started $ch  -> $RES/logs/${bb}_${ch}.log"
    done
    for pc in $pids; do
      if wait "${pc%%:*}"; then echo "  done    ${pc#*:} ($bb)"
      else echo "  FAILED  ${pc#*:} ($bb) -- see $RES/logs/${bb}_${pc#*:}.log"; failed="$failed ${bb}_${pc#*:}"
      fi
    done
  done
  [ -z "$failed" ] || { echo; echo "chains that failed:$failed  (rerun the same command to resume)"; exit 1; }
fi

# -- adaptive. Spec §7.8 attacks 1, 2, 4, 5 (3 is not run: it needs a person to say
#    yes). Each rewrites every adversarial context; benign writes replay from the
#    cache. Ours (v3.1 withhold) and no defence run in parallel, one chain each.
if only adaptive; then
  mkdir -p "$RES/logs"
  failed=""
  for bb in ${BACKBONES:-4omini}; do
    pids=""
    for sys in ours nodef; do
      (
        for v in forged_author forged_retention under_report conflict_evasion; do
          mp --config "configs/adaptive_${v}_${sys}.yaml" --backbone "$bb"
        done
      ) > "$RES/logs/${bb}_adaptive_${sys}.log" 2>&1 &
      pids="$pids $!:$sys"
      echo "  started adaptive/$sys -> $RES/logs/${bb}_adaptive_${sys}.log"
    done
    for pc in $pids; do
      wait "${pc%%:*}" && echo "  done    adaptive/${pc#*:} ($bb)" \
        || { echo "  FAILED  adaptive/${pc#*:} ($bb)"; failed="$failed ${pc#*:}"; }
    done
  done
  [ -z "$failed" ] || exit 1
fi

# -- scale_check. Spec §11 at today's scale: does the scalable retrieval path serve
#    what the exact one serves? Writes replay from the final sweep's cache. Opt-in,
#    gpt-4o-mini unless BACKBONES says otherwise.
if only scale_check; then
  for bb in ${BACKBONES:-4omini}; do
    mp --config configs/v3_mpbench_withhold_hnsw.yaml --backbone "$bb"
    lc --config configs/v3_locomo_withhold_a_hnsw.yaml --backbone "$bb"
    lc --config configs/v3_locomo_withhold_a_h1_merge.yaml --backbone "$bb"
  done
fi

# -- final_minja. MINJA on EHRAgent/MIMIC-III, both backbones. Run INSIDE the minja
#    env (scripts/setup_minja_env.sh, autogen 0.2). gpt56 path unverified end to end.
if only final_minja; then
  # gpt-4o-mini only by default: the gpt-5.6 Azure deployment is gone (2026-10-07).
  for bb in ${BACKBONES:-4omini}; do
    run env BACKBONE="$bb" bash scripts/minja_run.sh ${YES_ARG}
  done
fi

# -- v2. round v2 (2026-10-04): answer_typed v2 + retention check, gpt-4o-mini.
if only v2; then
  mp --config configs/mpbench_typed_v2.yaml
  mp --config configs/mpbench_typed_withhold_v2.yaml
fi

# -- v2g. round v2 on gpt-5.6-terra (memory + agent; judge stays gpt-4o-mini).
#         Azure first; the manifest records the share each endpoint served.
if only v2g; then
  mp --config configs/mpbench_typed_v2_gpt56.yaml
  mp --config configs/mpbench_typed_withhold_v2_gpt56.yaml
  mp --config configs/mpbench_v2_gpt56.yaml
fig configs/mpbench_typed_gpt5.yaml
  mp --config configs/mpbench.yaml --defences no_defence,a_memguard --pool-size 6240
fi

# -- 4. rescore the benign arms of runs made before the polarity fix -----------
if want 4; then
  for d in "$RES"/*typed*/ "$RES"/*ablation*/ "$RES"/mpbench_pool*/; do
    [ -f "$d/mpbench_outcomes.csv" ] && run $PY scripts/rescore_benign.py "$d"
  done
fi

# -- 5. LoCoMo beside the full-context column (spec §7.3), plus its ablations --
if want 5; then
  lc --config configs/locomo_typed.yaml          # carries the full-context column
  lc --config configs/locomo_simple.yaml
  for c in locomo_typed_withhold locomo_typed_h1 locomo_typed_h2 \
           locomo_ablation_g5_fixed locomo_ablation_g2_query locomo_ablation_g2_fixed; do
    lc --config "configs/$c.yaml"
  done
  archive
fi

# -- 6. MINJA on EHRAgent/MIMIC-III: the scope result. Opt-in: own conda env. --
if only 6; then
  echo "=== activate the minja conda env first (recipe at the top of scripts/minja_run.sh)"
  run bash scripts/minja_run.sh ${YES_ARG}
fi

# -- 8. the tables -------------------------------------------------------------
if want 8; then
  # Only finished runs: a crashed one has no manifest.json and would otherwise win on name.
  latest() { ls -d "$RES"/"$1"-2*/manifest.json 2>/dev/null | tail -1 | xargs -r dirname; }
  DIRS=""
  for n in mpbench_typed_pool6240 mpbench_pool6240 mpbench_typed_withhold_pool6240 \
           mpbench_baselines_pool6240 mpbench_drift_pool6240 \
           ablation_g5_query_pool6240 ablation_g5_fixed_pool6240 ablation_g2_query_pool6240 \
           ablation_g2_fixed_pool6240 mpbench_typed_cap1_pool6240 mpbench_typed_noscope_pool6240 \
           locomo_typed locomo_simple locomo_typed_withhold locomo_typed_h1 locomo_typed_h2 \
           locomo_ablation_g5_fixed locomo_ablation_g2_query locomo_ablation_g2_fixed; do
    d=$(latest "$n"); [ -n "$d" ] && DIRS="$DIRS $d"
  done
  if [ -n "$DIRS" ]; then
    mkdir -p reports
    out="reports/tables-$(date -u +%Y%m%dT%H%M%SZ).md"
    $PY scripts/report_tables.py $DIRS | tee "$out"
    echo; echo "tables: $out"
  fi
fi

echo; echo "done. $RESULTS_ROOT holds one self-describing directory per run."
