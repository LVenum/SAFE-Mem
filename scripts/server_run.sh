#!/usr/bin/env bash
# Launch a long experiment inside tmux on the school server.
#
#   ./scripts/server_run.sh preflight       --model azure:gpt-4o-mini --yes
#   ./scripts/server_run.sh controllability --model azure:gpt-4o-mini --yes
#   ./scripts/server_run.sh estimator       --model azure:gpt-4o-mini --yes
#
# Everything is resumable: responses are cached by content hash, so re-running
# after a crash, a disconnect, or a killed session costs nothing for the work
# already done. That is why this script does not try to checkpoint anything
# itself — the cache is the checkpoint.
set -euo pipefail

cd "$(dirname "$0")/.."

EXPERIMENT="${1:?usage: server_run.sh <mpbench|mpbench_raw> [args...]}"
shift || true

case "$EXPERIMENT" in
  mpbench)
    SCRIPT=scripts/run_mpbench.py;              CONFIG=configs/mpbench.yaml ;;
  mpbench_raw)
    SCRIPT=scripts/run_mpbench.py;              CONFIG=configs/mpbench_raw.yaml ;;
  *)
    echo "unknown experiment '$EXPERIMENT'" >&2
    echo "want: mpbench, mpbench_raw" >&2
    exit 2 ;;
esac

SESSION="ams-${EXPERIMENT}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p logs
LOG="logs/${EXPERIMENT}-${STAMP}.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found; running in the foreground instead" >&2
  exec uv run --extra local python "$SCRIPT" --config "$CONFIG" "$@" 2>&1 | tee "$LOG"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists. Attach with:  tmux attach -t $SESSION" >&2
  echo "Kill it first if you meant to start over:        tmux kill-session -t $SESSION" >&2
  exit 1
fi

# `; bash` keeps the pane alive after the run so a failure is still readable.
tmux new-session -d -s "$SESSION" \
  "uv run --extra local python $SCRIPT --config $CONFIG $* 2>&1 | tee $LOG; echo; echo '--- finished, pane held open ---'; bash"

echo "started tmux session '$SESSION'"
echo "  attach:  tmux attach -t $SESSION     (detach with ctrl-b d)"
echo "  follow:  tail -f $LOG"
echo "  stop:    tmux kill-session -t $SESSION"