#!/usr/bin/env bash
# One-time: the environment MINJA (EHRAgent, autogen 0.2) needs.
#
# MINJA pins two packages the main environment cannot share: pyautogen 0.2.0 (its
# agent API is not source-compatible with later releases) and openai 1.7.2. Every
# heavy dependency (torch, transformers, sentence-transformers, scikit-learn, pandas)
# is already in the main environment, so this is a thin venv on top of it
# (--system-site-packages) holding only the pinned packages: tens of MB, not GB.
# The venv, pip's cache and pip's temp files all live on a large volume, never home.
#
# Run it from the MAIN environment (e.g. `conda activate sec`):
#   bash scripts/setup_minja_env.sh                    # -> /data/$USER/envs/minja
#   MINJA_ENV=/some/big/disk/minja bash scripts/setup_minja_env.sh
# Then:
#   source /data/$USER/envs/minja/bin/activate
#   AMS_PY=<main env python> YES=1 scripts/run_all.sh --stage final_minja
set -euo pipefail

ENV_DIR="${MINJA_ENV:-/data/${USER}/envs/minja}"
BASE_PY="${BASE_PY:-$(command -v python)}"
ROOT="$(dirname "$ENV_DIR")"
mkdir -p "$ROOT/pip-cache" "$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/pip-cache" TMPDIR="$ROOT/tmp"

echo "base python : $BASE_PY ($("$BASE_PY" --version))"
echo "venv        : $ENV_DIR"
echo "pip cache   : $PIP_CACHE_DIR"
"$BASE_PY" -c "import torch, transformers, sentence_transformers, sklearn, pandas" \
  || { echo "run this from the main environment (it must already have torch etc.)"; exit 1; }

[ -x "$ENV_DIR/bin/python" ] || "$BASE_PY" -m venv --system-site-packages "$ENV_DIR"
# httpx<0.28: openai 1.7.2 passes `proxies=`, which httpx 0.28 removed.
"$ENV_DIR/bin/pip" install --no-input "pyautogen==0.2.0" "openai==1.7.2" "httpx<0.28" \
    jsonlines python-Levenshtein termcolor tiktoken
"$ENV_DIR/bin/python" - <<'PY'
import autogen, httpx, openai, torch, sentence_transformers
assert autogen.__version__.startswith("0.2"), autogen.__version__
openai.OpenAI(api_key="x")  # fails here, not mid-run, if httpx is incompatible
print(f"minja env ok: autogen {autogen.__version__}, openai {openai.__version__}, "
      f"torch {torch.__version__} (shared from the main environment)")
PY
echo "next: source $ENV_DIR/bin/activate"
