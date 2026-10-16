#!/usr/bin/env bash
# Fetch upstream baseline implementations at pinned commits.
#
# These are other people's repositories. We vendor them read-only, reuse their
# code as published, and never edit in place — an adapter on our side absorbs
# any interface mismatch, so their results stay attributable to them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

fetch() {  # name url sha
  local name="$1" url="$2" sha="$3"
  if [ -d "third_party/$name/.git" ]; then
    echo "third_party/$name already present"; return
  fi
  git clone "$url" "third_party/$name"
  git -C "third_party/$name" checkout --quiet "$sha"
  echo "third_party/$name @ $sha"
}

# A-MemGuard — consensus validation + dual-memory lessons for agent memory.
# MIT licence, Copyright (c) 2024 AI Secure.
fetch AMemGuard https://github.com/TangciuYueng/AMemGuard dd92f7ff21b9a904a703141be3d5b80170e57228

# TMA-NM — non-malleable origin-bound authority for agent memory; v2/ is the
# artifact for their current paper. MIT licence.
fetch mem-inv-bench https://github.com/yedidel/mem-inv-bench 050b2ea5d6560c40f4eec7e8b9afd03e0224d877

# MINJA — query-only memory injection (NeurIPS 2025). MIT licence.
# Its EHR databases are git-LFS objects; `git lfs pull` fetches them, and the
# same bytes are on Hugging Face (wshi83/EHRAgent-{eicu,mimic_iii}, apache-2.0)
# if LFS is unavailable. They are PhysioNet-derived: keep them out of git.
fetch MINJA https://github.com/dsh3n77/MINJA a3ec8da
if [ -d third_party/MINJA ] && command -v git-lfs >/dev/null 2>&1; then
  git -C third_party/MINJA lfs pull 2>/dev/null && echo "  MINJA: LFS objects fetched"
fi

# MINJA ships the author's own paths and a placeholder API key, so it does not
# run as checked out. The patch is idempotent; re-run it after any fresh clone.
python3 "$(dirname "$0")/minja_patch.py"

# LoCoMo (Maharana et al., ACL 2024), the released 10-conversation file.
mkdir -p data/locomo
[ -f data/locomo/locomo10.json ] || curl -fsSL -o data/locomo/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# RandEC/DistilBERT weights for A-MemGuard's `distil` baseline: the
# certified-llm-safety release (Dropbox link in its README). ~800 MB unzipped.
W=third_party/AMemGuard/certified-llm-safety
if [ -d "$W" ] && [ ! -f "$W/models/distilbert_suffix.pt" ]; then
  curl -fsSL -o "$W/models.zip" \
    'https://www.dropbox.com/scl/fi/ux4ew8y88uslu5064r2xh/models.zip?rlkey=4bo1njpnj4nc801tw1pkby52o&dl=1' \
    && unzip -oq "$W/models.zip" -d "$W" && rm "$W/models.zip"
fi
