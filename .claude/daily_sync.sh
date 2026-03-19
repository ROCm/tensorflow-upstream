#!/usr/bin/env bash
# ==============================================================================
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# Daily upstream sync for ROCm/tensorflow-upstream
#
# Runs the full /sync skill via Claude Code CLI.
# Assumes it is running inside a Docker container with:
#   - Claude Code installed and ANTHROPIC_API_KEY set
#   - ROCm tools, bazel, python3, etc. available
#   - The repo mounted/available at REPO_DIR
#
# Cron entry (add with `crontab -e`):
#   0 5 * * * /workspace/.claude/daily_sync.sh >> /var/log/tf_daily_sync.log 2>&1
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(dirname "$SCRIPT_DIR")}"
DATE_LABEL=$(date -u +%y%m%d)
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync_${DATE_LABEL}.log"

cleanup() {
    echo ""
    echo "=== Sync interrupted — $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
    exit 130
}
trap cleanup INT TERM

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== TF ROCm Daily Sync — $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

cd "$REPO_DIR"

# Ensure on develop-upstream and up to date
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "develop-upstream" ]]; then
    echo "Switching to develop-upstream (was on '$CURRENT_BRANCH')..."
    git checkout develop-upstream
fi
git pull --ff-only origin develop-upstream

# Skip if today's sync branch already exists
SYNC_BRANCH="develop-upstream-sync-${DATE_LABEL}"
if git rev-parse --verify "$SYNC_BRANCH" >/dev/null 2>&1 || \
   git ls-remote --heads origin "$SYNC_BRANCH" 2>/dev/null | grep -q "$SYNC_BRANCH"; then
    echo "Sync branch '$SYNC_BRANCH' already exists. Skipping."
    exit 0
fi

# Run the full sync.
# - stream-json is captured to JSONL_LOG via tee (full detail for post-mortem)
# - jq filter converts it to human-readable lines for the main LOG_FILE
JSONL_LOG="$LOG_DIR/sync_${DATE_LABEL}.jsonl"
STDERR_LOG="$LOG_DIR/sync_${DATE_LABEL}.stderr"
echo "Starting /sync $DATE_LABEL ..."
echo "Full event log: $JSONL_LOG"
set +e  # allow pipeline to fail without exiting
stdbuf -oL claude -p "/sync $DATE_LABEL" \
    --dangerously-skip-permissions \
    --model opus \
    --verbose \
    --output-format stream-json 2>"$STDERR_LOG" | \
stdbuf -oL tee "$JSONL_LOG" | \
jq --unbuffered -rj '
  if .type == "assistant" then
    (.message.content[] |
      if .type == "text" then .text + "\n"
      elif .type == "tool_use" then "[tool] " + .name + ": " + (.input | tostring | .[0:300]) + "\n"
      else empty end)
  elif .type == "result" then
    "[done] " + .subtype + " cost=$" + (.cost_usd | tostring) + "\n"
  else empty end
' 2>/dev/null
SYNC_EXIT=${PIPESTATUS[0]}
set -e

echo ""
if [[ $SYNC_EXIT -ne 0 ]]; then
    echo "claude exited with code $SYNC_EXIT"
    [[ -s "$STDERR_LOG" ]] && echo "stderr:" && cat "$STDERR_LOG"
fi
echo "=== Sync finished — $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# Report result
if git rev-parse --verify "$SYNC_BRANCH" >/dev/null 2>&1; then
    COMMIT_COUNT=$(git rev-list --count "develop-upstream..$SYNC_BRANCH")
    echo "Branch '$SYNC_BRANCH' created with $COMMIT_COUNT commit(s)."
    git log --oneline "develop-upstream..$SYNC_BRANCH"
else
    echo "WARNING: Sync branch '$SYNC_BRANCH' was not created. Check the log."
    exit 1
fi
