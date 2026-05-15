#!/bin/bash
# Download the model defined in deploy/config.sh ($MODEL) into the
# HF cache, with pre-cleanup and post-verification.
#
# Designed for MooseFS-mounted /workspace on RunPod, which is unreliable
# with HF's default `xet` chunked downloader. We force the older HTTP
# downloader and keep concurrent writes low.
#
# Idempotent: if the model is already fully cached, skips the actual
# download and just verifies. Safe to re-run after partial failures.
#
# Usage:
#   bash deploy/download_model.sh             # uses $MODEL from config.sh
#   MODEL=foo/bar bash deploy/download_model.sh   # override model

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

# ─────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────
: "${HF_TOKEN:?HF_TOKEN must be set (sourced from config.sh or pod env)}"
: "${MODEL:?MODEL must be set (sourced from config.sh)}"

# Cache layout:
#   HF stores models at: $HF_HOME/hub/models--org--repo/...
#   The `hub/` subdir is added automatically when HF_HOME is set.
#
# CRITICAL: do NOT pass `--cache-dir` to `hf download`. That flag is
# equivalent to setting HF_HUB_CACHE — it disables the `hub/` subdir
# convention and writes files directly under the given path. vLLM
# expects the `hub/` convention, so a `--cache-dir` invocation creates
# a duplicate model tree at the wrong path that vLLM can't find.
#
# Right approach: set HF_HOME, omit --cache-dir.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
CACHE_DIR="$HF_HOME"
mkdir -p "$CACHE_DIR"
mkdir -p "$CACHE_DIR/hub"

# Disable the xet downloader (flaky on MooseFS). Use traditional HTTP path.
export HF_HUB_DISABLE_XET=1
# Don't use hf-transfer either by default — its parallelism can overwhelm
# slow network filesystems. Enable explicitly if your storage is fast.
# export HF_HUB_ENABLE_HF_TRANSFER=1

# HF cache env var hygiene: HF_HOME and HF_HUB_CACHE produce DIFFERENT
# on-disk layouts ($HF_HOME/hub/... vs $HF_HUB_CACHE/... without hub/).
# If both are set to the same path, HF creates duplicate model trees,
# wasting GB. Unset HF_HUB_CACHE so HF_HOME's "with hub/" layout wins
# and --cache-dir resolves the same way.
unset HF_HUB_CACHE
unset HF_HUB_CACHE_DIR

# Model-specific cache directory (HF's per-model layout: "org/repo" → "models--org--repo")
# NOTE: tr can't expand '/' to two characters; use sed for the double-dash.
MODEL_DIR_NAME="models--$(echo "$MODEL" | sed 's:/:--:g')"
MODEL_CACHE_PATH="$CACHE_DIR/hub/$MODEL_DIR_NAME"

echo "═════════════════════════════════════════════════════════════════"
echo " Model download"
echo "═════════════════════════════════════════════════════════════════"
echo " Model:      $MODEL"
echo " Cache dir:  $CACHE_DIR"
echo " Model dir:  $MODEL_CACHE_PATH"
echo "═════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────
# Cleanup: kill stale vLLM processes that might hold files open
# ─────────────────────────────────────────────────────────────────────────
echo
echo "[1/4] Cleaning up stale processes and partial download state..."

# Kill anything that might be holding file handles on the cache.
# Cover both legacy `huggingface-cli download` and new `hf download`.
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
pkill -9 -f WorkerProc 2>/dev/null || true
pkill -9 -f huggingface-cli 2>/dev/null || true
pkill -9 -f "hf download" 2>/dev/null || true
pkill -9 -f hf-download 2>/dev/null || true
sleep 2

# Remove partial / locked files that confuse the downloader
incomplete_count=$(find "$MODEL_CACHE_PATH" -name "*.incomplete" 2>/dev/null | wc -l || echo 0)
lock_count=$(find "$CACHE_DIR" -name "*.lock" 2>/dev/null | wc -l || echo 0)

if [ "$incomplete_count" -gt 0 ]; then
    echo "  Removing $incomplete_count .incomplete file(s)..."
    find "$MODEL_CACHE_PATH" -name "*.incomplete" -delete 2>/dev/null || true
fi
if [ "$lock_count" -gt 0 ]; then
    echo "  Removing $lock_count .lock file(s)..."
    find "$CACHE_DIR" -name "*.lock" -delete 2>/dev/null || true
fi
echo "  Cleanup done."

# ─────────────────────────────────────────────────────────────────────────
# Disk-space check (best-effort; MFS quota isn't visible from inside)
# ─────────────────────────────────────────────────────────────────────────
echo
echo "[2/4] Disk-space check..."

avail_gb=$(df -BG "$CACHE_DIR" | awk 'NR==2 {print $4}' | sed 's/G$//')
echo "  /workspace mount reports ${avail_gb}GB available (note: per-pod quota may be much smaller)"
existing_size=$(du -sh "$MODEL_CACHE_PATH" 2>/dev/null | cut -f1 || echo "0")
echo "  Existing cache for this model: $existing_size"

# ─────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────
echo
echo "[3/4] Starting download with HF_HUB_DISABLE_XET=1, max-workers=2..."
echo "  (Resumes from any successfully-cached files; only fetches what's missing.)"
echo

# --max-workers 2 keeps concurrent writes to MFS low, reducing hang
# probability. hf resumes automatically — re-running this script after
# a failure picks up where it left off.
#
# NOTE on --cache-dir: deliberately omitted. With HF_HOME set above, HF
# uses $HF_HOME/hub/ by default — which matches vLLM's expectations and
# avoids creating a duplicate model tree at the no-`hub/` path.
#
# Use the new `hf` CLI; the legacy `huggingface-cli` is deprecated.
# `--exclude` skips Meta's original PyTorch checkpoint duplicates
# (consolidated.XX.pth) — those are the same weights as the safetensors
# files we're already downloading, costing another ~140GB for nothing.
EXCLUDE_PATTERNS=("--exclude" "original/*" "--exclude" "*.pth")

if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL" \
        --max-workers 2 \
        --token "$HF_TOKEN" \
        "${EXCLUDE_PATTERNS[@]}"
else
    echo "  Warning: 'hf' CLI not found, falling back to deprecated huggingface-cli"
    huggingface-cli download "$MODEL" \
        --max-workers 2 \
        --token "$HF_TOKEN" \
        "${EXCLUDE_PATTERNS[@]}"
fi

# ─────────────────────────────────────────────────────────────────────────
# Post-verification
# ─────────────────────────────────────────────────────────────────────────
echo
echo "[4/4] Verifying download..."

# Check for any leftover .incomplete files (download wasn't fully resumed)
remaining_incomplete=$(find "$MODEL_CACHE_PATH" -name "*.incomplete" 2>/dev/null | wc -l || echo 0)
if [ "$remaining_incomplete" -gt 0 ]; then
    echo "  WARNING: $remaining_incomplete .incomplete file(s) still present."
    echo "  Re-run this script to retry the missing chunks."
    exit 1
fi

# Find the snapshot directory (HF's canonical model files location)
snapshot_dir=$(find "$MODEL_CACHE_PATH/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
if [ -z "$snapshot_dir" ]; then
    echo "  WARNING: no snapshot directory found under $MODEL_CACHE_PATH/snapshots/"
    echo "  Cache may be corrupted. Consider 'rm -rf $MODEL_CACHE_PATH' and retrying."
    exit 1
fi

# Count weight files and report total size
weight_files=$(find -L "$snapshot_dir" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
total_size=$(du -sh "$MODEL_CACHE_PATH" 2>/dev/null | cut -f1 || echo "?")

echo "  Snapshot:       $snapshot_dir"
echo "  .safetensors:   $weight_files file(s)"
echo "  Total on-disk:  $total_size"
echo
echo "Download complete. Start vLLM with: bash deploy/start_prefill_nixl.sh"
