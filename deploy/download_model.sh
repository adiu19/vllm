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

# Cache directory — use HF_HOME if set, else default to /workspace/hf_cache
# to keep everything on the network volume.
CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-/workspace/hf_cache}}"
mkdir -p "$CACHE_DIR"

# Disable the xet downloader (flaky on MooseFS). Use traditional HTTP path.
export HF_HUB_DISABLE_XET=1
# Don't use hf-transfer either by default — its parallelism can overwhelm
# slow network filesystems. Enable explicitly if your storage is fast.
# export HF_HUB_ENABLE_HF_TRANSFER=1

# Model-specific cache directory (HF's per-model layout)
MODEL_DIR_NAME="models--$(echo "$MODEL" | tr '/' '--')"
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

# Kill anything that might be holding file handles on the cache
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f api_server 2>/dev/null || true
pkill -9 -f WorkerProc 2>/dev/null || true
pkill -9 -f huggingface-cli 2>/dev/null || true
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

# --max-workers 2 keeps concurrent writes to MFS low, reducing hang probability.
# huggingface-cli resumes automatically — re-running this script after a failure
# picks up where it left off.
huggingface-cli download "$MODEL" \
    --cache-dir "$CACHE_DIR" \
    --max-workers 2 \
    --token "$HF_TOKEN"

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
