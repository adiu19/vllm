#!/bin/bash
# Idempotent model download into $HF_HOME/hub. Reads $MODEL from config.py.
# Skips xet + .pth duplicates; tuned for slow/unreliable network FS.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py")" || { echo "config.py failed" >&2; exit 1; }

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${MODEL:?MODEL must be set}"

# Use HF_HOME (gives $HF_HOME/hub/... layout, matches vLLM).
# Do NOT pass --cache-dir to hf — that overrides to a no-hub/ layout.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
unset HF_HUB_CACHE HF_HUB_CACHE_DIR
export HF_HUB_DISABLE_XET=1

CACHE_DIR="$HF_HOME"
MODEL_DIR_NAME="models--$(echo "$MODEL" | sed 's:/:--:g')"
MODEL_CACHE_PATH="$CACHE_DIR/hub/$MODEL_DIR_NAME"

mkdir -p "$CACHE_DIR/hub"

echo "═════════════════════════════════════════════════════════════════"
echo " Model:      $MODEL"
echo " Cache dir:  $CACHE_DIR"
echo " Model dir:  $MODEL_CACHE_PATH"
echo "═════════════════════════════════════════════════════════════════"

echo
echo "[1/4] Cleanup..."
# Be SPECIFIC with patterns. A bare `pkill -f vllm` matches this script's
# own command line (since we run from /opt/vllm-fork/), causing it to
# kill itself. Match on the actual python invocations instead.
pkill -9 -f "python.*vllm\.entrypoints" 2>/dev/null || true
pkill -9 -f "python.*WorkerProc"        2>/dev/null || true
pkill -9 -f "python.*api_server"        2>/dev/null || true
pkill -9 -f "huggingface-cli"           2>/dev/null || true
pkill -9 -f "hf download"               2>/dev/null || true
sleep 2

incomplete_count=$(find "$MODEL_CACHE_PATH" -name "*.incomplete" 2>/dev/null | wc -l || echo 0)
lock_count=$(find "$CACHE_DIR" -name "*.lock" 2>/dev/null | wc -l || echo 0)
[ "$incomplete_count" -gt 0 ] && find "$MODEL_CACHE_PATH" -name "*.incomplete" -delete 2>/dev/null || true
[ "$lock_count" -gt 0 ]       && find "$CACHE_DIR"        -name "*.lock"       -delete 2>/dev/null || true
echo "  Removed $incomplete_count incomplete, $lock_count lock file(s)."

echo
echo "[2/4] Disk..."
avail_gb=$(df -BG "$CACHE_DIR" | awk 'NR==2 {print $4}' | sed 's/G$//')
existing_size=$(du -sh "$MODEL_CACHE_PATH" 2>/dev/null | cut -f1 || echo "0")
echo "  Available: ${avail_gb}GB    Existing for this model: $existing_size"

echo
echo "[3/4] Downloading (--max-workers 2, xet disabled, .pth excluded)..."
echo

EXCLUDES=("--exclude" "original/*" "--exclude" "*.pth")
if command -v hf >/dev/null 2>&1; then
    hf download "$MODEL" --max-workers 2 --token "$HF_TOKEN" "${EXCLUDES[@]}"
else
    echo "  'hf' not found; falling back to deprecated huggingface-cli"
    huggingface-cli download "$MODEL" --max-workers 2 --token "$HF_TOKEN" "${EXCLUDES[@]}"
fi

echo
echo "[4/4] Verifying..."
remaining_incomplete=$(find "$MODEL_CACHE_PATH" -name "*.incomplete" 2>/dev/null | wc -l || echo 0)
if [ "$remaining_incomplete" -gt 0 ]; then
    echo "  $remaining_incomplete .incomplete file(s) still present. Re-run to retry."
    exit 1
fi

snapshot_dir=$(find "$MODEL_CACHE_PATH/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
if [ -z "$snapshot_dir" ]; then
    echo "  No snapshot dir at $MODEL_CACHE_PATH/snapshots/. Consider 'rm -rf $MODEL_CACHE_PATH' and retrying."
    exit 1
fi

weight_files=$(find -L "$snapshot_dir" -maxdepth 1 -name "*.safetensors" 2>/dev/null | wc -l || echo 0)
total_size=$(du -sh "$MODEL_CACHE_PATH" 2>/dev/null | cut -f1 || echo "?")

echo "  Snapshot:     $snapshot_dir"
echo "  .safetensors: $weight_files"
echo "  Total:        $total_size"
echo
echo "Done."
