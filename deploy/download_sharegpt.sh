#!/bin/bash
# Idempotent ShareGPT dataset download. Used by the FlowPrefill load gen.
# Canonical source: anon8231489123/ShareGPT_Vicuna_unfiltered on HF,
# file ShareGPT_V3_unfiltered_cleaned_split.json (~600MB).

set -e

DATASET_DIR="${SHAREGPT_DIR:-/workspace/datasets/sharegpt}"
DATASET_FILE="ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/${DATASET_FILE}"
TARGET="${DATASET_DIR}/${DATASET_FILE}"
MIN_SIZE_BYTES=$((500 * 1024 * 1024))  # 500MB sanity floor

echo "═════════════════════════════════════════════════════════════════"
echo " Dataset:  ShareGPT V3 (unfiltered, cleaned, split)"
echo " Target:   $TARGET"
echo "═════════════════════════════════════════════════════════════════"

mkdir -p "$DATASET_DIR"

if [ -f "$TARGET" ]; then
    size=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET" 2>/dev/null || echo 0)
    if [ "$size" -ge "$MIN_SIZE_BYTES" ]; then
        size_h=$(du -sh "$TARGET" | cut -f1)
        echo "Already present ($size_h). Skipping."
        exit 0
    fi
    echo "Partial file ($(du -sh "$TARGET" | cut -f1) < 500MB floor). Removing and retrying."
    rm -f "$TARGET"
fi

echo
echo "[1/2] Downloading..."
curl -L --fail --retry 5 --retry-delay 5 --continue-at - \
    -o "$TARGET" \
    "$DATASET_URL"

echo
echo "[2/2] Verifying..."
size=$(stat -c%s "$TARGET" 2>/dev/null || stat -f%z "$TARGET" 2>/dev/null || echo 0)
if [ "$size" -lt "$MIN_SIZE_BYTES" ]; then
    echo "  File too small ($(du -sh "$TARGET" | cut -f1)). Likely truncated."
    exit 1
fi

# JSON smoke test: first byte should be '['.
first_char=$(head -c 1 "$TARGET")
if [ "$first_char" != "[" ]; then
    echo "  File does not start with '[' — not a JSON array. Got: '$first_char'"
    exit 1
fi

size_h=$(du -sh "$TARGET" | cut -f1)
echo "  Size: $size_h"
echo
echo "Done."
