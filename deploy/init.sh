#!/bin/bash
# Pre-flight checks and pod initialization. Called automatically at container
# startup by entrypoint.sh (unless SKIP_INIT=1). Also safe to run manually any
# time the pod state needs verification.
#
# Fails hard with a clear error on any check failure — better to fail fast than
# spend 10 minutes on a download that's going to 401, or sync up TP workers on
# a dirty GPU.
#
# What it does:
#   1. Validates required env vars (HF_TOKEN, MODE)
#   2. Git-syncs the fork to the branch matching MODE
#   3. Sources deploy/config.sh from the freshly-pulled branch
#   4. Verifies HuggingFace auth (token validity only — model-specific access
#      is checked implicitly when vLLM downloads it)
#   5. Checks GPU topology (NVLink presence) and memory state (no orphans)
#   6. Prints a summary

set -euo pipefail

FORK_DIR=/opt/vllm-fork
LOG=/tmp/init.log

# Tee output so it's both visible in container logs AND saved for SSH debugging
exec > >(tee -a "$LOG") 2>&1

echo "════════════════════════════════════════════════════════════════════"
echo "deploy/init.sh @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "════════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────
# 1. Required env vars
# ─────────────────────────────────────────────────────────────────────────
: "${HF_TOKEN:?ERROR: HF_TOKEN must be set as a pod env var}"
: "${MODE:?ERROR: MODE must be set (see deploy/experiments.sh for valid modes)}"

echo "[1/6] env: HF_TOKEN ✓ | MODE=$MODE ✓"

# ─────────────────────────────────────────────────────────────────────────
# 2. Resolve MODE → BRANCH via experiments.sh, then sync the fork
# ─────────────────────────────────────────────────────────────────────────
cd "$FORK_DIR"

# Source experiments.sh from whatever's currently on disk (last checked-out
# branch). This gives us $BRANCH for the target mode. If the current on-disk
# experiments.sh lacks the requested MODE, it fails clearly — fix by running
# `git fetch && git checkout <some-branch-with-the-mode>` manually once, or
# rebuilding the image with a default branch that defines all modes.
# shellcheck source=/dev/null
source "$FORK_DIR/deploy/experiments.sh"

echo "[2/6] git: syncing $FORK_DIR (MODE=$MODE → branch=$BRANCH)"

git fetch origin --quiet
git checkout "$BRANCH" --quiet
git pull origin "$BRANCH" --quiet

CURRENT_SHA=$(git rev-parse --short HEAD)
echo "       on branch=$BRANCH at $CURRENT_SHA"

# ─────────────────────────────────────────────────────────────────────────
# 3. Source config.sh from the freshly pulled branch. Each branch keeps its
#    own config.sh with the values (including EXTRA_VLLM_FLAGS) it wants.
# ─────────────────────────────────────────────────────────────────────────
echo "[3/6] config: sourcing deploy/config.sh"
# shellcheck source=/dev/null
source "$FORK_DIR/deploy/config.sh"
echo "       MODEL=${MODEL:-unset} | EXTRA_VLLM_FLAGS=${EXTRA_VLLM_FLAGS:-<empty>}"

# ─────────────────────────────────────────────────────────────────────────
# 4. HuggingFace auth + model access
# ─────────────────────────────────────────────────────────────────────────
echo "[4/6] hf: verifying auth"
WHO=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer ${HF_TOKEN}" \
    https://huggingface.co/api/whoami-v2)
HTTP_CODE=$(echo "$WHO" | tail -n1)
BODY=$(echo "$WHO" | head -n -1)
if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: HF auth failed (HTTP $HTTP_CODE): $BODY" >&2
    exit 1
fi
USER=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name','unknown'))")
echo "       authenticated as $USER"

# ─────────────────────────────────────────────────────────────────────────
# 5. GPU topology + health
# ─────────────────────────────────────────────────────────────────────────
echo "[5/6] gpu: topology + memory check"

if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not available — no GPU driver?" >&2
    exit 1
fi

# Topology: warn loudly if no NVLink (NV# entries) found between GPUs.
# TP collectives will be 10x slower on PCIe-only setups.
TOPO_OUT=$(nvidia-smi topo -m 2>&1)
if echo "$TOPO_OUT" | grep -qE "NV[0-9]+"; then
    NVLINK_LINE=$(echo "$TOPO_OUT" | grep -oE "NV[0-9]+" | sort -u | tr '\n' ' ')
    echo "       NVLink present: $NVLINK_LINE"
else
    echo "ERROR: no NVLink detected between GPUs (PCIe-only topology)." >&2
    echo "TP collectives will be slow and may not initialize cleanly." >&2
    echo "topo:" >&2
    echo "$TOPO_OUT" >&2
    exit 1
fi

# Memory orphans: any GPU with >500 MiB used at startup likely has leaked
# memory from a previous crashed run. Fail hard — fix before continuing.
DIRTY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F', ' '$2 > 500 { printf "GPU %s: %s MiB used\n", $1, $2 }')
if [ -n "$DIRTY" ]; then
    echo "ERROR: GPU(s) with orphan memory (>500 MiB used at startup):" >&2
    echo "$DIRTY" >&2
    echo "Run \`bash deploy/kill_gpu.sh\` to clean, or get a fresh pod." >&2
    exit 1
fi
echo "       all GPUs clean (<500 MiB used)"

# ─────────────────────────────────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────────────────────────────────
echo "[6/6] ready"
echo "════════════════════════════════════════════════════════════════════"
echo "Pod ready. Launch services:"
echo "  bash deploy/start_prefill_nixl.sh"
echo "  bash deploy/start_decode_nixl.sh"
echo "  bash deploy/start_standalone.sh"
echo "════════════════════════════════════════════════════════════════════"
