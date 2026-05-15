#!/bin/bash
# Launch a standalone vLLM service (no P/D disaggregation, no KV connector).
# Useful for sanity testing the code without the proxy/decode complexity.
# Standalone is debug-only — FlowPrefill not wired here (no SLO monitor).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py")" || { echo "config.py failed" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$STANDALONE_GPUS"
export VLLM_HOST_IP="$NODE_IP"

# Count GPUs from CUDA_VISIBLE_DEVICES for TP (allows STANDALONE_TP override)
IFS=',' read -ra _GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
TP="${STANDALONE_TP:-${#_GPU_ARR[@]}}"

# shellcheck disable=SC2086  # word splitting on $VLLM_FLAGS is intentional
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$STANDALONE_PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.7 \
    $VLLM_FLAGS > "$STANDALONE_LOG" 2>&1 &
echo $! > /tmp/standalone.pid
echo "Standalone vLLM started (PID $!, GPUs ${CUDA_VISIBLE_DEVICES}, TP=${TP}). Logs: tail -f $STANDALONE_LOG"
