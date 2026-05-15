#!/bin/bash
# Launch the decode-side vLLM service using NixlConnector (kv_both).
# Decode is policy-agnostic: FLOWPREFILL_ENABLED is intentionally NOT set
# here regardless of MODE — the SLO monitor only runs on the prefill node.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py")" || { echo "config.py failed" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$DECODE_GPUS"
export VLLM_HOST_IP="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT"
export UCX_NET_DEVICES=all

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

# shellcheck disable=SC2086  # word splitting on $VLLM_FLAGS is intentional
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$DECODE_PORT" \
    --tensor-parallel-size "$DECODE_TP" \
    --gpu-memory-utilization "$DECODE_GPU_MEM_UTIL" \
    $VLLM_FLAGS \
    --kv-transfer-config "$KV_CONFIG" > "$DECODE_LOG" 2>&1 &
echo $! > /tmp/decode.pid
echo "Decode (nixl, MODE=$MODE) started (PID $!, GPUs $DECODE_GPUS, side-channel :${VLLM_NIXL_SIDE_CHANNEL_PORT}). Logs: tail -f $DECODE_LOG"
