#!/bin/bash
# Launch the decode-side vLLM service using NixlConnector (kv_both).
# Mode-specific flags come from $EXTRA_VLLM_FLAGS set by deploy/config.sh.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

export CUDA_VISIBLE_DEVICES="$DECODE_GPUS"
export VLLM_HOST_IP="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT"
export UCX_NET_DEVICES=all

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

# shellcheck disable=SC2086  # word splitting on $EXTRA_VLLM_FLAGS is intentional
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$DECODE_PORT" \
    --tensor-parallel-size "$DECODE_TP" \
    --gpu-memory-utilization "$DECODE_GPU_MEM_UTIL" \
    $EXTRA_VLLM_FLAGS \
    --kv-transfer-config "$KV_CONFIG" > "$DECODE_LOG" 2>&1 &
echo $! > /tmp/decode.pid
echo "Decode (nixl) started (PID $!, GPUs $DECODE_GPUS, side-channel :${VLLM_NIXL_SIDE_CHANNEL_PORT}). Logs: tail -f $DECODE_LOG"
