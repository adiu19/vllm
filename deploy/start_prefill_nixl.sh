#!/bin/bash
# Launch the prefill-side vLLM service using NixlConnector (kv_both).
# Mode-specific flags come from $EXTRA_VLLM_FLAGS set by deploy/config.sh.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

export CUDA_VISIBLE_DEVICES="$PREFILL_GPUS"
export VLLM_HOST_IP="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT"
export UCX_NET_DEVICES=all
# FlowPrefill: opt this node in as the prefill node for the SLO monitor.
# NixlConnector uses kv_role="kv_both" on BOTH prefill and decode, so
# kv_role alone can't tell them apart. This env var disambiguates.
# Do NOT set on decode start scripts.
export FLOWPREFILL_ENABLED=1

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

# shellcheck disable=SC2086  # word splitting on $EXTRA_VLLM_FLAGS is intentional
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PREFILL_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization 0.8 \
    $EXTRA_VLLM_FLAGS \
    --kv-transfer-config "$KV_CONFIG" > "$PREFILL_LOG" 2>&1 &
echo $! > /tmp/prefill.pid
echo "Prefill (nixl) started (PID $!, GPUs $PREFILL_GPUS, side-channel :${VLLM_NIXL_SIDE_CHANNEL_PORT}). Logs: tail -f $PREFILL_LOG"
