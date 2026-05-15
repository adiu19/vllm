#!/bin/bash
# Launch the prefill-side vLLM service using NixlConnector (kv_both).
# Sources deploy/config.sh which picks a policy bucket from config.json
# (default conservative; override with MODE=control|aggressive in env).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py")" || { echo "config.py failed" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$PREFILL_GPUS"
export VLLM_HOST_IP="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT"
export UCX_NET_DEVICES=all

# FlowPrefill: opt this node in as the prefill node for the SLO monitor.
# NixlConnector uses kv_role="kv_both" on BOTH prefill and decode, so kv_role
# alone can't tell them apart. This is a node-role signal (not policy), so
# it lives in the start script — derived from $MODE, never from config.json.
if [ "$MODE" != "control" ]; then
    export FLOWPREFILL_ENABLED=1
fi

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

# shellcheck disable=SC2086  # word splitting on $VLLM_FLAGS is intentional
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PREFILL_PORT" \
    --tensor-parallel-size "$PREFILL_TP" \
    --gpu-memory-utilization "$PREFILL_GPU_MEM_UTIL" \
    $VLLM_FLAGS \
    --kv-transfer-config "$KV_CONFIG" > "$PREFILL_LOG" 2>&1 &
echo $! > /tmp/prefill.pid
echo "Prefill (nixl, MODE=$MODE) started (PID $!, GPUs $PREFILL_GPUS, side-channel :${VLLM_NIXL_SIDE_CHANNEL_PORT}). Logs: tail -f $PREFILL_LOG"
