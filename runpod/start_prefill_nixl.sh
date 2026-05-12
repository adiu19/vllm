#!/bin/bash
set -e

source /tmp/config.sh

export CUDA_VISIBLE_DEVICES="$PREFILL_GPUS"
export VLLM_HOST_IP="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_HOST="$NODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="${PREFILL_NIXL_PORT:-5600}"
export UCX_NET_DEVICES=all

KV_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PREFILL_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization 0.8 \
    --no-enable-chunked-prefill \
    --no-async-scheduling \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --kv-transfer-config "$KV_CONFIG" > "$PREFILL_LOG" 2>&1 &
echo $! > /tmp/prefill.pid
echo "Prefill (nixl) started (PID $!, GPUs $PREFILL_GPUS, side-channel :${VLLM_NIXL_SIDE_CHANNEL_PORT}). Logs: tail -f $PREFILL_LOG"
