#!/bin/bash
# Launch the prefill-side vLLM service using P2pNcclConnector.
# Mode-specific flags come from $EXTRA_VLLM_FLAGS set by deploy/config.sh.
# This is the alternate transport to the NixlConnector path; pick one or the other.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

export CUDA_VISIBLE_DEVICES="$PREFILL_GPUS"
export VLLM_HOST_IP="$NODE_IP"

KV_CONFIG=$(cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"${PREFILL_KV_PORT}","kv_connector_extra_config":{"proxy_ip":"${NODE_IP}","proxy_port":"${PROXY_ZMQ_PORT}","http_ip":"${NODE_IP}","http_port":"${PREFILL_PORT}","send_type":"PUT_ASYNC"}}
EOF
)

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
echo "Prefill started (PID $!, GPUs $PREFILL_GPUS). Logs: tail -f $PREFILL_LOG"
