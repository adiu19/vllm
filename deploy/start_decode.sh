#!/bin/bash
# Launch the decode-side vLLM service using P2pNcclConnector.
# Mode-specific flags come from $EXTRA_VLLM_FLAGS set by deploy/config.sh.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

export CUDA_VISIBLE_DEVICES="$DECODE_GPUS"
export VLLM_HOST_IP="$NODE_IP"

KV_CONFIG=$(cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"${DECODE_KV_PORT}","kv_connector_extra_config":{"proxy_ip":"${NODE_IP}","proxy_port":"${PROXY_ZMQ_PORT}","http_ip":"${NODE_IP}","http_port":"${DECODE_PORT}","send_type":"PUT_ASYNC"}}
EOF
)

# shellcheck disable=SC2086  # word splitting on $EXTRA_VLLM_FLAGS is intentional
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$DECODE_PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization 0.8 \
    $EXTRA_VLLM_FLAGS \
    --kv-transfer-config "$KV_CONFIG" > "$DECODE_LOG" 2>&1 &
echo $! > /tmp/decode.pid
echo "Decode started (PID $!, GPUs $DECODE_GPUS). Logs: tail -f $DECODE_LOG"
