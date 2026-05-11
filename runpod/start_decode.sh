#!/bin/bash
set -e

source /tmp/config.sh

export VLLM_HOST_IP="$DECODE_IP"

KV_CONFIG=$(cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":"1e10","kv_port":"${DECODE_KV_PORT}","kv_connector_extra_config":{"proxy_ip":"${PREFILL_IP}","proxy_port":"${PROXY_ZMQ_PORT}","http_ip":"${DECODE_IP}","http_port":"${DECODE_PORT}","send_type":"PUT_ASYNC"}}
EOF
)

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$DECODE_PORT" \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config "$KV_CONFIG" > "$DECODE_LOG" 2>&1 &
echo $! > /tmp/decode.pid
echo "Decode started (PID $!). Logs: tail -f $DECODE_LOG"
