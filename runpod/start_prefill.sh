#!/bin/bash
set -e

: "${PREFILL_IP:?Set PREFILL_IP to this node's IP}"
: "${DECODE_IP:?Set DECODE_IP to the decode node's IP}"
: "${MODEL:?Set MODEL to the HuggingFace model ID}"

KV_CONFIG=$(cat <<EOF
{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"14579","kv_connector_extra_config":{"proxy_ip":"${PREFILL_IP}","proxy_port":"30001","http_ip":"${PREFILL_IP}","http_port":"8100","send_type":"PUT_ASYNC"}}
EOF
)

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8100 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config "$KV_CONFIG"
