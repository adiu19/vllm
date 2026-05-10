#!/bin/bash
set -e

: "${PREFILL_IP:?Set PREFILL_IP to the prefill node's IP}"
: "${DECODE_IP:?Set DECODE_IP to the decode node's IP}"

python3 -m pip install -q quart

python3 benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
    --prefill-url "http://${PREFILL_IP}:8100" \
    --decode-url "http://${DECODE_IP}:8200" \
    --port 8000
