#!/bin/bash
set -e

source /tmp/config.sh

PROXY_PORT="${PROXY_HTTP_PORT:-10001}"
LOG="${PROXY_LOG:-/tmp/proxy.log}"

nohup python3 -u tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port "$PROXY_PORT" \
    --prefiller-hosts "$NODE_IP" \
    --prefiller-ports "$PREFILL_PORT" \
    --decoder-hosts "$NODE_IP" \
    --decoder-ports "$DECODE_PORT" > "$LOG" 2>&1 &
echo $! > /tmp/proxy.pid
echo "Nixl proxy started (PID $!). Logs: tail -f $LOG"
