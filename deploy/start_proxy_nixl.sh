#!/bin/bash
# Launch the Nixl-path P/D request proxy.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

nohup python3 -u tests/v1/kv_connector/nixl_integration/toy_proxy_server.py \
    --port "$PROXY_HTTP_PORT" \
    --prefiller-hosts "$NODE_IP" \
    --prefiller-ports "$PREFILL_PORT" \
    --decoder-hosts "$NODE_IP" \
    --decoder-ports "$DECODE_PORT" > "$PROXY_LOG" 2>&1 &
echo $! > /tmp/proxy.pid
echo "Nixl proxy started (PID $!). Logs: tail -f $PROXY_LOG"
