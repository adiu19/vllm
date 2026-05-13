#!/bin/bash
# Launch the Nixl-path P/D request proxy.
#
# TODO(flowprefill): the toy_proxy_server appears to serialize requests to the
# prefill backend — concurrent curls at :10001 don't produce backlog on the
# prefill node (waiting=0 always). For FlowPrefill to demonstrate preempt
# behavior we need the proxy to fire requests to prefill concurrently so a
# waiting queue actually forms. Either:
#   - patch toy_proxy_server.py for concurrent prefill dispatch, or
#   - swap it for a production-grade async proxy
# Until then, validate FlowPrefill by hitting :8100 (prefill direct).
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
