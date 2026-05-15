#!/bin/bash
# Launch the FlowPrefill benchmark proxy (forked from
# tests/v1/kv_connector/nixl_integration/toy_proxy_server.py).
#
# Forks rationale: we keep our timing-header instrumentation and any
# future concurrency fixes in benchmarks/flowprefill/proxy.py so the
# upstream test fixture stays pristine. If we discover the upstream
# proxy serializes (TODO from a previous investigation: concurrent
# curls at :10001 produced waiting=0 throughout — though that was
# likely a max_num_batched_tokens=8192 batch-absorption effect, not
# proxy serialization), the fix lives in our fork.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py")" || { echo "config.py failed" >&2; exit 1; }

nohup python3 -u benchmarks/flowprefill/proxy.py \
    --port "$PROXY_HTTP_PORT" \
    --prefiller-hosts "$NODE_IP" \
    --prefiller-ports "$PREFILL_PORT" \
    --decoder-hosts "$NODE_IP" \
    --decoder-ports "$DECODE_PORT" > "$PROXY_LOG" 2>&1 &
echo $! > /tmp/proxy.pid
echo "FlowPrefill proxy started (PID $!). Logs: tail -f $PROXY_LOG"
