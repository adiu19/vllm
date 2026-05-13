#!/bin/bash
# Launch the P2pNccl-path P/D request proxy.
#
# TODO(flowprefill): verify the proxy dispatches concurrently to the prefill
# backend. If it serializes (one request at a time to prefill), there will
# never be a waiting queue on the prefill node and PREEMPT INTENT can't fire.
# See deploy/start_proxy_nixl.sh for the same concern on the Nixl path.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

nohup python3 -u "$SCRIPT_DIR/proxy.py" > "$PROXY_LOG" 2>&1 &
echo $! > /tmp/proxy.pid
echo "Proxy started (PID $!). Logs: tail -f $PROXY_LOG"
