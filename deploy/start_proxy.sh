#!/bin/bash
# Launch the P2pNccl-path P/D request proxy.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/config.sh"

nohup python3 -u "$SCRIPT_DIR/proxy.py" > "$PROXY_LOG" 2>&1 &
echo $! > /tmp/proxy.pid
echo "Proxy started (PID $!). Logs: tail -f $PROXY_LOG"
