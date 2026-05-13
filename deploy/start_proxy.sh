#!/bin/bash
set -e

source /tmp/config.sh

nohup python3 -u runpod/proxy.py > "$PROXY_LOG" 2>&1 &
echo $! > /tmp/proxy.pid
echo "Proxy started (PID $!). Logs: tail -f $PROXY_LOG"
