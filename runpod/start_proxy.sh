#!/bin/bash
nohup python3 runpod/proxy.py > /tmp/proxy.log 2>&1 &
echo $! > /tmp/proxy.pid
echo "Proxy started (PID $!). Logs: tail -f /tmp/proxy.log"
