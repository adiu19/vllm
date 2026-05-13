#!/bin/bash
# Usage: check_network.sh <remote_ip> [port1 port2 ...]
# Checks RunPod Global Networking connectivity to a remote pod.

REMOTE_IP="${1:?Usage: check_network.sh <remote_ip> [port1 port2 ...]}"
shift
if [ "$#" -eq 0 ]; then
    PORTS=(8100 8200 10001 30001 22)
else
    PORTS=("$@")
fi

echo "=== Local network interfaces ==="
ip addr show | grep "inet " | awk '{print $2, $NF}'

echo ""
MY_10=$(ip addr show | grep "inet 10\." | awk '{print $2}' | head -1)
if [ -z "$MY_10" ]; then
    echo "❌ No 10.x.x.x address found — Global Networking may not be enabled on this pod"
else
    echo "✅ Global Networking IP: $MY_10"
fi

echo ""
echo "=== Ping $REMOTE_IP (3 packets, 3s timeout) ==="
if ping -c 3 -W 3 "$REMOTE_IP" > /tmp/ping_out 2>&1; then
    echo "✅ Ping OK"
    grep "rtt" /tmp/ping_out || true
else
    echo "❌ Ping failed"
    cat /tmp/ping_out
fi

echo ""
echo "=== TCP port checks ==="
for port in "${PORTS[@]}"; do
    result=$(python3 -c "
import socket, sys
s = socket.socket()
s.settimeout(3)
try:
    s.connect(('$REMOTE_IP', $port))
    print('open')
except Exception as e:
    print(f'FAIL: {e}')
finally:
    s.close()
" 2>&1)
    if echo "$result" | grep -q "^open"; then
        echo "  ✅ $REMOTE_IP:$port — open"
    else
        echo "  ❌ $REMOTE_IP:$port — $result"
    fi
done
