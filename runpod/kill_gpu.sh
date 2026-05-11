#!/bin/bash
# Kill all vllm and proxy processes and wait for GPU memory to free

pkill -9 -f vllm
pkill -9 -f "disagg_proxy"
pkill -9 -f "start_prefill"
pkill -9 -f "start_decode"
pkill -9 -f "start_standalone"
pkill -9 -f "proxy.py"

# Kill any remaining processes holding GPU memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9

# Release KV transfer ports (may survive if process died before GPU alloc)
for port in 14579 14580 14581 14590 14591 8100 8200 10001 30001; do
    fuser -k ${port}/tcp 2>/dev/null && echo "Freed port ${port}"
done

# Wait until GPU memory is actually freed
echo "Waiting for GPU memory to free..."
for i in $(seq 1 20); do
    sleep 2
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    echo "  GPU 0 memory used: ${mem} MiB"
    if [ "$mem" -lt 500 ]; then
        echo "GPU clear."
        break
    fi
done
nvidia-smi
