#!/bin/bash
# Kill all vllm and proxy processes and wait for GPU memory to free

pkill -9 -f vllm
pkill -9 -f "disagg_proxy"
pkill -9 -f "start_prefill"
pkill -9 -f "start_decode"

# Kill any remaining processes holding GPU memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9

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
