#!/bin/bash
# Kill all vllm and proxy processes and wait for GPU memory to free

pkill -9 -f vllm
pkill -9 -f "disagg_proxy"
pkill -9 -f "start_prefill"
pkill -9 -f "start_decode"

# Kill any remaining EngineCore processes holding GPU memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9

sleep 5
nvidia-smi
