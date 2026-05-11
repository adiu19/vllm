#!/bin/bash
set -e

source /tmp/config.sh

export CUDA_VISIBLE_DEVICES="${STANDALONE_GPUS:-0,1,2,3}"
export VLLM_HOST_IP="$NODE_IP"

# Count GPUs from CUDA_VISIBLE_DEVICES instead of using P/D TENSOR_PARALLEL_SIZE
IFS=',' read -ra _GPU_ARR <<< "$CUDA_VISIBLE_DEVICES"
TP="${STANDALONE_TP:-${#_GPU_ARR[@]}}"
PORT="${STANDALONE_PORT:-8300}"
LOG="${STANDALONE_LOG:-/tmp/standalone.log}"

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.8 > "$LOG" 2>&1 &
echo $! > /tmp/standalone.pid
echo "Standalone vLLM started (PID $!, GPUs ${CUDA_VISIBLE_DEVICES}, TP=${TP}). Logs: tail -f $LOG"
