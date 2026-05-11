#!/bin/bash
set -e

source /tmp/config.sh

export CUDA_VISIBLE_DEVICES="${STANDALONE_GPUS:-0,1,2,3}"
export VLLM_HOST_IP="$NODE_IP"

TP="${STANDALONE_TP:-${TENSOR_PARALLEL_SIZE:-4}}"
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
