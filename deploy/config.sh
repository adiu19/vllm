#!/bin/bash
# Branch config. Sourced by deploy/init.sh and every deploy/start_*.sh.
# Edit + push to change values. Sync this file to the control branch but
# drop EXTRA_VLLM_FLAGS (or set it to "") — control runs stock vLLM defaults.

: "${HF_TOKEN:?ERROR: HF_TOKEN must be set as a pod env var}"

export NODE_IP=127.0.0.1
export MODEL=meta-llama/Llama-3.3-70B-Instruct

export PREFILL_PORT=8100
export DECODE_PORT=8200
export STANDALONE_PORT=8300
export PROXY_HTTP_PORT=10001
export PROXY_ZMQ_PORT=30001

export PREFILL_KV_PORT=14579
export DECODE_KV_PORT=14590
export PREFILL_NIXL_PORT=5600
export DECODE_NIXL_PORT=5601

# 6-GPU setup: 4 prefill (TP=4) + 2 decode (TP=2), zero idle.
# For 8-GPU upgrade: change DECODE_GPUS to 4,5,6,7 and DECODE_TP to 4.
export PREFILL_GPUS=0,1,2,3
export DECODE_GPUS=4,5
export STANDALONE_GPUS=0,1,2,3
export PREFILL_TP=4
export DECODE_TP=2
export STANDALONE_TP=$PREFILL_TP

# gpu_memory_utilization differs by role for 70B on 80GB GPUs:
#   Prefill TP=4: 35GB weights/GPU → 0.85 (68GB cap) leaves 33GB free, comfortable.
#   Decode TP=2:  70GB weights/GPU → 0.95 (76GB cap) leaves 6GB free, tight but workable.
# At 0.8 on decode, weights wouldn't fit. At 0.97+ you risk OOM from any reserved overhead.
export PREFILL_GPU_MEM_UTIL=0.85
export DECODE_GPU_MEM_UTIL=0.95

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO

export PREFILL_LOG=/tmp/prefill.log
export DECODE_LOG=/tmp/decode.log
export PROXY_LOG=/tmp/proxy.log
export STANDALONE_LOG=/tmp/standalone.log

# max-model-len=8500 supports the 8000-token profiling bucket with margin.
# max-num-batched-tokens must be >= max-model-len when chunked prefill is off.
export EXTRA_VLLM_FLAGS="--no-enable-chunked-prefill --no-async-scheduling --no-enable-prefix-caching --max-model-len 8500 --max-num-batched-tokens 9000"
