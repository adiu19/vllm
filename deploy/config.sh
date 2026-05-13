#!/bin/bash
# Branch config. Sourced by deploy/init.sh and every deploy/start_*.sh.
# Edit + push to change values. Sync this file to the control branch but
# drop EXTRA_VLLM_FLAGS (or set it to "") — control runs stock vLLM defaults.

: "${HF_TOKEN:?ERROR: HF_TOKEN must be set as a pod env var}"

export NODE_IP=127.0.0.1
export MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct

export PREFILL_PORT=8100
export DECODE_PORT=8200
export STANDALONE_PORT=8300
export PROXY_HTTP_PORT=10001
export PROXY_ZMQ_PORT=30001

export PREFILL_KV_PORT=14579
export DECODE_KV_PORT=14590
export PREFILL_NIXL_PORT=5600
export DECODE_NIXL_PORT=5601

export PREFILL_GPUS=0,1
export DECODE_GPUS=2,3
export STANDALONE_GPUS=0,1
export TENSOR_PARALLEL_SIZE=2
export STANDALONE_TP=$TENSOR_PARALLEL_SIZE

export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export VLLM_LOGGING_LEVEL=INFO

export PREFILL_LOG=/tmp/prefill.log
export DECODE_LOG=/tmp/decode.log
export PROXY_LOG=/tmp/proxy.log
export STANDALONE_LOG=/tmp/standalone.log

export EXTRA_VLLM_FLAGS="--no-enable-chunked-prefill --no-async-scheduling --max-model-len 2048 --max-num-batched-tokens 2048"
