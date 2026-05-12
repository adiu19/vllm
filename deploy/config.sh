#!/bin/bash
# Sourced by deploy/init.sh and every deploy/start_*.sh script.
#
# No secrets in here — HF_TOKEN comes from the pod env (RunPod dashboard).
# All `export`s below are values that determine WHAT we run (model, ports, GPUs)
# and HOW we run it (the per-MODE flag bundle).
#
# This file lives in the branch; each branch's config.sh tells us how that
# branch wants to be run. Currently treatment-branch settings.

# ─────────────────────────────────────────────────────────────────────────
# Required env vars — fail fast if not set (init.sh validates these too,
# but a manual `bash deploy/start_*.sh` should still complain clearly).
# ─────────────────────────────────────────────────────────────────────────
: "${HF_TOKEN:?ERROR: HF_TOKEN must be set as a pod env var}"
: "${MODE:?ERROR: MODE must be set: control or treatment}"

# ─────────────────────────────────────────────────────────────────────────
# Base settings (same regardless of MODE)
# ─────────────────────────────────────────────────────────────────────────
export NODE_IP=${NODE_IP:-127.0.0.1}
export MODEL=${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}

export PREFILL_PORT=${PREFILL_PORT:-8100}
export DECODE_PORT=${DECODE_PORT:-8200}
export STANDALONE_PORT=${STANDALONE_PORT:-8300}
export PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10001}
export PROXY_ZMQ_PORT=${PROXY_ZMQ_PORT:-30001}

export PREFILL_KV_PORT=${PREFILL_KV_PORT:-14579}
export DECODE_KV_PORT=${DECODE_KV_PORT:-14590}
export PREFILL_NIXL_PORT=${PREFILL_NIXL_PORT:-5600}
export DECODE_NIXL_PORT=${DECODE_NIXL_PORT:-5601}

export PREFILL_GPUS=${PREFILL_GPUS:-0,1}
export DECODE_GPUS=${DECODE_GPUS:-2,3}
export STANDALONE_GPUS=${STANDALONE_GPUS:-0,1}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
export STANDALONE_TP=${STANDALONE_TP:-$TENSOR_PARALLEL_SIZE}

# NCCL — single-node, no IB. Allow NVLink P2P (don't disable on SXM topology).
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
# NCCL_P2P_DISABLE: leave UNSET on SXM/NVLink topology. Only set to 1 on
# PCIe-only nodes where container P2P is broken (see deploy/README.md).

# Logging — INFO is the right default; bump to DEBUG only when actively
# diagnosing a specific issue.
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}

# Log files
export PREFILL_LOG=${PREFILL_LOG:-/tmp/prefill.log}
export DECODE_LOG=${DECODE_LOG:-/tmp/decode.log}
export PROXY_LOG=${PROXY_LOG:-/tmp/proxy.log}
export STANDALONE_LOG=${STANDALONE_LOG:-/tmp/standalone.log}

# ─────────────────────────────────────────────────────────────────────────
# MODE-specific flag bundle (passed to vllm.entrypoints.openai.api_server)
#
# control:   stock vLLM defaults (chunked prefill ON, async sched ON)
# treatment: flow-prefill gate requires chunked prefill OFF + async sched OFF;
#            max-model-len capped so max-num-batched-tokens >= max-model-len
#            is satisfied without an enormous batch budget.
# ─────────────────────────────────────────────────────────────────────────
case "$MODE" in
    control)
        export EXTRA_VLLM_FLAGS=""
        ;;
    treatment)
        export EXTRA_VLLM_FLAGS="--no-enable-chunked-prefill --no-async-scheduling --max-model-len 2048 --max-num-batched-tokens 2048"
        ;;
    *)
        echo "ERROR: Unknown MODE='$MODE' (must be 'control' or 'treatment')" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac
