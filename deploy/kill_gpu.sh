#!/bin/bash
# Kill all vLLM-related processes and wait for GPU memory to free.
#
# Catches three classes of process:
#   1. NVML-registered compute processes (visible in `nvidia-smi`).
#   2. Pattern-matched processes by name (api_server, WorkerProc, etc.).
#   3. Zombie processes holding /dev/nvidia* device handles that died mid-init
#      and never registered with NVML — only visible via `fuser` / `lsof`.
#      This is the case that bit us with the EngineCore that crashed during
#      model load: held all GPU device file descriptors but invisible to
#      `nvidia-smi --query-compute-apps`.

set -u

# ─────────────────────────────────────────────────────────────────────────
# 1. Pattern-based kills covering all FlowPrefill-related processes
# ─────────────────────────────────────────────────────────────────────────
pkill -9 -f vllm
pkill -9 -f api_server
pkill -9 -f WorkerProc
pkill -9 -f multiproc_executor
pkill -9 -f "start_prefill_nixl"
pkill -9 -f "start_decode_nixl"
pkill -9 -f "start_proxy_nixl"
pkill -9 -f "start_standalone"
pkill -9 -f "benchmarks/flowprefill/proxy.py"
pkill -9 -f "benchmarks/flowprefill/profile_ttft"
pkill -9 -f "benchmarks/flowprefill/loadgen"

# ─────────────────────────────────────────────────────────────────────────
# 2. NVML-registered compute processes (visible to nvidia-smi)
# ─────────────────────────────────────────────────────────────────────────
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | xargs -r kill -9

# ─────────────────────────────────────────────────────────────────────────
# 3. Zombie cleanup: anything still holding /dev/nvidia* handles.
#
# fuser shows ALL processes with the device open, including those that
# never finished CUDA init (and thus don't show up in nvidia-smi).
# This catches partially-started worker procs after a model-load failure.
# ─────────────────────────────────────────────────────────────────────────
echo "Checking for processes holding /dev/nvidia* device handles..."
NVIDIA_PIDS=$(
    fuser /dev/nvidia* 2>/dev/null \
        | tr -s ' \t' '\n' \
        | grep -oE '^[0-9]+' \
        | sort -u
)
if [ -n "${NVIDIA_PIDS:-}" ]; then
    echo "  Zombie PIDs found: $(echo "$NVIDIA_PIDS" | tr '\n' ' ')"
    for pid in $NVIDIA_PIDS; do
        # Skip PID 1 (init) defensively. Skip self.
        if [ "$pid" = "1" ] || [ "$pid" = "$$" ]; then
            continue
        fi
        # Skip kernel threads (no /proc/$pid/exe). Check via /proc/$pid/comm
        # so we can log what we're killing.
        if [ -r "/proc/$pid/comm" ]; then
            cmd=$(cat "/proc/$pid/comm" 2>/dev/null || echo '?')
            echo "  Killing PID $pid ($cmd)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
else
    echo "  None."
fi

# ─────────────────────────────────────────────────────────────────────────
# 3b. Last-resort: any Python process with CUDA libraries mmap'd in.
#
# Some processes have CUDA contexts open (and are holding GPU memory)
# but their /dev/nvidia* file handles have already been closed — fuser
# returns empty even though `nvidia-smi` still shows allocated memory.
# Catches this by scanning /proc/<pid>/maps for libcuda/libnvidia,
# which proves the process has CUDA runtime loaded and is the most
# likely owner of any orphaned GPU memory.
# ─────────────────────────────────────────────────────────────────────────
echo "Checking for Python processes with CUDA libraries loaded..."
CUDA_PYTHON_PIDS=$(
    pgrep -f python 2>/dev/null | while read -r pid; do
        if grep -qE 'libcuda|libnvidia' "/proc/$pid/maps" 2>/dev/null; then
            echo "$pid"
        fi
    done
)
if [ -n "${CUDA_PYTHON_PIDS:-}" ]; then
    echo "  Python PIDs with CUDA loaded: $(echo "$CUDA_PYTHON_PIDS" | tr '\n' ' ')"
    for pid in $CUDA_PYTHON_PIDS; do
        if [ "$pid" = "1" ] || [ "$pid" = "$$" ]; then continue; fi
        if [ -r "/proc/$pid/comm" ]; then
            cmd=$(cat "/proc/$pid/comm" 2>/dev/null || echo '?')
            echo "  Killing PID $pid ($cmd)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
else
    echo "  None."
fi

# ─────────────────────────────────────────────────────────────────────────
# 4. Release ports (KV-transfer, HTTP, NIXL side-channels, etc.)
# ─────────────────────────────────────────────────────────────────────────
for port in 14579 14580 14581 14590 14591 8100 8200 8300 10001 30001 5600 5601; do
    fuser -k ${port}/tcp 2>/dev/null && echo "Freed port ${port}"
done

# ─────────────────────────────────────────────────────────────────────────
# 5. Wait for GPU memory to free across ALL visible GPUs (not just GPU 0)
# ─────────────────────────────────────────────────────────────────────────
echo "Waiting for GPU memory to free across all visible GPUs..."
for i in $(seq 1 20); do
    sleep 2
    MAX_MEM=$(
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
            | sort -n \
            | tail -1
    )
    if [ "${MAX_MEM:-99999}" -lt 500 ]; then
        echo "All GPUs clear."
        break
    fi
    echo "  Iteration $i/20: max GPU memory still used = ${MAX_MEM} MiB"
done

# Final state for the user to inspect.
echo
nvidia-smi --query-gpu=index,name,memory.used --format=csv
