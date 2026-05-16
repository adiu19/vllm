#!/bin/bash
# Pre-flight config vet for the FlowPrefill benchmark.
#
# Read-only. Runs the full audit of:
#   - SMOKE flag + DEFAULT_MODE
#   - config.json values per policy bucket
#   - Predictor constants (slo_monitor.py)
#   - Proxy header forwarding
#   - preempt_check.py early-return for control
#   - SLO monitor log dedup
#   - Per-mode env resolution + VLLM_FLAGS
#   - Loadgen predictor import
#   - Sweep size / ETA
#
# Run this before launching run_sweep.sh on a new pod. Anything in red is
# a real problem to fix; everything else should match the expected values
# listed inline.
#
# Usage:
#   bash deploy/vet_configs.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── pretty helpers ───────────────────────────────────────────────────────
RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
BOLD=$'\033[1m'
RESET=$'\033[0m'

section() {
    echo
    echo "${BOLD}═══ $* ═══${RESET}"
}
ok() {
    echo "  ${GREEN}✓${RESET} $*"
}
warn() {
    echo "  ${YELLOW}⚠${RESET} $*"
}
fail() {
    echo "  ${RED}✗${RESET} $*"
}
note() {
    echo "    $*"
}

# ─────────────────────────────────────────────────────────────────────────
section "1. deploy/config.py — SMOKE + DEFAULT_MODE"

SMOKE_LINE=$(grep -E "^SMOKE\s*=" deploy/config.py | head -1)
echo "    $SMOKE_LINE"
if echo "$SMOKE_LINE" | grep -q "= False"; then
    ok "SMOKE=False (prod mode)"
else
    warn "SMOKE is NOT False — will load config.smoke.json instead of config.json"
fi
DM=$(grep -E "^DEFAULT_MODE\s*=" deploy/config.py | head -1)
echo "    $DM"

# ─────────────────────────────────────────────────────────────────────────
section "2. config.json — all values per policy bucket"

python3 << 'EOF'
import json
cfg = json.load(open('deploy/config.json'))
for k in ('control','conservative','aggressive'):
    b = cfg[k]
    print(f"  ─── {k} ───")
    print(f"    model              : {b['model']['name']}")
    print(f"    max_model_len      : {b['model']['max_model_len']}")
    print(f"    max_num_batched_t  : {b['model']['max_num_batched_tokens']}")
    print(f"    chunked_prefill    : {b['model']['enable_chunked_prefill']}")
    print(f"    async_scheduling   : {b['model']['async_scheduling']}")
    print(f"    prefix_caching     : {b['model']['enable_prefix_caching']}")
    print(f"    prefill            : TP={b['topology']['prefill_tp']} GPUs={b['topology']['prefill_gpus']} mem={b['topology']['prefill_gpu_mem_util']}")
    print(f"    decode             : TP={b['topology']['decode_tp']} GPUs={b['topology']['decode_gpus']} mem={b['topology']['decode_gpu_mem_util']}")
    print(f"    ports              : prefill={b['ports']['prefill']} decode={b['ports']['decode']} proxy={b['ports']['proxy_http']}")
    print(f"    env                : {dict(b['env'])}")
    print()
EOF

# ─────────────────────────────────────────────────────────────────────────
section "3. Predictor constants (slo_monitor.py) — calibration values"
grep -E "^TTFT_COEFF_" vllm/v1/core/sched/slo_monitor.py | while IFS= read -r line; do
    echo "    $line"
done

# ─────────────────────────────────────────────────────────────────────────
section "4. Proxy forwards X-FlowPrefill-SLO-MS to backends"
if grep -q "flowprefill_slo_ms" benchmarks/flowprefill/proxy.py; then
    ok "proxy.py has the flowprefill_slo_ms forwarding logic"
    grep -n "flowprefill_slo_ms\|X-FlowPrefill-SLO" benchmarks/flowprefill/proxy.py | head -8 | sed 's/^/    /'
else
    fail "proxy.py does NOT forward X-FlowPrefill-SLO-MS — sweep will not honor per-request SLOs"
fi

# ─────────────────────────────────────────────────────────────────────────
section "5. preempt_check.py early-return for control mode"
if grep -q "_preempt_target_step_id is None" vllm/v1/core/sched/preempt_check.py; then
    ok "preempt_check has the no-monitor fast path (zero overhead for control)"
    grep -B 1 -A 2 "_preempt_target_step_id is None" vllm/v1/core/sched/preempt_check.py | head -6 | sed 's/^/    /'
else
    fail "preempt_check missing the fast-path early return"
fi

# ─────────────────────────────────────────────────────────────────────────
section "6. SLO monitor log dedup (avoids 5ms-tick spam)"
if grep -q "_last_logged_step_id" vllm/v1/core/sched/slo_monitor.py; then
    ok "slo_monitor has dedup state for PREEMPT INTENT logging"
else
    warn "slo_monitor MAY emit duplicate PREEMPT INTENT lines every 5ms — log noise but not functional bug"
fi

# ─────────────────────────────────────────────────────────────────────────
section "7. Per-mode env resolution + assembled VLLM_FLAGS"
for m in control conservative aggressive; do
    echo "  ─── MODE=$m ───"
    MODE=$m bash -c '
        eval "$(python3 deploy/config.py 2>/dev/null)"
        [ "$MODE" != "control" ] && export FLOWPREFILL_ENABLED=1
        printf "    %-25s %s\n" MODE                  "$MODE"
        printf "    %-25s %s\n" FLOWPREFILL_ENABLED   "${FLOWPREFILL_ENABLED:-<unset>}"
        printf "    %-25s %s\n" FLOWPREFILL_POLICY    "${FLOWPREFILL_POLICY:-<unset>}"
        printf "    %-25s %s\n" NCCL_IB_DISABLE       "${NCCL_IB_DISABLE:-<unset>}"
        printf "    %-25s %s\n" NCCL_DEBUG            "${NCCL_DEBUG:-<unset>}"
        printf "    %-25s %s\n" VLLM_LOGGING_LEVEL    "${VLLM_LOGGING_LEVEL:-<unset>}"
        printf "    %-25s %s\n" PREFILL_TP            "$PREFILL_TP"
        printf "    %-25s %s\n" DECODE_TP             "$DECODE_TP"
        printf "    %-25s %s\n" PREFILL_GPUS          "$PREFILL_GPUS"
        printf "    %-25s %s\n" DECODE_GPUS           "$DECODE_GPUS"
        printf "    %-25s %s\n" VLLM_FLAGS            "$VLLM_FLAGS"
    '
    echo
done

# ─────────────────────────────────────────────────────────────────────────
section "8. Loadgen imports predictor from slo_monitor (single SoT)"
# Two simple checks: import line present, and the constant is referenced
# in the file at least twice (once in the import, once in usage).
has_import=$(grep -c "from vllm.v1.core.sched.slo_monitor" benchmarks/flowprefill/loadgen.py)
has_const=$(grep -c "TTFT_COEFF_A_MS_PER_TOKEN" benchmarks/flowprefill/loadgen.py)
if [ "$has_import" -ge 1 ] && [ "$has_const" -ge 2 ]; then
    ok "loadgen imports TTFT_COEFF_* from slo_monitor AND uses them ($has_const references)"
else
    fail "loadgen predictor wiring looks off (import=$has_import, const_refs=$has_const)"
fi

# ─────────────────────────────────────────────────────────────────────────
section "9. Sweep size + ETA"
RATES=$(grep -E "^RATES=" benchmarks/flowprefill/run_sweep.sh \
        | head -1 | sed 's/.*"\(.*\)".*/\1/')
POLICIES=$(grep -E "^POLICIES=" benchmarks/flowprefill/run_sweep.sh \
        | head -1 | sed 's/.*"\(.*\)".*/\1/')
TRIALS=$(grep -E "^TRIALS=" benchmarks/flowprefill/run_sweep.sh \
        | head -1 | sed 's/.*"\(.*\)".*/\1/')
WARMUP_S=$(grep -E "^WARMUP_S=" benchmarks/flowprefill/run_sweep.sh \
        | head -1 | sed 's/.*:-\([0-9]*\)}/\1/')
MEASURE_S=$(grep -E "^MEASURE_S=" benchmarks/flowprefill/run_sweep.sh \
        | head -1 | sed 's/.*:-\([0-9]*\)}/\1/')
BROKEN=$(grep -E "^BROKEN_GPUS=" benchmarks/flowprefill/run_sweep.sh \
        | head -1 | sed 's/.*"\(.*\)".*/\1/')
n_cells=$(( $(echo $RATES | wc -w) * $(echo $POLICIES | wc -w) * $(echo $TRIALS | wc -w) ))
sec_per_cell=$(( WARMUP_S + MEASURE_S ))
n_switches=$(( $(echo $RATES | wc -w) * $(echo $POLICIES | wc -w) ))
switch_s=$(( n_switches * 120 ))
total_s=$(( n_cells * sec_per_cell + switch_s ))
echo "    rates:           ${RATES}"
echo "    policies:        ${POLICIES}"
echo "    trials:          ${TRIALS}"
echo "    broken GPUs:     ${BROKEN:-<none>}"
echo "    cells:           $n_cells"
echo "    per-cell:        ${sec_per_cell}s"
echo "    switch overhead: ${switch_s}s (${n_switches} switches × 2 min)"
echo "    total estimate:  $(( total_s / 3600 ))h $(( (total_s % 3600) / 60 ))m"

# ─────────────────────────────────────────────────────────────────────────
section "10. GPU state right now"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv \
    | head -1 | sed 's/^/    /'
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | while IFS= read -r line; do
        idx=$(echo "$line" | awk -F', ' '{print $1}')
        mem=$(echo "$line" | awk -F', ' '{print $2}' | awk '{print $1}')
        if [ "$mem" -gt 500 ] 2>/dev/null; then
            echo "    ${RED}$line${RESET}"
        else
            echo "    $line"
        fi
    done
broken_in_sweep=$(grep -E "^BROKEN_GPUS=" benchmarks/flowprefill/run_sweep.sh \
        | head -1 | sed 's/.*"\(.*\)".*/\1/')
if [ -n "$broken_in_sweep" ]; then
    note "(sweep is configured to skip GPU(s) [$broken_in_sweep] in kill_gpu wait loop)"
fi

# ─────────────────────────────────────────────────────────────────────────
section "11. Disk + downloads"
df -h /workspace 2>&1 | head -2 | sed 's/^/    /'
model_dir="/workspace/hf_cache/hub/models--meta-llama--Llama-3.3-70B-Instruct"
if [ -d "$model_dir" ]; then
    ok "70B model present: $(du -sh $model_dir 2>/dev/null | cut -f1)"
else
    fail "70B model NOT downloaded — bash deploy/download_model.sh"
fi
sharegpt="/workspace/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
if [ -f "$sharegpt" ]; then
    ok "ShareGPT present: $(du -sh $sharegpt | cut -f1)"
else
    fail "ShareGPT NOT downloaded — bash deploy/download_sharegpt.sh"
fi

echo
echo "${BOLD}Done. Cross-check any ⚠ or ✗ above before launching the sweep.${RESET}"
