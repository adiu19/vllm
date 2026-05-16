#!/bin/bash
# FlowPrefill benchmark sweep orchestrator.
#
# Drives the full benchmark matrix:
#   - 5 arrival rates (under-loaded → near-saturation)
#   - 3 policies (control, conservative, aggressive)
#   - 5 trials per (rate, policy) cell
#
# For each (rate, policy):
#   1. Tear down + restart the prefill/decode/proxy stack with MODE=policy.
#      (Same MODE drives loadgen's CSV labeling — single source of truth.)
#   2. Run loadgen 5 times back-to-back with trial_id ∈ {0..4}.
#      Same workload (paired by master_seed); only the policy varies.
#
# Time budget per (rate, policy): 5 × (warmup + measure) ≈ 28 min.
# Total: 5 rates × 3 policies × 28 min + restart overhead ≈ 7 hours.
#
# Output: $RUN_DIR/rate_<R>/trial_<N>_policy_<POLICY>.{csv,meta.json}
# Plus  : $RUN_DIR/rate_<R>/{prefill,decode,proxy}.log per policy bucket
#         (renamed at policy boundaries so we can post-process per slot).
#
# Usage:
#   bash benchmarks/flowprefill/run_sweep.sh [RUN_NAME]
#
# Configure RATES below to match the operating points you want.

set -e
set -u

# ─── Knobs ────────────────────────────────────────────────────────────────
RATES=(${RATES:-"2 4 6 8 10"})         # req/s sweep points
TRIALS=(${TRIALS:-"0 1 2 3 4"})        # paired-trial IDs (same seed family)
POLICIES=(${POLICIES:-"control conservative aggressive"})
MASTER_SEED=${MASTER_SEED:-42}
WARMUP_S=${WARMUP_S:-30}
MEASURE_S=${MEASURE_S:-300}
TIER_SPLIT=${TIER_SPLIT:-0.2}
MAX_TOKENS=${MAX_TOKENS:-32}
# ──────────────────────────────────────────────────────────────────────────

RUN_NAME="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="/workspace/benchmark_runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Banner so the run dir is unambiguous in logs.
echo "════════════════════════════════════════════════════════════════════"
echo "FlowPrefill sweep:  $RUN_NAME"
echo "  rates    : ${RATES[*]}"
echo "  policies : ${POLICIES[*]}"
echo "  trials   : ${TRIALS[*]}"
echo "  warmup   : ${WARMUP_S}s   measure: ${MEASURE_S}s"
echo "  seed     : $MASTER_SEED   tier_split: $TIER_SPLIT"
echo "  output   : $RUN_DIR"
echo "════════════════════════════════════════════════════════════════════"

wait_for_uvicorn() {
    # $1 = log path, $2 = port. Polls log until uvicorn's ready message
    # appears, then a /v1/models check. Cap at 8 min per service.
    local log_path=$1
    local port=$2
    local svc=$3
    local deadline=$(( $(date +%s) + 480 ))
    echo "    waiting for $svc on :$port ..."
    while [ $(date +%s) -lt $deadline ]; do
        if grep -q "Uvicorn running on http://0.0.0.0:$port" "$log_path" 2>/dev/null; then
            sleep 1
            if curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1; then
                echo "    $svc ready."
                return 0
            fi
        fi
        sleep 3
    done
    echo "ERROR: $svc did not come up in 8 min. See $log_path"
    return 1
}

bring_up_stack() {
    # $1 = MODE for this run
    local mode=$1
    echo "  [stack] killing previous instances..."
    bash deploy/kill_gpu.sh || true
    sleep 3

    echo "  [stack] starting prefill (MODE=$mode)..."
    MODE=$mode bash deploy/start_prefill_nixl.sh
    wait_for_uvicorn /tmp/prefill.log 8100 prefill

    echo "  [stack] starting decode (MODE=$mode)..."
    MODE=$mode bash deploy/start_decode_nixl.sh
    wait_for_uvicorn /tmp/decode.log 8200 decode

    echo "  [stack] starting proxy..."
    bash deploy/start_proxy_nixl.sh
    sleep 2
    # Proxy doesn't expose /v1/models — single completion smoke check.
    if ! curl -sf -o /dev/null -X POST "http://localhost:10001/v1/completions" \
         -H "Content-Type: application/json" \
         -H "X-FlowPrefill-SLO-MS: 10000" \
         -d '{"model":"'"$MODEL"'","prompt":"ok","max_tokens":1,"stream":false}'; then
        echo "ERROR: proxy didn't respond to a smoke completion."
        return 1
    fi
    echo "  [stack] up."
}

archive_logs_for_policy() {
    # Snapshot the prefill/decode/proxy logs at the end of a (rate, policy)
    # block. analyze.py's log-parser walks these for PREEMPT INTENT events.
    local rate=$1
    local policy=$2
    local rate_dir="$RUN_DIR/rate_${rate}"
    mkdir -p "$rate_dir"
    cp /tmp/prefill.log "$rate_dir/prefill_${policy}.log" 2>/dev/null || true
    cp /tmp/decode.log  "$rate_dir/decode_${policy}.log"  2>/dev/null || true
    cp /tmp/proxy.log   "$rate_dir/proxy_${policy}.log"   2>/dev/null || true
}

# Pull MODEL once for the smoke completion in bring_up_stack.
eval "$(python3 deploy/config.py 2>/dev/null)"

for rate in "${RATES[@]}"; do
    rate_dir="$RUN_DIR/rate_${rate}"
    mkdir -p "$rate_dir"
    echo
    echo "──── rate = $rate req/s ─────────────────────────────────────────"

    for policy in "${POLICIES[@]}"; do
        echo
        echo "  policy = $policy"
        bring_up_stack "$policy"

        for trial in "${TRIALS[@]}"; do
            echo "    trial $trial ..."
            MODE=$policy python3 benchmarks/flowprefill/loadgen.py \
                --rate "$rate" \
                --warmup-s "$WARMUP_S" \
                --measure-s "$MEASURE_S" \
                --tier-split "$TIER_SPLIT" \
                --master-seed "$MASTER_SEED" \
                --trial-id "$trial" \
                --max-tokens "$MAX_TOKENS" \
                --output-dir "$rate_dir"
        done

        archive_logs_for_policy "$rate" "$policy"
    done
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "Sweep complete: $RUN_DIR"
echo "Next: python3 benchmarks/flowprefill/analyze.py $RUN_DIR/rate_${RATES[2]}"
echo "  (use the contention-band rate's directory; analyze each rate dir"
echo "   separately, or sweep-aggregate manually)"
echo "════════════════════════════════════════════════════════════════════"
