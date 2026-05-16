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
# `read -ra` does the word splitting explicitly — the older
# `POLICIES=(${POLICIES:-"a b c"})` pattern depends on bash word-splitting
# inside array assignment and can collapse to a 1-element array if the
# default-value quotes survive expansion. read -ra is unambiguous.
read -ra RATES    <<< "${RATES:-2 4 6 8 10}"
read -ra TRIALS   <<< "${TRIALS:-0 1 2 3 4}"
read -ra POLICIES <<< "${POLICIES:-control conservative aggressive}"
MASTER_SEED=${MASTER_SEED:-42}
WARMUP_S=${WARMUP_S:-30}
MEASURE_S=${MEASURE_S:-300}
TIER_SPLIT=${TIER_SPLIT:-0.2}
MAX_TOKENS=${MAX_TOKENS:-4}
BROKEN_GPUS="4"   # comma-separated indices to skip in kill_gpu wait loop
# ──────────────────────────────────────────────────────────────────────────

RUN_NAME="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="/workspace/benchmark_runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# tee everything to a sweep log under RUN_DIR so it's reviewable after
# the SSH session ends or the script finishes.
SWEEP_LOG="$RUN_DIR/sweep.log"
exec > >(tee -a "$SWEEP_LOG") 2>&1

# Total cell count + progress tracker. Counter file lets us resume-aware
# this later (skip cells whose output CSV already exists) — for now,
# just used for the progress display.
TOTAL_CELLS=$(( ${#RATES[@]} * ${#POLICIES[@]} * ${#TRIALS[@]} ))
CELLS_DONE=0
SWEEP_START_S=$(date +%s)

# Timestamped log helper. Use `log "..."` instead of bare echo for
# anything we want to time-stamp in the sweep log.
log() {
    printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

human_dur() {
    # $1 = seconds → "1h23m45s"
    local s=$1
    printf "%dh%02dm%02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

# Banner so the run dir is unambiguous in logs.
log "════════════════════════════════════════════════════════════════════"
log "FlowPrefill sweep:  $RUN_NAME"
log "  rates       : ${RATES[*]}"
log "  policies    : ${POLICIES[*]}"
log "  trials      : ${TRIALS[*]}"
log "  warmup      : ${WARMUP_S}s   measure: ${MEASURE_S}s"
log "  seed        : $MASTER_SEED   tier_split: $TIER_SPLIT"
log "  max_tokens  : $MAX_TOKENS"
log "  broken_gpus : ${BROKEN_GPUS:-<none>}    # GPUs excluded from kill_gpu wait loop"
log "  output      : $RUN_DIR"
log "  sweep log   : $SWEEP_LOG"
log "  cells       : $TOTAL_CELLS total"
log "════════════════════════════════════════════════════════════════════"

# Snapshot the GPU state at sweep start. Lets us cross-reference any
# "broken_gpus" entries above with what the hardware actually shows.
log "GPU state at sweep start:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | while IFS= read -r line; do log "  $line"; done
log "════════════════════════════════════════════════════════════════════"

wait_for_uvicorn() {
    # $1 = log path, $2 = port. Two-stage check:
    #   (a) log says "Application startup complete" (FastAPI ready)
    #   (b) /v1/models responds 200 (HTTP layer actively serving)
    # Both must pass — guards against curl racing the bind AND guards
    # against false positives if stale log content lingers (we truncate
    # logs at bring_up_stack entry to make (a) reliable).
    # Cap at 15 min per service (70B + CUDA graph capture + NIXL).
    local log_path=$1
    local port=$2
    local svc=$3
    local deadline=$(( $(date +%s) + 900 ))
    echo "    waiting for $svc on :$port ..."
    while [ $(date +%s) -lt $deadline ]; do
        if grep -q "Application startup complete" "$log_path" 2>/dev/null \
           && curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "    $svc ready."
            return 0
        fi
        sleep 3
    done
    echo "ERROR: $svc did not come up in 15 min. See $log_path"
    return 1
}

bring_up_stack() {
    # $1 = MODE for this run.
    #
    # Prefill and decode launch in parallel — both are reading the same
    # already-downloaded weights from disk, no I/O competition between
    # them since each loads its own TP-sharded slices into separate
    # GPUs. Halves restart time per policy switch (~60s → ~30s on warm
    # disk cache).
    local mode=$1
    echo "  [stack] killing previous instances..."
    # BROKEN_GPUS lets kill_gpu skip the wait-loop on GPUs whose memory
    # we can't clear (e.g. host-side stuck allocation). Set via env so
    # the same script works on healthy hosts too.
    BROKEN_GPUS="${BROKEN_GPUS:-}" bash deploy/kill_gpu.sh || true
    sleep 3

    # Truncate the per-service logs BEFORE launching, so wait_for_uvicorn's
    # grep for "Application startup complete" can't match stale content
    # from a previous policy's run or any manual debug append.
    : > /tmp/prefill.log
    : > /tmp/decode.log
    : > /tmp/proxy.log

    echo "  [stack] starting prefill + decode in parallel (MODE=$mode)..."
    MODE=$mode bash deploy/start_prefill_nixl.sh
    MODE=$mode bash deploy/start_decode_nixl.sh

    # Both vllm processes are loading concurrently in the background
    # now. Wait for each to expose its API in parallel.
    wait_for_uvicorn /tmp/prefill.log 8100 prefill &
    local pwait=$!
    wait_for_uvicorn /tmp/decode.log  8200 decode  &
    local dwait=$!
    wait $pwait || { echo "ERROR: prefill failed to come up"; return 1; }
    wait $dwait || { echo "ERROR: decode failed to come up";  return 1; }

    echo "  [stack] starting proxy..."
    bash deploy/start_proxy_nixl.sh
    # Proxy health: retry up to 30s. Even after uvicorn binds, the
    # first end-to-end completion through P → NIXL → D needs a moment.
    local proxy_ok=0
    for i in $(seq 1 15); do
        if curl -sf -o /dev/null -X POST "http://localhost:10001/v1/completions" \
             -H "Content-Type: application/json" \
             -H "X-FlowPrefill-SLO-MS: 10000" \
             -d '{"model":"'"$MODEL"'","prompt":"ok","max_tokens":1,"stream":false}' \
             2>/dev/null; then
            proxy_ok=1
            break
        fi
        sleep 2
    done
    if [ $proxy_ok -eq 0 ]; then
        echo "ERROR: proxy didn't pass a smoke completion within 30s."
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

cell_csv_path() {
    # Where this cell's CSV would land. Same naming loadgen uses.
    echo "$1/trial_${2}_policy_${3}.csv"
}

cell_meta_path() {
    echo "$1/trial_${2}_policy_${3}.meta.json"
}

cell_is_complete() {
    # A cell counts as complete only if BOTH its csv AND its meta.json
    # exist on disk AND are non-empty. Catches the failure mode where
    # an MFS hiccup wrote the CSV but left meta.json as 0 bytes — that
    # cell needs to be re-run, not skipped.
    local csv=$(cell_csv_path "$1" "$2" "$3")
    local meta=$(cell_meta_path "$1" "$2" "$3")
    [ -s "$csv" ] && [ -s "$meta" ]
}

policy_has_pending_trials() {
    # Returns 0 (true) if at least one trial in this (rate, policy)
    # still needs running — meaning we need to bring the stack up.
    local rate_dir=$1
    local policy=$2
    for t in "${TRIALS[@]}"; do
        if ! cell_is_complete "$rate_dir" "$t" "$policy"; then
            return 0
        fi
    done
    return 1
}

clean_partial_cell() {
    # Remove any half-written artifacts so loadgen starts from clean
    # state on a re-run. Safe to call on a never-started cell too.
    local csv=$(cell_csv_path "$1" "$2" "$3")
    local meta=$(cell_meta_path "$1" "$2" "$3")
    rm -f "$csv" "$meta"
}

for rate in "${RATES[@]}"; do
    rate_dir="$RUN_DIR/rate_${rate}"
    mkdir -p "$rate_dir"
    log ""
    log "──── rate = $rate req/s ─────────────────────────────────────────"

    for policy in "${POLICIES[@]}"; do
        log ""
        log "  policy = $policy"
        policy_start_s=$(date +%s)

        # Resume: skip bringing the stack up if every trial in this
        # (rate, policy) cell already has its CSV on disk. Saves the
        # ~60s bring-up cost on a re-run that's only filling gaps.
        if ! policy_has_pending_trials "$rate_dir" "$policy"; then
            log "  [skip] all ${#TRIALS[@]} trials already complete for ($rate, $policy)"
            CELLS_DONE=$((CELLS_DONE + ${#TRIALS[@]}))
            continue
        fi

        bring_up_stack "$policy"
        log "  [stack] bring-up took $(human_dur $(($(date +%s) - policy_start_s)))"

        for trial in "${TRIALS[@]}"; do
            CELLS_DONE=$((CELLS_DONE + 1))

            # Resume: per-trial skip when the cell is fully complete
            # (both csv AND meta.json present + non-empty). Partial
            # writes get cleaned up so loadgen starts from scratch.
            if cell_is_complete "$rate_dir" "$trial" "$policy"; then
                log "    trial $trial  (cell $CELLS_DONE/$TOTAL_CELLS) — skipped (complete)"
                continue
            fi
            clean_partial_cell "$rate_dir" "$trial" "$policy"

            cell_start_s=$(date +%s)
            elapsed_s=$((cell_start_s - SWEEP_START_S))
            # ETA = elapsed * (remaining / done). Crude but useful after
            # the first ~5 cells warm up the estimate.
            if [ $CELLS_DONE -gt 1 ]; then
                eta_s=$(( elapsed_s * (TOTAL_CELLS - CELLS_DONE + 1) / (CELLS_DONE - 1) ))
                eta_str=$(human_dur $eta_s)
            else
                eta_str="—"
            fi
            log "    trial $trial  (cell $CELLS_DONE/$TOTAL_CELLS  elapsed=$(human_dur $elapsed_s)  ETA=$eta_str)"

            MODE=$policy python3 benchmarks/flowprefill/loadgen.py \
                --rate "$rate" \
                --warmup-s "$WARMUP_S" \
                --measure-s "$MEASURE_S" \
                --tier-split "$TIER_SPLIT" \
                --master-seed "$MASTER_SEED" \
                --trial-id "$trial" \
                --max-tokens "$MAX_TOKENS" \
                --output-dir "$rate_dir"

            log "    trial $trial done in $(human_dur $(($(date +%s) - cell_start_s)))"
        done

        archive_logs_for_policy "$rate" "$policy"
        log "  policy=$policy total $(human_dur $(($(date +%s) - policy_start_s)))"
    done
done

log ""
log "Sweep finished in $(human_dur $(($(date +%s) - SWEEP_START_S)))"

echo
echo "════════════════════════════════════════════════════════════════════"
echo "Sweep complete: $RUN_DIR"
# Pick the middle rate of whatever was actually swept as the
# "suggested first analyze target" — works for 1-rate sanity runs too.
mid_idx=$(( ${#RATES[@]} / 2 ))
echo "Next: python3 benchmarks/flowprefill/analyze.py $RUN_DIR/rate_${RATES[$mid_idx]}"
echo "  (use the contention-band rate's directory; analyze each rate dir"
echo "   separately, or sweep-aggregate manually)"
echo "════════════════════════════════════════════════════════════════════"
