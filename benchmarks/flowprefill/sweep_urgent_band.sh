#!/bin/bash
# FlowPrefill urgent-band sweep — characterize the SLO regime where
# slack-aware preemption demonstrably helps urgent attainment.
#
# Fixes arrival rate at the contention point (default λ=6 req/s — the
# rate where the original sweep showed visible queueing + preempt
# activity) and varies the URGENT_BAND lower-multiplier across values
# spanning structurally-tight to comfortably-loose. Generous band stays
# fixed.
#
# Hypothesis: there exists a band where most urgents arrive with positive
# slack (savable) AND contention pushes them toward miss — that's the
# regime where FlowPrefill should show measurable lift vs control.
#
# Time budget: 5 bands × 3 policies × 3 trials = 45 cells × ~5.5 min
# + 15 stack restarts × ~2 min ≈ 2h 45m.
#
# Output: $RUN_DIR/band_<lo>_<hi>/trial_<N>_policy_<P>.{csv,meta.json}

set -e
set -u

# ─── Knobs ────────────────────────────────────────────────────────────────
# Urgent band lower multipliers to sweep. Upper bound is held at +0.3
# above lower (matches the original [1.0, 1.3] spacing), but bumped to
# +0.5 above lower for the wider bands where the +0.3 gap is too narrow.
URGENT_BAND_LOS=(1.0 1.2 1.5 2.0 3.0)
URGENT_BAND_WIDTH=0.5   # upper = lower + width, for bands >= 1.2
URGENT_BAND_WIDTH_TIGHT=0.3   # for the tightest band (1.0) — keep original [1.0, 1.3]

GENEROUS_BAND_LO=3.0
GENEROUS_BAND_HI=10.0

RATE=${RATE:-6}                      # contention-band operating point
TRIALS=(${TRIALS:-"0 1 2"})          # 3 trials per cell for exploratory CIs
POLICIES=(${POLICIES:-"control conservative aggressive"})
MASTER_SEED=${MASTER_SEED:-42}
WARMUP_S=${WARMUP_S:-30}
MEASURE_S=${MEASURE_S:-300}
TIER_SPLIT=${TIER_SPLIT:-0.2}
MAX_TOKENS=${MAX_TOKENS:-4}
BROKEN_GPUS="${BROKEN_GPUS:-4}"
# ──────────────────────────────────────────────────────────────────────────

RUN_NAME="${1:-band_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="/workspace/benchmark_runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SWEEP_LOG="$RUN_DIR/sweep.log"
exec > >(tee -a "$SWEEP_LOG") 2>&1

# Compute total cells for ETA
TOTAL_CELLS=$(( ${#URGENT_BAND_LOS[@]} * ${#POLICIES[@]} * ${#TRIALS[@]} ))
CELLS_DONE=0
SWEEP_START_S=$(date +%s)

log() { printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
human_dur() {
    local s=$1
    printf "%dh%02dm%02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

log "════════════════════════════════════════════════════════════════════"
log "FlowPrefill urgent-band sweep:  $RUN_NAME"
log "  rate (fixed)   : $RATE req/s"
log "  urgent bands   : ${URGENT_BAND_LOS[*]}  (each band: [lo, lo+0.3/0.5])"
log "  generous band  : [$GENEROUS_BAND_LO, $GENEROUS_BAND_HI]"
log "  policies       : ${POLICIES[*]}"
log "  trials         : ${TRIALS[*]}"
log "  warmup         : ${WARMUP_S}s   measure: ${MEASURE_S}s"
log "  seed           : $MASTER_SEED"
log "  max_tokens     : $MAX_TOKENS"
log "  broken_gpus    : ${BROKEN_GPUS:-<none>}"
log "  output         : $RUN_DIR"
log "  sweep log      : $SWEEP_LOG"
log "  cells          : $TOTAL_CELLS total (~2h 45m est)"
log "════════════════════════════════════════════════════════════════════"

log "GPU state at sweep start:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | while IFS= read -r line; do log "  $line"; done
log "════════════════════════════════════════════════════════════════════"

# ─── stack lifecycle helpers (mirror run_sweep.sh) ────────────────────────

wait_for_uvicorn() {
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
    echo "ERROR: $svc did not come up in 15 min."
    return 1
}

bring_up_stack() {
    local mode=$1
    echo "  [stack] killing previous instances..."
    BROKEN_GPUS="$BROKEN_GPUS" bash deploy/kill_gpu.sh || true
    sleep 3
    : > /tmp/prefill.log
    : > /tmp/decode.log
    : > /tmp/proxy.log

    echo "  [stack] starting prefill + decode in parallel (MODE=$mode)..."
    MODE=$mode bash deploy/start_prefill_nixl.sh
    MODE=$mode bash deploy/start_decode_nixl.sh
    wait_for_uvicorn /tmp/prefill.log 8100 prefill &
    local pwait=$!
    wait_for_uvicorn /tmp/decode.log  8200 decode  &
    local dwait=$!
    wait $pwait || { echo "ERROR: prefill failed to come up"; return 1; }
    wait $dwait || { echo "ERROR: decode failed to come up";  return 1; }

    echo "  [stack] starting proxy..."
    bash deploy/start_proxy_nixl.sh
    local proxy_ok=0
    for i in $(seq 1 15); do
        if curl -sf -o /dev/null -X POST "http://localhost:10001/v1/completions" \
             -H "Content-Type: application/json" \
             -H "X-FlowPrefill-SLO-MS: 10000" \
             -d '{"model":"'"$MODEL"'","prompt":"ok","max_tokens":1,"stream":false}' \
             2>/dev/null; then
            proxy_ok=1; break
        fi
        sleep 2
    done
    [ $proxy_ok -eq 1 ] || { echo "ERROR: proxy didn't respond"; return 1; }
    echo "  [stack] up."
}

archive_logs_for_cell() {
    # Per (band, policy) snapshot. Smaller granularity than rate sweep
    # because each band's trials run under their own stack.
    local band_dir=$1
    local policy=$2
    cp /tmp/prefill.log "$band_dir/prefill_${policy}.log" 2>/dev/null || true
    cp /tmp/decode.log  "$band_dir/decode_${policy}.log"  2>/dev/null || true
    cp /tmp/proxy.log   "$band_dir/proxy_${policy}.log"   2>/dev/null || true
}

# ─── resume helpers (mirror run_sweep.sh) ─────────────────────────────────

cell_csv_path() { echo "$1/trial_${2}_policy_${3}.csv"; }
cell_meta_path() { echo "$1/trial_${2}_policy_${3}.meta.json"; }
cell_is_complete() {
    [ -s "$(cell_csv_path "$1" "$2" "$3")" ] && [ -s "$(cell_meta_path "$1" "$2" "$3")" ]
}
clean_partial_cell() {
    rm -f "$(cell_csv_path "$1" "$2" "$3")" "$(cell_meta_path "$1" "$2" "$3")"
}
band_has_pending_for_policy() {
    local band_dir=$1
    local policy=$2
    for t in "${TRIALS[@]}"; do
        cell_is_complete "$band_dir" "$t" "$policy" || return 0
    done
    return 1
}

# Pull MODEL once for the smoke completion.
eval "$(python3 deploy/config.py 2>/dev/null)"

# ─── main loop ────────────────────────────────────────────────────────────

for band_lo in "${URGENT_BAND_LOS[@]}"; do
    # Pick band width: 0.3 for the tightest (1.0) to keep original spacing;
    # 0.5 for wider bands so the band has meaningful range without
    # overlapping with the next band's lower bound.
    if awk "BEGIN{exit !($band_lo <= 1.0)}"; then
        width="$URGENT_BAND_WIDTH_TIGHT"
    else
        width="$URGENT_BAND_WIDTH"
    fi
    band_hi=$(awk "BEGIN{printf \"%.1f\", $band_lo + $width}")
    band_dir="$RUN_DIR/band_${band_lo}_${band_hi}"
    mkdir -p "$band_dir"

    log ""
    log "──── urgent band = [$band_lo, $band_hi] ────────────────────────"

    for policy in "${POLICIES[@]}"; do
        log ""
        log "  policy = $policy"
        policy_start_s=$(date +%s)

        if ! band_has_pending_for_policy "$band_dir" "$policy"; then
            log "  [skip] all ${#TRIALS[@]} trials already complete for (band=$band_lo, $policy)"
            CELLS_DONE=$((CELLS_DONE + ${#TRIALS[@]}))
            continue
        fi

        bring_up_stack "$policy"
        log "  [stack] bring-up took $(human_dur $(($(date +%s) - policy_start_s)))"

        for trial in "${TRIALS[@]}"; do
            CELLS_DONE=$((CELLS_DONE + 1))
            if cell_is_complete "$band_dir" "$trial" "$policy"; then
                log "    trial $trial  (cell $CELLS_DONE/$TOTAL_CELLS) — skipped (complete)"
                continue
            fi
            clean_partial_cell "$band_dir" "$trial" "$policy"

            cell_start_s=$(date +%s)
            elapsed_s=$((cell_start_s - SWEEP_START_S))
            if [ $CELLS_DONE -gt 1 ]; then
                eta_s=$(( elapsed_s * (TOTAL_CELLS - CELLS_DONE + 1) / (CELLS_DONE - 1) ))
                eta_str=$(human_dur $eta_s)
            else
                eta_str="—"
            fi
            log "    trial $trial  (cell $CELLS_DONE/$TOTAL_CELLS  elapsed=$(human_dur $elapsed_s)  ETA=$eta_str)"

            MODE=$policy python3 benchmarks/flowprefill/loadgen.py \
                --rate "$RATE" \
                --warmup-s "$WARMUP_S" \
                --measure-s "$MEASURE_S" \
                --tier-split "$TIER_SPLIT" \
                --master-seed "$MASTER_SEED" \
                --trial-id "$trial" \
                --max-tokens "$MAX_TOKENS" \
                --urgent-band-lo "$band_lo" \
                --urgent-band-hi "$band_hi" \
                --generous-band-lo "$GENEROUS_BAND_LO" \
                --generous-band-hi "$GENEROUS_BAND_HI" \
                --output-dir "$band_dir"

            log "    trial $trial done in $(human_dur $(($(date +%s) - cell_start_s)))"
        done

        archive_logs_for_cell "$band_dir" "$policy"
        log "  policy=$policy total $(human_dur $(($(date +%s) - policy_start_s)))"
    done
done

log ""
log "Band sweep finished in $(human_dur $(($(date +%s) - SWEEP_START_S)))"

echo
echo "════════════════════════════════════════════════════════════════════"
echo "Band sweep complete: $RUN_DIR"
echo "Next: analyze attainment per band — does conservative-vs-control"
echo "      gap show a peak in the middle of the band range?"
echo "════════════════════════════════════════════════════════════════════"
