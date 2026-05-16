#!/bin/bash
# FlowPrefill stubbornness-threshold sweep — test whether Rule 2's 90%
# default is the dominant brake on preempt activity at contention.
#
# Motivation (from the first sweep's archived logs):
#   rate=10 conservative: 54 PREEMPT INTENT vs 272 Rule 2 stubborn blocks
#     → 5× as many preempts were *blocked by stubbornness* as fired
#   rate=8  conservative: 6 INTENT vs 16 Rule 2 blocks (~2.7:1)
#
# Hypothesis: lowering FLOWPREFILL_STUBBORN_LAYER_FRAC from 0.9 toward
# 0.5 lets more preempts actually fire under load. If attainment % moves
# with the threshold, Rule 2 is the gating bottleneck. If flat, the
# zero-sum / snapshot-staleness hypotheses are stronger.
#
# Fixes rate, urgent band, and trials; varies STUBBORN_FRAC.
#
# Time budget: 2 fracs × 3 policies × 3 trials = 18 cells × ~5.5 min
#   + 6 stack restarts × ~2 min ≈ 1h 50m. Combine with existing
#   first_real_run/rate_8/ data (frac=0.9) for a 3-point analysis.
#
# Output: $RUN_DIR/frac_<X>/trial_<N>_policy_<P>.{csv,meta.json}
# Plus per-cell archived prefill/decode/proxy logs.

set -e
set -u

# ─── Knobs ────────────────────────────────────────────────────────────────
# Two extremes — combine with existing first_real_run/rate_8/ data
# (which is at the default frac=0.9) for a 3-point characterization:
#   0.0 → Rule 2 blocks every in-batch preempt (progress > 0 after layer 1).
#         Tests what FlowPrefill achieves with SLO-aware ADMISSION ONLY
#         (heapify) — no in-batch preempts whatsoever.
#   0.5 → aggressive — preempts allowed past the half-way layer
# (0.9 is the default and already measured in first_real_run/rate_8/)
STUBBORN_FRACS=(0.0 0.5)

# read -ra explicitly word-splits — never trust the (${VAR:-"a b c"})
# pattern; it collapses to a 1-element array on some bash versions.
read -ra TRIALS    <<< "${TRIALS:-0 1 2}"
read -ra POLICIES  <<< "${POLICIES:-control conservative aggressive}"

RATE=${RATE:-8}                       # the contention point where Rule 2 already blocked
MASTER_SEED=${MASTER_SEED:-42}
WARMUP_S=${WARMUP_S:-30}
MEASURE_S=${MEASURE_S:-300}
TIER_SPLIT=${TIER_SPLIT:-0.2}
MAX_TOKENS=${MAX_TOKENS:-4}

# Urgent band — leaving at default [1.0, 1.3] so this sweep compares
# directly to the first-real-run (same workload, only stubborn_frac
# varies). If you want to combine with the band sweep findings,
# override via env: URGENT_BAND_LO=1.5 URGENT_BAND_HI=2.0.
URGENT_BAND_LO=${URGENT_BAND_LO:-1.0}
URGENT_BAND_HI=${URGENT_BAND_HI:-1.3}
GENEROUS_BAND_LO=${GENEROUS_BAND_LO:-3.0}
GENEROUS_BAND_HI=${GENEROUS_BAND_HI:-10.0}

BROKEN_GPUS="${BROKEN_GPUS:-4}"
# ──────────────────────────────────────────────────────────────────────────

RUN_NAME="${1:-stubborn_sweep_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="/workspace/benchmark_runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SWEEP_LOG="$RUN_DIR/sweep.log"
exec > >(tee -a "$SWEEP_LOG") 2>&1

TOTAL_CELLS=$(( ${#STUBBORN_FRACS[@]} * ${#POLICIES[@]} * ${#TRIALS[@]} ))
CELLS_DONE=0
SWEEP_START_S=$(date +%s)

log() { printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
human_dur() {
    local s=$1
    printf "%dh%02dm%02ds" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

log "════════════════════════════════════════════════════════════════════"
log "FlowPrefill stubborn-frac sweep:  $RUN_NAME"
log "  stubborn fracs : ${STUBBORN_FRACS[*]}"
log "  rate (fixed)   : $RATE req/s"
log "  urgent band    : [$URGENT_BAND_LO, $URGENT_BAND_HI]"
log "  generous band  : [$GENEROUS_BAND_LO, $GENEROUS_BAND_HI]"
log "  policies       : ${POLICIES[*]}"
log "  trials         : ${TRIALS[*]}"
log "  warmup         : ${WARMUP_S}s   measure: ${MEASURE_S}s"
log "  seed           : $MASTER_SEED"
log "  max_tokens     : $MAX_TOKENS"
log "  broken_gpus    : ${BROKEN_GPUS:-<none>}"
log "  output         : $RUN_DIR"
log "  cells          : $TOTAL_CELLS total (~4h 30m est)"
log "════════════════════════════════════════════════════════════════════"

log "GPU state at sweep start:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | while IFS= read -r line; do log "  $line"; done
log "════════════════════════════════════════════════════════════════════"

# ─── stack lifecycle ──────────────────────────────────────────────────────

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
    # $1 = MODE, $2 = STUBBORN_FRAC for this run.
    #
    # FLOWPREFILL_STUBBORN_LAYER_FRAC is read at module-import time by
    # preempt_check.py (top-level constant). So we MUST set it in the
    # env *before* launching the prefill process — exporting it after
    # would have no effect.
    local mode=$1
    local frac=$2
    echo "  [stack] killing previous instances..."
    BROKEN_GPUS="$BROKEN_GPUS" bash deploy/kill_gpu.sh || true
    sleep 3
    : > /tmp/prefill.log
    : > /tmp/decode.log
    : > /tmp/proxy.log

    echo "  [stack] starting prefill + decode (MODE=$mode  STUBBORN_FRAC=$frac)..."
    # STUBBORN_FRAC only meaningful on prefill (where the SLO monitor
    # and preempt_check live). Set it inline on the prefill launch.
    MODE=$mode FLOWPREFILL_STUBBORN_LAYER_FRAC=$frac \
        bash deploy/start_prefill_nixl.sh
    MODE=$mode bash deploy/start_decode_nixl.sh

    wait_for_uvicorn /tmp/prefill.log 8100 prefill &
    local pwait=$!
    wait_for_uvicorn /tmp/decode.log  8200 decode  &
    local dwait=$!
    wait $pwait || { echo "ERROR: prefill failed"; return 1; }
    wait $dwait || { echo "ERROR: decode failed";  return 1; }

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
    local frac_dir=$1
    local policy=$2
    cp /tmp/prefill.log "$frac_dir/prefill_${policy}.log" 2>/dev/null || true
    cp /tmp/decode.log  "$frac_dir/decode_${policy}.log"  2>/dev/null || true
    cp /tmp/proxy.log   "$frac_dir/proxy_${policy}.log"   2>/dev/null || true
}

# ─── resume support ───────────────────────────────────────────────────────

cell_csv_path() { echo "$1/trial_${2}_policy_${3}.csv"; }
cell_meta_path() { echo "$1/trial_${2}_policy_${3}.meta.json"; }
cell_is_complete() {
    [ -s "$(cell_csv_path "$1" "$2" "$3")" ] && [ -s "$(cell_meta_path "$1" "$2" "$3")" ]
}
clean_partial_cell() {
    rm -f "$(cell_csv_path "$1" "$2" "$3")" "$(cell_meta_path "$1" "$2" "$3")"
}
frac_has_pending_for_policy() {
    local frac_dir=$1
    local policy=$2
    for t in "${TRIALS[@]}"; do
        cell_is_complete "$frac_dir" "$t" "$policy" || return 0
    done
    return 1
}

# Pull MODEL once for the smoke completion in bring_up_stack.
eval "$(python3 deploy/config.py 2>/dev/null)"

# ─── main loop ────────────────────────────────────────────────────────────

for frac in "${STUBBORN_FRACS[@]}"; do
    frac_dir="$RUN_DIR/frac_${frac}"
    mkdir -p "$frac_dir"

    log ""
    log "──── stubborn_frac = $frac ─────────────────────────────────────"

    for policy in "${POLICIES[@]}"; do
        log ""
        log "  policy = $policy"
        policy_start_s=$(date +%s)

        if ! frac_has_pending_for_policy "$frac_dir" "$policy"; then
            log "  [skip] all ${#TRIALS[@]} trials already complete for (frac=$frac, $policy)"
            CELLS_DONE=$((CELLS_DONE + ${#TRIALS[@]}))
            continue
        fi

        bring_up_stack "$policy" "$frac"
        log "  [stack] bring-up took $(human_dur $(($(date +%s) - policy_start_s)))"

        for trial in "${TRIALS[@]}"; do
            CELLS_DONE=$((CELLS_DONE + 1))
            if cell_is_complete "$frac_dir" "$trial" "$policy"; then
                log "    trial $trial  (cell $CELLS_DONE/$TOTAL_CELLS) — skipped (complete)"
                continue
            fi
            clean_partial_cell "$frac_dir" "$trial" "$policy"

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
                --urgent-band-lo "$URGENT_BAND_LO" \
                --urgent-band-hi "$URGENT_BAND_HI" \
                --generous-band-lo "$GENEROUS_BAND_LO" \
                --generous-band-hi "$GENEROUS_BAND_HI" \
                --output-dir "$frac_dir"

            log "    trial $trial done in $(human_dur $(($(date +%s) - cell_start_s)))"
        done

        archive_logs_for_cell "$frac_dir" "$policy"
        log "  policy=$policy total $(human_dur $(($(date +%s) - policy_start_s)))"
    done
done

log ""
log "Stubborn sweep finished in $(human_dur $(($(date +%s) - SWEEP_START_S)))"

echo
echo "════════════════════════════════════════════════════════════════════"
echo "Stubborn sweep complete: $RUN_DIR"
echo "Next: compare attainment + preempt counts across frac values."
echo "      Watch: rate of (PREEMPT INTENT) / (PREEMPT INTENT + Rule 2 stubborn)"
echo "      should rise as frac → 1.0, and attainment should track."
echo "════════════════════════════════════════════════════════════════════"
