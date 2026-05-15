# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slack-aware SLO monitor for FlowPrefill.

Runs as a daemon thread alongside the engine core busy loop. Reads the
scheduler's atomic snapshot, computes slack-aware priorities (S-EDF) for
waiting + running requests, and logs preempt intent when a waiting request
should displace a running one.

Logging milestone: emits intent only; does not act on the scheduler.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)


# TTFT prediction coefficients. Linear model: predicted_TTFT_ms = a * tokens + c.
# Refit from benchmarks/flowprefill/profile_ttft.py on Llama 3.3 70B / A100 TP=4
# (single-flight, uncontended, 6 buckets 256-8000 tokens, R²=0.9996).
# Re-run the profiler on any new (model, hardware, TP) combination.
TTFT_COEFF_A_MS_PER_TOKEN = 0.173046
TTFT_COEFF_C_MS = 34.133

# SLO target.
# Overridable via env var FLOWPREFILL_SLO_BASE_MS for validation runs — lets
# us lower the SLO below normal TTFT so a regular request will breach it
# without needing artificially slow prompts. Set in deploy/config.sh.
# TODO: make this a function of num_prompt_tokens once we vary prompt lengths
# in the benchmark — base_ms + k_ms_per_token * num_prompt_tokens.
# TODO: replace env var with proper config plumbing before upstream merge
# (see FlowPrefill Merge Plan).
TTFT_SLO_BASE_MS = float(os.environ.get("FLOWPREFILL_SLO_BASE_MS", "500.0"))

# Preemption decision margin. Trigger intent when top_waiting.priority exceeds
# bottom_running.priority by this factor. Avoids thrashing on near-equal
# priorities.
PREEMPT_MARGIN = 1.2

# Poll interval for the monitor thread.
MONITOR_POLL_INTERVAL_S = 0.005

# Heartbeat cadence when there is no preempt intent. Prevents log spam at the
# 5ms tick rate while still giving us periodic visibility.
HEARTBEAT_INTERVAL_S = 30.0


@dataclass
class SchedulerSnapshot:
    """Lock-free snapshot of scheduler state, double-buffered via atomic
    reference swap on Scheduler._snapshot.

    Contains references (not deep copies) to Request objects. The fields the
    monitor consumes (num_prompt_tokens, arrival_time, request_id) are set in
    Request.__init__ and never mutated. Mutable fields like num_computed_tokens
    are not used.
    """

    waiting: list["Request"] = field(default_factory=list)
    running: list["Request"] = field(default_factory=list)
    snapshot_time: float = 0.0  # time.monotonic() when snapshot was built
    # FlowPrefill: step_id this snapshot corresponds to. Set by the scheduler
    # atomically with snapshot publication (see Race Conditions.md #2).
    # Monitor uses this to scope its preempt target to a specific step.
    step_id: int = -1
    # FlowPrefill: req_ids that are in the CURRENT step's forward pass
    # (i.e. keys of scheduler_output.num_scheduled_tokens). Monitor uses
    # this to distinguish in-batch vs not-in-batch running requests so it
    # can route the preempt action correctly:
    #   - in-batch target  → abort the forward pass (mp.Value step_id)
    #   - not-in-batch target → scheduler-level removal, no abort needed
    current_batch_req_ids: set[str] = field(default_factory=set)


@dataclass
class RequestEvaluation:
    """Per-request S-EDF evaluation result. Captured for logging so we can
    back-fit SLO and TTFT coefficients later from production traces."""

    request_id: str
    num_prompt_tokens: int
    arrival_time: float
    predicted_ttft_ms: float
    slo_ms: float
    time_until_deadline_ms: float
    slack_ms: float
    priority: float
    # Adaptive stubbornness Rule 1: a request that's already been preempted
    # once refuses to be preempted again. Read from Request.num_preemptions
    # which the scheduler increments inside _preempt_request. Used by the
    # monitor to filter ineligible victims before calling policy.decide.
    num_preemptions: int = 0


class TTFTPredictor:
    """Linear-regression TTFT estimator. Coefficients are hardcoded for now;
    profile + refit before the SLO numbers are trustworthy."""

    def __init__(
        self,
        a_ms_per_token: float = TTFT_COEFF_A_MS_PER_TOKEN,
        c_ms: float = TTFT_COEFF_C_MS,
    ) -> None:
        self.a = a_ms_per_token
        self.c = c_ms

    def predict_ms(self, num_prompt_tokens: int, batch_size: int = 1) -> float:
        # batch_size is unused for now. Prefill is compute-bound past the GPU
        # saturation point — per-request TTFT in a batch scales with total
        # batched tokens, not batch size. When we predict for batched contexts
        # we will pass total batched tokens instead of num_prompt_tokens.
        del batch_size
        return self.a * num_prompt_tokens + self.c


def _extract_per_request_slo_ms(request: "Request") -> float | None:
    """Read FlowPrefill per-request SLO from the request's trace_headers.

    The completion server stuffs the value of `X-FlowPrefill-SLO-MS` into
    `trace_headers` under the lowercase key `x-flowprefill-slo-ms`. We're
    abusing trace_headers (which is intended for OpenTelemetry) as a
    generic header carrier to avoid schema changes. See Merge Plan.

    Returns None if no header was set; caller falls back to the
    server-side default.
    """
    headers = getattr(request, "trace_headers", None)
    if not headers:
        return None
    raw = headers.get("x-flowprefill-slo-ms")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        # Malformed header — fall back silently to default.
        return None


def compute_ttft_slo_ms(request: "Request | None" = None) -> float:
    """SLO target in ms for a request.

    Priority order:
    1. Per-request override via `X-FlowPrefill-SLO-MS` header (read from
       request.trace_headers).
    2. Server-side default — currently the env-var-overridable
       TTFT_SLO_BASE_MS constant.

    Production-grade would replace step 2 with a tier-based lookup or a
    linear-regression default (see Merge Plan).
    """
    if request is not None:
        per_request = _extract_per_request_slo_ms(request)
        if per_request is not None:
            return per_request
    return TTFT_SLO_BASE_MS


def compute_request_priority(
    request: "Request", predictor: "TTFTPredictor | None" = None
) -> float:
    """S-EDF priority for a request, shared between SLOMonitor and the
    scheduler's SLO-aware admission heapify (Phase 5).

    Mirrors `SLOMonitor._evaluate`'s priority math so the monitor's
    snapshot-time judgment and the scheduler's admission-time judgment agree
    on which waiting request is most urgent.

    Higher priority = more urgent.
    """
    if predictor is None:
        predictor = TTFTPredictor()
    predicted_ttft_ms = predictor.predict_ms(request.num_prompt_tokens)
    slo_ms = compute_ttft_slo_ms(request)
    deadline_s = request.arrival_time + slo_ms / 1000.0
    time_until_deadline_ms = (deadline_s - time.time()) * 1000.0
    slack_ms = time_until_deadline_ms - predicted_ttft_ms
    sign = 1.0 if slack_ms >= 0 else -1.0
    denom_ms = max(abs(time_until_deadline_ms), 1.0)
    return sign / denom_ms


class PreemptPolicy:
    """Strategy for deciding which request (if any) to preempt.

    Returns the `RequestEvaluation` of the chosen victim, or None for
    "no preempt this tick." Two implementations:

    - `ConservativePolicy` (default): preempt only when even the BEST
      running request is less urgent than the top waiting (margin-adjusted).
      Treats preemption as exception, not norm. Avoids regressing throughput
      under load — preempts happen only when truly justified.

    - `AggressivePolicy` (paper's S-EDF): preempt when the worst running
      is beaten by the top waiting. Preempts happen often; more sensitive
      to SLO breaches but more thrash-prone.

    The policy is injected at SLOMonitor construction. Default is
    conservative; switch to aggressive only when explicitly desired.
    """

    def decide(
        self,
        waiting_evals: list[RequestEvaluation],
        running_evals: list[RequestEvaluation],
        margin: float,
    ) -> RequestEvaluation | None:
        raise NotImplementedError


class ConservativePolicy(PreemptPolicy):
    """Default. Preempt only if top_waiting beats best_running by margin.

    Branches on slack signs first, then applies the margin only where it
    is mathematically well-defined (both priorities positive). The
    multiplicative margin inverts under negative priorities: for any
    negative best_running, best_running * 1.2 sits BELOW best_running on
    the number line, so a less-urgent waiter would still clear the gate.
    See Unknowns.md "Infinite preempt loop" for the empirical evidence.

    Four sign quadrants:
      - top_waiting hopeless (slack < 0): never preempt for it. A hopeless
        waiter shouldn't displace a running request — preempting wastes
        the running request's prefill work for a request that will miss
        its deadline anyway.
      - running hopeless (slack < 0), waiter meetable: always preempt.
        Even the best running won't meet its deadline; give the slot to
        the waiter that can.
      - both meetable: standard S-EDF margin check.
    """

    def decide(
        self,
        waiting_evals: list[RequestEvaluation],
        running_evals: list[RequestEvaluation],
        margin: float,
    ) -> RequestEvaluation | None:
        top_waiting = max(waiting_evals, key=lambda e: e.priority)
        best_running = max(running_evals, key=lambda e: e.priority)

        if top_waiting.slack_ms < 0:
            return None
        if best_running.slack_ms < 0:
            return min(running_evals, key=lambda e: e.priority)
        if top_waiting.priority > best_running.priority * margin:
            return min(running_evals, key=lambda e: e.priority)
        return None


class AggressivePolicy(PreemptPolicy):
    """Paper's S-EDF. Preempt if top_waiting beats bottom_running by margin.

    More sensitive to SLO breaches, but causes more preempts and
    higher thrash risk under load. Kept as an injectable alternative
    for benchmark comparisons against the conservative default.

    Sign-quadrant branching matches ConservativePolicy (see that
    docstring for the rationale); the only difference is comparing
    against bottom_running rather than best_running in the margin
    branch.
    """

    def decide(
        self,
        waiting_evals: list[RequestEvaluation],
        running_evals: list[RequestEvaluation],
        margin: float,
    ) -> RequestEvaluation | None:
        top_waiting = max(waiting_evals, key=lambda e: e.priority)
        bottom_running = min(running_evals, key=lambda e: e.priority)

        if top_waiting.slack_ms < 0:
            return None
        if bottom_running.slack_ms < 0:
            return bottom_running
        if top_waiting.priority > bottom_running.priority * margin:
            return bottom_running
        return None


class SLOMonitor:
    """Background monitor that evaluates slack-aware urgency every poll tick
    and signals preempt intent.

    The monitor never mutates scheduler state directly. When it decides to
    preempt a running request, it routes the action based on whether the
    target is in the current step's forward pass:

    - **Target in current batch**: writes the snapshot's step_id into the
      process-shared `preempt_target_step_id` (mp.Value). Worker processes
      read it at every attention op, compare to their current step_id, and
      raise PreemptionException through a TP collective vote. The forward
      pass aborts mid-flight.

    - **Target NOT in current batch**: pushes the target's request_id onto
      a scheduler-level preempt queue. Engine core drains this queue
      before the next `schedule()` call and removes the target from
      running without touching the forward pass. Cheap — no abort needed.

    The choice of which running request to preempt is delegated to a
    `PreemptPolicy` (Conservative by default; Aggressive available for
    benchmark comparison).

    See Race Conditions.md and Decisions.md for the full design rationale.
    """

    def __init__(
        self,
        scheduler: "Scheduler",
        preempt_target_step_id,
        pending_scheduler_preempts=None,
        policy: PreemptPolicy | None = None,
        poll_interval_s: float = MONITOR_POLL_INTERVAL_S,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        preempt_margin: float = PREEMPT_MARGIN,
    ) -> None:
        self._scheduler = scheduler
        self._preempt_target_step_id = preempt_target_step_id
        # Thread-safe queue (queue.Queue) of req_ids that the monitor wants
        # the engine core to preempt at the scheduler level (not in batch
        # → no abort needed). Engine core drains this at the top of step().
        # If None, scheduler-level preempts are not supported (degrades to
        # always-abort mode).
        self._pending_scheduler_preempts = pending_scheduler_preempts
        self._policy = policy if policy is not None else ConservativePolicy()
        self._predictor = TTFTPredictor()
        self._poll_interval_s = poll_interval_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._preempt_margin = preempt_margin
        self._last_heartbeat_monotonic = 0.0
        self._stop_event = threading.Event()

        # Tell the scheduler to start producing snapshots at the end of
        # each schedule() call. Gated so non-prefill nodes pay no cost.
        scheduler._snapshot_enabled = True

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="slo-monitor"
        )

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "SLO monitor started: poll=%dms heartbeat=%ds margin=%.2fx "
            "ttft_pred=%.2f*tokens+%.1fms slo_base=%.0fms",
            int(self._poll_interval_s * 1000),
            int(self._heartbeat_interval_s),
            self._preempt_margin,
            TTFT_COEFF_A_MS_PER_TOKEN,
            TTFT_COEFF_C_MS,
            TTFT_SLO_BASE_MS,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                # Best-effort: log and continue. Stale snapshots can produce
                # transient errors (e.g. concurrent mutation of Request lists);
                # we don't want a stray exception to kill the monitor thread.
                logger.exception("SLO monitor tick failed")
            self._stop_event.wait(self._poll_interval_s)

    def _tick(self) -> None:
        snap: SchedulerSnapshot | None = getattr(
            self._scheduler, "_snapshot", None
        )
        if snap is None:
            return

        waiting_evals = [self._evaluate(r) for r in snap.waiting]
        running_evals = [self._evaluate(r) for r in snap.running]

        now_monotonic = time.monotonic()

        # Adaptive stubbornness Rule 1: a request that's already been
        # preempted once refuses to give up its slot. Filter ineligible
        # runners before the policy sees them so victim selection AND the
        # "best running" reference both reflect only preempt-eligible
        # candidates. If every running request is stubborn there's
        # nothing the monitor can do — fall through to heartbeat.
        eligible_running_evals = [
            e for e in running_evals if e.num_preemptions == 0
        ]
        stubborn_count = len(running_evals) - len(eligible_running_evals)

        # If either side is empty (no waiters, no runners, or no eligible
        # runners after stubbornness filter), no preemption decision to
        # make — emit a heartbeat with queue stats.
        if not waiting_evals or not eligible_running_evals:
            self._maybe_heartbeat(
                now_monotonic,
                snap,
                waiting_evals,
                running_evals,
                stubborn_count,
            )
            return

        # Delegate the decision to the injected policy, passing only
        # preempt-eligible runners. Default (Conservative): preempt only
        # if top_waiting beats best_running.
        target = self._policy.decide(
            waiting_evals, eligible_running_evals, self._preempt_margin
        )

        if target is None:
            self._maybe_heartbeat(
                now_monotonic,
                snap,
                waiting_evals,
                running_evals,
                stubborn_count,
            )
            return

        top_waiting = max(waiting_evals, key=lambda e: e.priority)
        target_in_batch = target.request_id in snap.current_batch_req_ids

        # FlowPrefill: tell the scheduler which waiter this preempt is for,
        # so the freed slot goes to top_waiting on next admission rather
        # than whoever heapify ranks first (which can differ under
        # priority ties, wall-clock drift, or fresh arrivals between
        # snapshot and schedule()). STORE_ATTR is GIL-atomic; if this
        # races with the scheduler's read+clear and the hint is lost, the
        # monitor's next tick will re-fire and re-set it.
        self._scheduler._preempt_hint_request_id = top_waiting.request_id

        if target_in_batch:
            # In-batch preempt: target is being processed RIGHT NOW.
            # Trigger forward-pass abort via the existing mp.Value path.
            # Workers will see step_id match → all_reduce vote=1 → raise.
            self._preempt_target_step_id.value = snap.step_id
            log_route = "abort"
        else:
            # Not-in-batch preempt: target is in running but not in this
            # step's forward pass. No abort needed — push to engine core's
            # scheduler-level preempt queue, which drains before the next
            # schedule(). Cheap path. Falls back to logging only if the
            # queue wasn't provided (e.g. older config).
            if self._pending_scheduler_preempts is not None:
                self._pending_scheduler_preempts.put(target.request_id)
                log_route = "scheduler-level"
            else:
                log_route = "noop (queue unavailable)"

        logger.info(
            "PREEMPT INTENT (route=%s, step_id=%d): waiting %s "
            "(tokens=%d slack=%.1fms ttft_pred=%.1fms prio=%.4g) "
            "would displace running %s "
            "(tokens=%d slack=%.1fms ttft_pred=%.1fms prio=%.4g) "
            "[in_batch=%s, margin=%.2fx, snapshot_age=%.1fms, "
            "stubborn=%d/%d]",
            log_route,
            snap.step_id,
            top_waiting.request_id,
            top_waiting.num_prompt_tokens,
            top_waiting.slack_ms,
            top_waiting.predicted_ttft_ms,
            top_waiting.priority,
            target.request_id,
            target.num_prompt_tokens,
            target.slack_ms,
            target.predicted_ttft_ms,
            target.priority,
            target_in_batch,
            self._preempt_margin,
            (now_monotonic - snap.snapshot_time) * 1000.0,
            stubborn_count,
            len(running_evals),
        )

    def _evaluate(self, request: "Request") -> RequestEvaluation:
        num_prompt_tokens = request.num_prompt_tokens
        arrival_time = request.arrival_time

        predicted_ttft_ms = self._predictor.predict_ms(num_prompt_tokens)
        slo_ms = compute_ttft_slo_ms(request)
        deadline_s = arrival_time + slo_ms / 1000.0

        # arrival_time is wall-clock (time.time); keep slack computation in the
        # same frame.
        time_until_deadline_ms = (deadline_s - time.time()) * 1000.0
        slack_ms = time_until_deadline_ms - predicted_ttft_ms

        # S-EDF priority. Paper writes priority = sgn(slack) / deadline; we
        # use time_until_deadline (remaining time) as the denominator so
        # priority magnitudes are meaningful for the margin check. Using the
        # absolute deadline timestamp would compress all priorities into a
        # near-identical range and break the 1.2x margin gate.
        sign = 1.0 if slack_ms >= 0 else -1.0
        denom_ms = max(abs(time_until_deadline_ms), 1.0)
        priority = sign / denom_ms

        return RequestEvaluation(
            request_id=request.request_id,
            num_prompt_tokens=num_prompt_tokens,
            arrival_time=arrival_time,
            predicted_ttft_ms=predicted_ttft_ms,
            slo_ms=slo_ms,
            time_until_deadline_ms=time_until_deadline_ms,
            slack_ms=slack_ms,
            priority=priority,
            num_preemptions=getattr(request, "num_preemptions", 0),
        )

    def _maybe_heartbeat(
        self,
        now_monotonic: float,
        snap: SchedulerSnapshot,
        waiting_evals: list[RequestEvaluation],
        running_evals: list[RequestEvaluation],
        stubborn_count: int = 0,
    ) -> None:
        if (
            now_monotonic - self._last_heartbeat_monotonic
            < self._heartbeat_interval_s
        ):
            return
        self._last_heartbeat_monotonic = now_monotonic

        all_evals = waiting_evals + running_evals
        slack_vals = [e.slack_ms for e in all_evals]
        logger.info(
            "SLO monitor heartbeat: waiting=%d running=%d stubborn=%d/%d "
            "slack_ms[min/max]=%s/%s snapshot_age=%.1fms",
            len(snap.waiting),
            len(snap.running),
            stubborn_count,
            len(snap.running),
            f"{min(slack_vals):.1f}" if slack_vals else "n/a",
            f"{max(slack_vals):.1f}" if slack_vals else "n/a",
            (now_monotonic - snap.snapshot_time) * 1000.0,
        )
