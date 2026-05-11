# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slack-aware SLO monitor for FlowPrefill.

Runs as a daemon thread alongside the engine core busy loop. Reads the
scheduler's atomic snapshot, computes slack-aware priorities (S-EDF) for
waiting + running requests, and logs preempt intent when a waiting request
should displace a running one.

Logging milestone: emits intent only; does not act on the scheduler.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)


# TTFT prediction coefficients.
# Linear model: predicted_TTFT_ms = a * num_prompt_tokens + c
# Anchor: Cursor blog, Llama 2 70B FP16, 2x A100 TP=2, 512 tokens -> 217ms.
# Scaling TP=2 -> TP=4 halves the slope; overhead is roughly constant.
# TODO: replace with coefficients fitted from a profiling sweep on our setup.
TTFT_COEFF_A_MS_PER_TOKEN = 0.21
TTFT_COEFF_C_MS = 20.0

# SLO target.
# TODO: make this a function of num_prompt_tokens once we vary prompt lengths
# in the benchmark — base_ms + k_ms_per_token * num_prompt_tokens.
TTFT_SLO_BASE_MS = 500.0

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


def compute_ttft_slo_ms(num_prompt_tokens: int, batch_size: int = 1) -> float:
    """SLO target in ms. Parameterized signature for forward compatibility;
    constant body for the logging milestone.

    TODO: scale with prompt length once benchmarks vary it:
        return TTFT_SLO_BASE_MS + k_ms_per_token * num_prompt_tokens
    """
    del num_prompt_tokens, batch_size
    return TTFT_SLO_BASE_MS


class SLOMonitor:
    """Background monitor that evaluates slack-aware urgency every poll tick
    and logs preempt intent.

    The monitor never mutates scheduler state. When preemption is wired in
    later, it will set a threading.Event the main thread checks at layer
    boundaries — the main thread does the actual preemption with its own
    consistent view of scheduler state.
    """

    def __init__(
        self,
        scheduler: "Scheduler",
        poll_interval_s: float = MONITOR_POLL_INTERVAL_S,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        preempt_margin: float = PREEMPT_MARGIN,
    ) -> None:
        self._scheduler = scheduler
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

        # If either side is empty we can't reason about preemption — only
        # emit a heartbeat with whatever queue stats we have.
        if not waiting_evals or not running_evals:
            self._maybe_heartbeat(now_monotonic, snap, waiting_evals, running_evals)
            return

        top_waiting = max(waiting_evals, key=lambda e: e.priority)
        bottom_running = min(running_evals, key=lambda e: e.priority)

        # The margin check is sign-aware: when bottom_running has negative
        # priority (already missing SLO), any positive-priority waiting
        # request trivially beats it. When both are positive, we require
        # top_waiting to be at least margin times more urgent.
        if top_waiting.priority > bottom_running.priority * self._preempt_margin:
            logger.info(
                "PREEMPT INTENT: waiting %s (tokens=%d slack=%.1fms "
                "ttft_pred=%.1fms prio=%.4g) would displace running %s "
                "(tokens=%d slack=%.1fms ttft_pred=%.1fms prio=%.4g) "
                "[margin=%.2fx, snapshot_age=%.1fms]",
                top_waiting.request_id,
                top_waiting.num_prompt_tokens,
                top_waiting.slack_ms,
                top_waiting.predicted_ttft_ms,
                top_waiting.priority,
                bottom_running.request_id,
                bottom_running.num_prompt_tokens,
                bottom_running.slack_ms,
                bottom_running.predicted_ttft_ms,
                bottom_running.priority,
                self._preempt_margin,
                (now_monotonic - snap.snapshot_time) * 1000.0,
            )
        else:
            self._maybe_heartbeat(
                now_monotonic, snap, waiting_evals, running_evals
            )

    def _evaluate(self, request: "Request") -> RequestEvaluation:
        num_prompt_tokens = request.num_prompt_tokens
        arrival_time = request.arrival_time

        predicted_ttft_ms = self._predictor.predict_ms(num_prompt_tokens)
        slo_ms = compute_ttft_slo_ms(num_prompt_tokens)
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
        )

    def _maybe_heartbeat(
        self,
        now_monotonic: float,
        snap: SchedulerSnapshot,
        waiting_evals: list[RequestEvaluation],
        running_evals: list[RequestEvaluation],
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
            "SLO monitor heartbeat: waiting=%d running=%d "
            "slack_ms[min/max]=%s/%s snapshot_age=%.1fms",
            len(snap.waiting),
            len(snap.running),
            f"{min(slack_vals):.1f}" if slack_vals else "n/a",
            f"{max(slack_vals):.1f}" if slack_vals else "n/a",
            (now_monotonic - snap.snapshot_time) * 1000.0,
        )
