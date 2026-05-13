# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-boundary preemption check for FlowPrefill.

Called at each attention op boundary (once per transformer layer). Performs
a 1-byte collective vote across TP workers via `torch.distributed.all_reduce`
with `ReduceOp.MAX`. The collective acts as both the decision mechanism
(unanimous result) and the TP sync barrier.

Vote scoping: the local vote is `1` iff the SLO monitor's
`preempt_target_step_id` matches this worker's `current_step_id` (set per
forward pass from `SchedulerOutput.step_id`). Stale targets — pointing at a
step that's already finished — don't match any future step's current_step_id
and are naturally ignored. See Race Conditions.md for the design rationale.

Ordering invariant (CRITICAL): the all_reduce participates BEFORE any
PreemptionException is raised. Skipping the all_reduce on any worker would
deadlock the cluster — surviving workers would block waiting forever for the
non-participant. See Race Conditions.md #3.
"""

from typing import Optional

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


class PreemptionException(Exception):
    """Raised inside the model forward pass when the TP-wide preempt vote
    returns 1. Engine core catches this exception, releases the in-flight
    batch's KV cache via scheduler.preempt(), and lets the next step admit
    fresh requests according to SLO priority.
    """

    def __init__(self, step_id: int, layer_name: str) -> None:
        super().__init__(
            f"Preempted at step_id={step_id} layer={layer_name}"
        )
        self.step_id = step_id
        self.layer_name = layer_name


# Per-call counter for throttling the INFO-level summary log. DEBUG-level
# logs fire on every call.
_invocation_count = 0
_LOG_EVERY_N = 80  # ~once per Llama 70B forward pass (80 attention boundaries)

# Process-shared preempt target. Holds the step_id the SLO monitor wants
# to preempt. Set by the SLO monitor (in engine core); read by every
# worker at every attention op boundary. None until a worker registers
# its inherited mp.Value reference via set_preempt_target(); when unset,
# the local vote defaults to 0 (no signal).
#
# Module-global rather than per-instance is a research-scope choice — matches
# the existing _invocation_count pattern and avoids threading state through
# the attention call sites. See Merge Plan for the refactor needed before
# upstream merge.
_preempt_target_step_id: Optional["object"] = None  # mp.Value('i'), loose-typed

# Worker-local current step_id, set at the start of each forward pass from
# SchedulerOutput.step_id via set_current_step_id(). -1 = no step in flight.
_current_step_id: int = -1


def set_preempt_target(target) -> None:
    """Register the process-shared preempt target for this worker.

    Called once during worker init (after the mp.Value has been passed via
    process kwargs at spawn). All workers see the same underlying shared int.
    """
    global _preempt_target_step_id
    _preempt_target_step_id = target
    logger.info("preempt_check: preempt target registered")


def set_current_step_id(step_id: int) -> None:
    """Set this worker's current step_id at the start of a forward pass.

    Called by the worker's execute_model entry point before the model runs.
    `preempt_check_at_attention` reads this and compares to the shared
    preempt target. Step_ids are monotonic, set by engine core in
    SchedulerOutput.step_id (Race Conditions.md #2).
    """
    global _current_step_id
    _current_step_id = step_id


def preempt_check_at_attention(layer_name: str) -> None:
    """Vote across TP workers on whether to preempt the current step.

    Local vote = 1 iff (preempt target is registered) AND (target step_id ==
    this worker's current step_id). The TP collective MAX-aggregates votes
    so that any rank voting 1 propagates globally.

    Ordering MUST be: compute local vote → all_reduce → raise.
    Raising before the all_reduce would deadlock peer ranks (Race Conditions.md #3).
    """
    global _invocation_count

    # Skip during CUDA stream capture: vLLM captures CUDA graphs at warmup
    # time (for uniform decode batches). Inside the capture context, any
    # CPU-GPU sync (like `.item()`) raises `cudaErrorStreamCaptureUnsupported`.
    # Our vote is meaningless during a dummy warmup batch anyway, and at
    # replay time the captured graph runs without our Python check — which
    # is the right semantic (we don't preempt decode batches).
    if torch.cuda.is_current_stream_capturing():
        return

    # TP may not be initialized very early in startup (before model creation
    # completes the parallel-state setup). Skip silently in that case.
    try:
        tp_group = get_tp_group()
    except AssertionError:
        return

    # Compute the local vote first. Cheap user-space load on a shared
    # mmap'd byte for the target; comparison against worker-local int.
    target_step_id = (
        _preempt_target_step_id.value
        if _preempt_target_step_id is not None
        else -1
    )
    matches = target_step_id >= 0 and target_step_id == _current_step_id
    local_value = 1 if matches else 0

    if tp_group.world_size > 1:
        # All workers MUST reach this all_reduce. Do not raise above this
        # line — peers would hang at the collective forever.
        local_flag = torch.full(
            (1,), local_value, device=tp_group.device, dtype=torch.int32
        )
        dist.all_reduce(
            local_flag, op=dist.ReduceOp.MAX, group=tp_group.device_group
        )
        vote = int(local_flag.item())
    else:
        # Single GPU: no peers, the local value is the global value.
        # TP=1 preempt behavior is research-scope-only; see Merge Plan.
        vote = local_value

    _invocation_count += 1

    # Per-call DEBUG log — fine-grained diagnosis only.
    logger.debug(
        "preempt_check: rank=%d layer=%s vote=%d target=%d current=%d "
        "invocation=%d",
        tp_group.rank_in_group,
        layer_name,
        vote,
        target_step_id,
        _current_step_id,
        _invocation_count,
    )

    # Periodic INFO summary for validation runs.
    if _invocation_count % _LOG_EVERY_N == 0:
        logger.info(
            "preempt_check fired: rank=%d layer=%s vote=%d target=%d "
            "current=%d total_invocations=%d (logging every %d)",
            tp_group.rank_in_group,
            layer_name,
            vote,
            target_step_id,
            _current_step_id,
            _invocation_count,
            _LOG_EVERY_N,
        )

    # ORDERING (Race Conditions.md #3): raise only AFTER the all_reduce has
    # returned on this worker. Since all_reduce is a collective barrier and
    # all workers compute the same global vote, all workers reach this point
    # together and either all raise or none do.
    if vote == 1:
        raise PreemptionException(step_id=_current_step_id, layer_name=layer_name)
