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

import os
from typing import Optional

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


# Adaptive stubbornness Rule 2: a request that has finished more than this
# fraction of its forward pass refuses to be preempted. Default 0.9 → only
# the last 10% of layers are "non-stubborn"; once we're past 90% the local
# vote is forced to 0 even if the SLO monitor's preempt target matches our
# step_id. Configurable via env var so we can sweep it in benchmarks.
_STUBBORN_LAYER_FRAC = float(
    os.environ.get("FLOWPREFILL_STUBBORN_LAYER_FRAC", "0.9")
)


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

# Adaptive stubbornness Rule 2 state.
#
# _current_layer_idx: incremented on each preempt_check_at_attention call
# (one call per transformer layer, in layer order); reset to 0 at the
# start of each forward pass by set_current_step_id(). Worker-local, no
# cross-process sharing — each worker tracks its own forward pass.
#
# _total_num_layers: set once during worker init from
# vllm_config.model_config.hf_config.num_hidden_layers via
# set_total_num_layers(). Deterministic — not inferred from observation.
# 0 means "not registered" (e.g., non-prefill node, FlowPrefill disabled);
# Rule 2 is skipped in that case.
_current_layer_idx: int = 0
_total_num_layers: int = 0


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

    Also resets the Rule 2 layer counter — each forward pass starts at
    layer 0 regardless of how the previous one terminated (completed,
    preempted, errored).
    """
    global _current_step_id, _current_layer_idx
    _current_step_id = step_id
    _current_layer_idx = 0


def set_total_num_layers(n: int) -> None:
    """Register the model's transformer-layer count for Rule 2 stubbornness.

    Called once during worker init alongside set_preempt_target, sourced
    from vllm_config.model_config.hf_config.num_hidden_layers. Constant
    across the worker's lifetime; deterministic — no inference needed.

    If never called (e.g. non-prefill node, FlowPrefill disabled), the
    module-level _total_num_layers stays 0 and Rule 2 is skipped.
    """
    global _total_num_layers
    _total_num_layers = n
    logger.info(
        "preempt_check: total_num_layers=%d registered "
        "(stubborn_layer_frac=%.2f)",
        n,
        _STUBBORN_LAYER_FRAC,
    )


def preempt_check_at_attention(layer_name: str) -> None:
    """Vote across TP workers on whether to preempt the current step.

    Local vote = 1 iff (preempt target is registered) AND (target step_id ==
    this worker's current step_id). The TP collective MAX-aggregates votes
    so that any rank voting 1 propagates globally.

    Ordering MUST be: compute local vote → all_reduce → raise.
    Raising before the all_reduce would deadlock peer ranks (Race Conditions.md #3).
    """
    global _invocation_count, _current_layer_idx

    # Fast path for control mode (FlowPrefill disabled on this node): if no
    # preempt target was registered by the SLO monitor, the vote is forever
    # 0 — skip the all_reduce, the TP lookup, and the layer counter entirely.
    # Zero overhead vs. vanilla vLLM, which is the property the control arm
    # of the benchmark needs. Without this, every attention layer still runs
    # a 1-element MAX all_reduce (~10-30μs × 80 layers ≈ 1-2ms per prefill).
    if _preempt_target_step_id is None:
        return

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

    # Advance the layer counter for this forward pass. Each transformer
    # layer makes exactly one call to preempt_check_at_attention (in
    # layer order), so the post-increment value IS our 1-indexed layer
    # position. Reset to 0 at the start of each forward pass by
    # set_current_step_id().
    _current_layer_idx += 1

    # Compute the local vote first. Cheap user-space load on a shared
    # mmap'd byte for the target; comparison against worker-local int.
    target_step_id = (
        _preempt_target_step_id.value
        if _preempt_target_step_id is not None
        else -1
    )
    matches = target_step_id >= 0 and target_step_id == _current_step_id
    local_value = 1 if matches else 0

    # Adaptive stubbornness Rule 2: even if the SLO monitor wants to
    # preempt us, refuse if we're past the layer-fraction threshold. A
    # request 90%+ through prefill has done the expensive work; throwing
    # it away to admit a fresher waiter wastes more compute than it
    # saves. Worker-local check — all TP workers run the same layers in
    # lockstep, so they see the same _current_layer_idx and reach the
    # same vote independently. The collective MAX then composes to a
    # unanimous 0 (no preempt).
    #
    # Skipped when _total_num_layers is 0 (setter never called → Rule 2
    # disabled). In that case we fall through to the standard step_id
    # match vote — Rule 1 (monitor-side) still applies regardless.
    rule2_blocked = False
    if matches and _total_num_layers > 0:
        progress_frac = _current_layer_idx / _total_num_layers
        if progress_frac > _STUBBORN_LAYER_FRAC:
            local_value = 0
            rule2_blocked = True

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
        "preempt_check: rank=%d layer=%s layer_idx=%d/%d vote=%d "
        "target=%d current=%d invocation=%d rule2_blocked=%s",
        tp_group.rank_in_group,
        layer_name,
        _current_layer_idx,
        _total_num_layers,
        vote,
        target_step_id,
        _current_step_id,
        _invocation_count,
        rule2_blocked,
    )

    # Rule 2 firing is a debuggability event worth surfacing immediately
    # (not throttled like the periodic summary). Only logs when the
    # monitor's target matched our step and we refused — i.e. a preempt
    # that would have fired without stubbornness. Each TP rank logs once
    # per blocked layer; under load this is N_layers_remaining * N_ranks
    # lines per blocked preempt, bounded and useful.
    if rule2_blocked:
        logger.info(
            "Rule 2 stubborn: rank=%d step_id=%d layer_idx=%d/%d "
            "(progress=%.2f > threshold=%.2f) — refusing preempt",
            tp_group.rank_in_group,
            _current_step_id,
            _current_layer_idx,
            _total_num_layers,
            _current_layer_idx / _total_num_layers,
            _STUBBORN_LAYER_FRAC,
        )

    # Periodic INFO summary for validation runs.
    if _invocation_count % _LOG_EVERY_N == 0:
        logger.info(
            "preempt_check fired: rank=%d layer=%s layer_idx=%d/%d "
            "vote=%d target=%d current=%d total_invocations=%d "
            "(logging every %d)",
            tp_group.rank_in_group,
            layer_name,
            _current_layer_idx,
            _total_num_layers,
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
