# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-boundary preemption check for FlowPrefill.

Called at each attention op boundary (once per transformer layer). Performs
a 1-byte collective vote across TP workers via `torch.distributed.all_reduce`
with `ReduceOp.MAX`. The collective acts as both the decision mechanism
(unanimous result) and the TP sync barrier.

Milestone A (current): the local flag is always 0. We log the vote to validate
that the injection point fires reliably under PIECEWISE compilation and that
TP coordination works. Actual preemption signaling (an mp.Event from the SLO
monitor) is Milestone B; raising PreemptionException to abort the forward
pass is Milestone C.
"""

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)

# Per-call counter for throttling the INFO-level summary log. DEBUG-level
# logs fire on every call.
_invocation_count = 0
_LOG_EVERY_N = 80  # ~once per Llama 70B forward pass (80 attention boundaries)


def preempt_check_at_attention(layer_name: str) -> None:
    """Vote across TP workers on whether to preempt.

    Milestone A: log-only, vote always 0. Always-on regardless of whether the
    SLO monitor is enabled on this node — keeps the code path uniform across
    configurations. Cost is ~10-20us per call on NVLink (1-byte all-reduce),
    negligible against attention compute.
    """
    global _invocation_count

    # TP may not be initialized very early in startup (before model creation
    # completes the parallel-state setup). Skip silently in that case.
    try:
        tp_group = get_tp_group()
    except AssertionError:
        return

    if tp_group.world_size > 1:
        # Real vote: 1-element int32 with value 0 (no signal source yet).
        # MAX semantics: any worker voting 1 -> all workers see 1.
        local_flag = torch.zeros(1, device=tp_group.device, dtype=torch.int32)
        dist.all_reduce(
            local_flag, op=dist.ReduceOp.MAX, group=tp_group.device_group
        )
        vote = local_flag.item()
    else:
        # Single GPU: no peers, skip the collective but still account for the
        # invocation so the fire-rate validation works on dev setups.
        vote = 0

    _invocation_count += 1

    # Per-call DEBUG log — suitable for fine-grained diagnosis.
    logger.debug(
        "preempt_check: rank=%d layer=%s vote=%d invocation=%d",
        tp_group.rank_in_group,
        layer_name,
        vote,
        _invocation_count,
    )

    # Periodic INFO summary — suitable for validation runs. At Llama 70B with
    # _LOG_EVERY_N=80, this fires once per forward pass per worker.
    if _invocation_count % _LOG_EVERY_N == 0:
        logger.info(
            "preempt_check fired: rank=%d layer=%s vote=%d "
            "total_invocations=%d (logging every %d)",
            tp_group.rank_in_group,
            layer_name,
            vote,
            _invocation_count,
            _LOG_EVERY_N,
        )
