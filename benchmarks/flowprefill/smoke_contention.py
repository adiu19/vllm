# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlowPrefill contention smoke test.

Fires two concurrent completion requests through the proxy, sized so that
their token sum exceeds max_num_batched_tokens — forcing the prefill
scheduler into a contention scenario where one runs and one waits.

  - Generous: SLO_MS=5000  → slack stays positive, won't be displaced
  - Urgent  : SLO_MS=50    → slack flips negative immediately

Expected behaviour per mode:
  control      : no PREEMPT INTENT in /tmp/prefill.log. Urgent waits behind generous.
  conservative : PREEMPT INTENT fires for urgent vs generous. Urgent gets ahead.
  aggressive   : PREEMPT INTENT fires sooner / more readily.

This is a SMOKE test — it verifies the pipeline observes the policy
difference, not that the SLO numbers are accurate (8B vs 70B predictor
constants — see project_flowprefill_loadgen_slo.md).

Usage:
  python benchmarks/flowprefill/smoke_contention.py
  # then:
  grep "PREEMPT INTENT" /tmp/prefill.log | tail
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx

# Pull repo root onto sys.path so `from deploy.config import load` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deploy.config import load as _load_deploy_config  # noqa: E402


def main() -> int:
    cfg = _load_deploy_config()
    model = cfg["model"]["name"]
    endpoint = f"http://localhost:{cfg['ports']['proxy_http']}"
    mode = cfg["mode"]

    # ~6000 tokens each; two of these will not co-batch under
    # max_num_batched_tokens=9000 — forces serialization.
    prompt = " ".join(["distributed"] * 6000)

    async def fire(slo_ms: int, label: str) -> None:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{endpoint}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 4,
                    "stream": True,
                },
                headers={"X-FlowPrefill-SLO-MS": str(slo_ms)},
            ) as resp:
                t_first = None
                async for line in resp.aiter_lines():
                    if line.startswith("data: {") and t_first is None:
                        t_first = time.time() - t0
                total = time.time() - t0
        first_ms = f"{t_first * 1000:.0f}ms" if t_first else "—"
        total_ms = f"{total * 1000:.0f}ms"
        print(f"  {label:9s} slo={slo_ms:5d}ms   TTFT={first_ms:>8s}   total={total_ms:>8s}")

    async def run() -> None:
        # Fire generous first; sleep so it claims the running slot; then urgent.
        g = asyncio.create_task(fire(5000, "generous"))
        await asyncio.sleep(0.05)
        u = asyncio.create_task(fire(50,   "urgent"))
        await asyncio.gather(g, u)

    print(f"[smoke] mode={mode}  model={model}  endpoint={endpoint}")
    print(f"[smoke] firing two ~6000-token completions concurrently")
    asyncio.run(run())
    print()
    print("Check the prefill log for preempt activity:")
    print(f"  grep 'PREEMPT INTENT' {cfg['logs']['prefill']} | tail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
