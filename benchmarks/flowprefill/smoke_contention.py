# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlowPrefill contention smoke test.

Fires two concurrent completion requests through the proxy, sized so that
their token sum exceeds max_num_batched_tokens — forcing the prefill
scheduler into a contention scenario where one runs and one waits.

  - Generous: SLO_MS=5000  → slack stays positive, won't be displaced
  - Urgent  : SLO_MS=50    → slack flips negative immediately

Expected behaviour per mode:
  control      : no PREEMPT INTENT in prefill log. Urgent waits behind generous.
  conservative : PREEMPT INTENT fires for urgent vs generous. Urgent gets ahead.
  aggressive   : PREEMPT INTENT fires sooner / more readily.

This is a SMOKE test — it verifies the pipeline observes the policy
difference, not that the SLO numbers are accurate (8B vs 70B predictor
constants — see project_flowprefill_loadgen_slo.md).

Usage:
  python benchmarks/flowprefill/smoke_contention.py
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx

# Pull repo root onto sys.path so `from deploy.config import load` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deploy.config import load as _load_deploy_config  # noqa: E402


# Each prompt must individually exceed half of max_num_batched_tokens so
# the two together can NOT co-batch — that's what forces the waiting/
# running split the SLO monitor needs to act on. Default 7500 vs the
# 9000 cap leaves headroom; auto-grown below if tokenizer collapses.
TARGET_TOKENS_PER_PROMPT = 7500


def build_contention_prompt(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Repeat a stable word until tokenized length >= target_tokens.

    Returns (prompt_text, actual_token_count). Auto-grows the repetition
    if BPE compresses the input below target.
    """
    word = "distributed"
    reps = target_tokens
    for _ in range(4):
        text = " ".join([word] * reps)
        n = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n >= target_tokens:
            return text, n
        # Tokenizer compressed below target — bump and retry.
        reps = int(reps * (target_tokens / max(n, 1)) * 1.05)
    # Last attempt — return whatever we got.
    text = " ".join([word] * reps)
    n = len(tokenizer(text, add_special_tokens=False).input_ids)
    return text, n


def grep_prefill_log(log_path: Path, after_ms: int) -> tuple[int, list[str]]:
    """Return (count_of_PREEMPT_INTENT_lines, last_5_lines) emitted after
    `after_ms` epoch ms. Matches by ISO-ish timestamp grep — best effort.
    Returns (count, lines) globally if no timestamp parsing possible.
    """
    if not log_path.exists():
        return 0, []
    relevant = []
    with log_path.open("r", errors="replace") as f:
        for line in f:
            if "PREEMPT INTENT" in line or "Rule 2 stubborn" in line:
                relevant.append(line.rstrip())
    return sum(1 for ln in relevant if "PREEMPT INTENT" in ln), relevant[-5:]


def main() -> int:
    cfg = _load_deploy_config()
    model = cfg["model"]["name"]
    endpoint = f"http://localhost:{cfg['ports']['proxy_http']}"
    mode = cfg["mode"]
    prefill_log = Path(cfg["logs"]["prefill"])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)

    prompt, n_tokens = build_contention_prompt(tok, TARGET_TOKENS_PER_PROMPT)

    # Snapshot preempt count before firing
    n_before, _ = grep_prefill_log(prefill_log, 0)

    print(f"[smoke] mode={mode}  model={model}  endpoint={endpoint}")
    print(f"[smoke] prompt_tokens={n_tokens} (target {TARGET_TOKENS_PER_PROMPT})")
    print(f"[smoke] two of these sum to {2*n_tokens} tokens — "
          f"exceeds max_num_batched_tokens cap, forces contention")
    print(f"[smoke] preempts_before={n_before}")

    async def fire(slo_ms: int, label: str) -> None:
        t0 = time.time()
        async with httpx.AsyncClient(timeout=180.0) as client:
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
        g = asyncio.create_task(fire(5000, "generous"))
        await asyncio.sleep(0.05)
        u = asyncio.create_task(fire(50,   "urgent"))
        await asyncio.gather(g, u)

    print(f"[smoke] firing two concurrent completions...")
    asyncio.run(run())

    n_after, tail = grep_prefill_log(prefill_log, 0)
    delta = n_after - n_before
    print()
    print(f"[smoke] preempts_after={n_after}  delta={delta}")
    if delta > 0:
        print(f"[smoke] PREEMPT INTENT fired ✓  ({delta} new events)")
    else:
        print(f"[smoke] No PREEMPT INTENT in {prefill_log} — check policy + waiting/running state")
    if tail:
        print("[smoke] last 5 preempt/stubborn lines:")
        for ln in tail:
            print(f"   {ln}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
