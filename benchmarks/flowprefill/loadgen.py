# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlowPrefill benchmark load generator.

Issues Poisson-arrival completion requests through the FlowPrefill proxy
under a 2-tier mixed-SLO workload (urgent / generous) drawn from ShareGPT.
Writes one CSV row per request plus a sibling .meta.json. The CSV schema
is consumed by analyze.py.

Determinism: the per-trial RNG is seeded by `hash((master_seed, trial_id))`
so re-running the same (master_seed, trial_id) produces identical arrival
times, identical prompt picks, and identical SLO draws. The paired-trial
design hinges on this: at trial T we compare control / conservative /
aggressive against the SAME workload.

Per-request CSV columns (consumed by analyze.py):
  - t_client_send_ms, t_server_arrival_ms, t_prefill_done_ms,
    t_decode_first_byte_ms, t_decode_last_byte_ms : epoch milliseconds (int),
    -1 if missing.
  - tier : "urgent" | "generous"
  - slo_ms : SLO assigned to this request (int)
  - prompt_length : tokenizer's prompt token count
  - request_id : server-assigned cmpl-... id from the first SSE chunk
  - in_warmup : true|false (warmup rows are kept for inspection but
    analyze.py filters them out for the headline metrics)
  - bucket : nearest baseline-bucket the prompt was anchored to
  - error : "" if ok, otherwise a short tag

Usage:
  python benchmarks/flowprefill/loadgen.py \\
      --endpoint http://localhost:10001 \\
      --model meta-llama/Llama-3.3-70B-Instruct \\
      --rate 5 --warmup-s 30 --measure-s 300 \\
      --mode conservative --trial-id 0 --master-seed 42 \\
      --output-dir benchmarks/flowprefill/runs/test
"""

import argparse
import asyncio
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Use the same predictor (a, c) that slo_monitor uses at runtime. Single
# source of truth — calibrated from an offline profile_ttft.py run on
# Llama-3.3-70B-Instruct/A100/TP=4. Smoke runs on 8B/TP=1 reuse these
# constants; the resulting SLOs will be loose (8B is much faster than
# 70B) but smoke only validates pipeline correctness, not SLO calibration.
from vllm.v1.core.sched.slo_monitor import (
    TTFT_COEFF_A_MS_PER_TOKEN,
    TTFT_COEFF_C_MS,
)

# Pull the repo root onto sys.path so `from deploy.config import load`
# resolves regardless of how loadgen is invoked (`python loadgen.py`,
# `python -m benchmarks.flowprefill.loadgen`, etc.).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from deploy.config import load as _load_deploy_config  # noqa: E402


# Tier SLO bands as multipliers on the predicted baseline TTFT. From
# Flow Prefill/Benchmark Design.md, Step 2. SLO drawn uniformly within
# the band; one draw per request.
TIER_BANDS = {
    "urgent":   (1.0, 1.3),
    "generous": (3.0, 10.0),
}

# Completion-side defaults. max_tokens is intentionally small — TTFT is the
# headline metric, not decode throughput. On 6-GPU setups (TP=4 prefill +
# TP=2 decode) the decode KV pool is small (~4.6 concurrent requests at
# 8k context); short decodes keep us well under that ceiling. 2 tokens is
# enough to guarantee decode emits a visible response chunk; 1 is risky
# because vLLM's disagg accounting may not yield a visible decode token.
DEFAULT_MAX_TOKENS = 2

# Prompt-length window we draw from. 256 is the smallest bucket the
# profiler measured — using the predictor below this is extrapolation,
# and TTFT at very short lengths is dominated by kernel-launch fixed
# cost (priorities don't differentiate). 8000 is set by max_model_len
# (config: 8500) — longer prompts fail at the server.
MIN_PROMPT_TOKENS = 256
MAX_PROMPT_TOKENS = 8000

# Lazy import for the tokenizer (heavy).
_TOKENIZER = None


def get_tokenizer(model_name: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def nearest_bucket(prompt_length: int, bucket_keys: list[int]) -> int:
    """Snap a prompt length to the closest baseline bucket. Used only as
    a grouping label in CSV output; SLOs are computed via the continuous
    predictor, not via bucket lookup."""
    return min(bucket_keys, key=lambda b: abs(b - prompt_length))


# Bucket labels for the per-bucket CSV grouping (matches what the profiler
# measured). These are NOT used to compute SLO — that's done via the
# predictor (a*L + c).
BUCKET_LABELS = [256, 512, 1024, 2000, 4000, 8000]


def predicted_baseline_ms(prompt_length: int) -> float:
    """Predicted TTFT for a prompt of `prompt_length` tokens. Same model
    as slo_monitor uses for slack computation."""
    return TTFT_COEFF_A_MS_PER_TOKEN * prompt_length + TTFT_COEFF_C_MS


def load_sharegpt(path: Path) -> list[str]:
    """Load ShareGPT conversations; return the first-turn prompt of each
    entry with >=2 turns. No tokenizer work yet — that happens lazily as
    we pick prompts."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for entry in data:
        convs = entry.get("conversations", [])
        if len(convs) < 2:
            continue
        first = convs[0].get("value", "")
        if first:
            prompts.append(first)
    if not prompts:
        raise RuntimeError(f"No usable prompts found in {path}")
    return prompts


def build_request_schedule(
    *,
    master_seed: int,
    trial_id: int,
    rate: float,
    warmup_s: float,
    measure_s: float,
    tier_split: float,
    prompts_pool: list[str],
    tokenizer,
    min_prompt_tokens: int = MIN_PROMPT_TOKENS,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> list[dict]:
    """Pre-generate the full request schedule deterministically.

    Returns a list of dicts, each describing one request with its
    scheduled relative arrival time (seconds from trial start), tier,
    slo_ms, bucket (grouping label), prompt text, and prompt_length.
    """
    rng = np.random.default_rng(abs(hash((master_seed, trial_id))) % (2**32))
    duration = warmup_s + measure_s

    # Exponential inter-arrivals; cumulative sum gives absolute arrivals.
    # Over-allocate (2x) so we don't truncate the tail; trim by duration.
    n_target = int(rate * duration * 2 + 100)
    inter = rng.exponential(scale=1.0 / rate, size=n_target)
    arrivals = np.cumsum(inter)
    arrivals = arrivals[arrivals < duration]
    if len(arrivals) == 0:
        raise RuntimeError("Arrival schedule is empty — rate or duration too low.")

    # Deterministic prompt picks via the same RNG.
    shuffled_prompts = list(prompts_pool)
    py_rng = random.Random(int(rng.integers(0, 2**31)))
    py_rng.shuffle(shuffled_prompts)

    schedule: list[dict] = []
    prompt_cursor = 0
    skipped_oor = 0

    for t in arrivals:
        # Walk through prompts until we find one inside the [min, max]
        # token window. No per-request retry cap — walking the whole shuffled
        # corpus is bounded by len(prompts). The only fatal case is the entire
        # corpus exhausted without enough in-range matches.
        chosen = None
        chosen_len = None
        while prompt_cursor < len(shuffled_prompts):
            candidate = shuffled_prompts[prompt_cursor]
            prompt_cursor += 1
            ids = tokenizer(candidate, add_special_tokens=False).input_ids
            n_tok = len(ids)
            if min_prompt_tokens <= n_tok <= max_prompt_tokens:
                chosen = candidate
                chosen_len = n_tok
                break
            skipped_oor += 1
        if chosen is None:
            raise RuntimeError(
                f"Corpus exhausted before scheduling all {len(arrivals)} "
                f"requests (skipped {skipped_oor} out-of-range prompts in "
                f"window [{min_prompt_tokens}, {max_prompt_tokens}]). "
                f"Widen the window or use a larger corpus."
            )

        # SLO sized off the continuous predictor — no bucket snapping.
        baseline_ms = predicted_baseline_ms(chosen_len)

        is_urgent = rng.random() < tier_split
        tier = "urgent" if is_urgent else "generous"
        lo, hi = TIER_BANDS[tier]
        slo_ms = int(round(baseline_ms * rng.uniform(lo, hi)))

        schedule.append({
            "rel_arrival_s": float(t),
            "tier": tier,
            "slo_ms": slo_ms,
            "bucket": nearest_bucket(chosen_len, BUCKET_LABELS),
            "prompt": chosen,
            "prompt_length": chosen_len,
            "in_warmup": float(t) < warmup_s,
        })

    print(
        f"[loadgen] schedule built: n={len(schedule)} duration={duration}s "
        f"warmup={warmup_s}s rate={rate}/s tier_split={tier_split} "
        f"skipped_out_of_range_prompts={skipped_oor}",
        flush=True,
    )
    return schedule


async def fire_request(
    *,
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    item: dict,
    max_tokens: int,
    rows: list[dict],
    rows_lock: asyncio.Lock,
    t0_wall_ms: int,
) -> None:
    """Send one request, capture all five timestamps + request_id, append row."""
    payload = {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {
        "X-FlowPrefill-SLO-MS": str(item["slo_ms"]),
    }

    t_client_send_ms = int(time.time() * 1000)
    t_server_arrival_ms = -1
    t_prefill_done_ms = -1
    t_decode_first_byte_ms = -1
    t_decode_last_byte_ms = -1
    request_id = ""
    error = ""

    try:
        async with client.stream(
            "POST",
            f"{endpoint}/v1/completions",
            json=payload,
            headers=headers,
            # 70B prefill on long prompts can take 1-2s by itself; under
            # contention with deep queues a request can wait several
            # seconds before its prefill even starts. 5min total guard.
            timeout=httpx.Timeout(300.0, connect=10.0),
        ) as resp:
            # Proxy emits these headers as soon as the prefill is done.
            sa = resp.headers.get("X-Server-Arrival-Time-Ms")
            pd = resp.headers.get("X-Prefill-Done-Time-Ms")
            if sa is not None:
                try:
                    t_server_arrival_ms = int(sa)
                except ValueError:
                    pass
            if pd is not None:
                try:
                    t_prefill_done_ms = int(pd)
                except ValueError:
                    pass

            if resp.status_code != 200:
                error = f"http_{resp.status_code}"
                # Drain so the connection can be reused.
                async for _ in resp.aiter_text():
                    pass
            else:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload_str = line[len("data: "):]
                    if payload_str.strip() == "[DONE]":
                        continue
                    if t_decode_first_byte_ms < 0:
                        t_decode_first_byte_ms = int(time.time() * 1000)
                        try:
                            chunk = json.loads(payload_str)
                            request_id = chunk.get("id", "") or ""
                        except json.JSONDecodeError:
                            pass
                t_decode_last_byte_ms = int(time.time() * 1000)
    except (httpx.ReadTimeout, httpx.ConnectTimeout):
        error = "timeout"
    except httpx.HTTPError as e:
        error = f"http_error:{type(e).__name__}"
    except Exception as e:
        error = f"exception:{type(e).__name__}"

    row = {
        "t_client_send_ms": t_client_send_ms,
        "t_server_arrival_ms": t_server_arrival_ms,
        "t_prefill_done_ms": t_prefill_done_ms,
        "t_decode_first_byte_ms": t_decode_first_byte_ms,
        "t_decode_last_byte_ms": t_decode_last_byte_ms,
        "tier": item["tier"],
        "slo_ms": item["slo_ms"],
        "prompt_length": item["prompt_length"],
        "request_id": request_id,
        "in_warmup": item["in_warmup"],
        "bucket": item["bucket"],
        "error": error,
    }
    async with rows_lock:
        rows.append(row)


async def drive(args, schedule: list[dict]) -> list[dict]:
    """Schedule + dispatch all requests on the asyncio loop, wait for all."""
    rows: list[dict] = []
    rows_lock = asyncio.Lock()
    t0_wall_ms = int(time.time() * 1000)
    t0_mono = time.monotonic()

    limits = httpx.Limits(max_keepalive_connections=128, max_connections=512)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        for item in schedule:
            # Sleep until this request's absolute arrival time relative to
            # the start of the trial. Using monotonic clock here so jitter
            # in wall-clock doesn't drift the schedule.
            sleep_for = item["rel_arrival_s"] - (time.monotonic() - t0_mono)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            task = asyncio.create_task(
                fire_request(
                    client=client,
                    endpoint=args.endpoint,
                    model=args.model,
                    item=item,
                    max_tokens=args.max_tokens,
                    rows=rows,
                    rows_lock=rows_lock,
                    t0_wall_ms=t0_wall_ms,
                )
            )
            tasks.append(task)

        # Wait for all in-flight requests to finish their decodes.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return rows


def git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _resolved_env_snapshot(deploy_cfg: dict) -> dict[str, str]:
    """Record what the SERVER was actually configured with. Prefer the
    values resolved by deploy.config.load() (the same SoT the server
    uses); fall back to os.environ for keys not in the bucket.

    Why not just read os.environ: loadgen.main() doesn't mutate the
    process env after load() — env_snapshot would say "<unset>" even
    though the server sees MODE / FLOWPREFILL_POLICY correctly.
    """
    keys = [
        "MODE", "FLOWPREFILL_ENABLED", "FLOWPREFILL_POLICY",
        "FLOWPREFILL_STUBBORN_LAYER_FRAC", "FLOWPREFILL_SLO_BASE_MS",
        "HF_HOME",
    ]
    bucket_env = deploy_cfg.get("env", {})
    snapshot: dict[str, str] = {}
    for k in keys:
        if k == "MODE":
            snapshot[k] = deploy_cfg.get("mode", os.environ.get(k, "<unset>"))
        elif k in bucket_env:
            snapshot[k] = str(bucket_env[k])
        else:
            snapshot[k] = os.environ.get(k, "<unset>")
    return snapshot


def write_outputs(
    args, rows: list[dict], schedule: list[dict], deploy_cfg: dict
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"trial_{args.trial_id}_policy_{args.mode}.csv"
    meta_path = output_dir / f"trial_{args.trial_id}_policy_{args.mode}.meta.json"

    fieldnames = [
        "t_client_send_ms", "t_server_arrival_ms", "t_prefill_done_ms",
        "t_decode_first_byte_ms", "t_decode_last_byte_ms",
        "tier", "slo_ms", "prompt_length", "request_id",
        "in_warmup", "bucket", "error",
    ]
    # Sort by t_client_send_ms for readability — analyze.py doesn't depend on order.
    rows_sorted = sorted(rows, key=lambda r: r["t_client_send_ms"])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_sorted)

    meta = {
        "master_seed": args.master_seed,
        "trial_id": args.trial_id,
        "mode": args.mode,
        "endpoint": args.endpoint,
        "model": args.model,
        "rate": args.rate,
        "warmup_s": args.warmup_s,
        "measure_s": args.measure_s,
        "tier_split": args.tier_split,
        "max_tokens": args.max_tokens,
        "n_requests_scheduled": len(schedule),
        "n_requests_recorded": len(rows),
        "n_errors": sum(1 for r in rows if r["error"]),
        "git_sha": git_short_sha(),
        "started_at_ms": min((r["t_client_send_ms"] for r in rows), default=0),
        "ended_at_ms": max((r["t_decode_last_byte_ms"] for r in rows), default=0),
        "env_snapshot": _resolved_env_snapshot(deploy_cfg),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"[loadgen] wrote {csv_path} ({len(rows_sorted)} rows)")
    print(f"[loadgen] wrote {meta_path}")


def main() -> int:
    # MODE must be in env, same value used to start the prefill server.
    # We deliberately do NOT default this — silent defaults caused trial_6
    # to get labeled "conservative" while the server was running aggressive.
    # Requiring it explicit means the operator (or the benchmark orchestrator)
    # passes the same MODE that launched the server. Catches mismatch loud.
    env_mode = os.environ.get("MODE")
    if env_mode is None:
        print(
            "ERROR: MODE env var must be set, with the SAME value used when\n"
            "starting the prefill node (start_prefill_nixl.sh). Example:\n"
            "  MODE=conservative python3 benchmarks/flowprefill/loadgen.py ...\n"
            "Valid: control | conservative | aggressive",
            file=sys.stderr,
        )
        return 1

    # Resolve deploy config (uses MODE from env). Endpoint + model default
    # from this.
    deploy_cfg = _load_deploy_config()
    default_model = deploy_cfg["model"]["name"]
    default_proxy_port = deploy_cfg["ports"]["proxy_http"]
    default_endpoint = f"http://localhost:{default_proxy_port}"

    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=default_endpoint,
                    help="Proxy endpoint (HTTP, no trailing slash).")
    ap.add_argument("--model", default=default_model,
                    help="Defaults to $MODEL from deploy/config.py.")
    ap.add_argument("--dataset",
                    default="/workspace/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json")
    ap.add_argument("--rate", type=float, required=True,
                    help="Mean arrival rate (req/s).")
    ap.add_argument("--warmup-s", type=float, default=30.0)
    ap.add_argument("--measure-s", type=float, default=300.0)
    ap.add_argument("--tier-split", type=float, default=0.2,
                    help="Fraction urgent (the rest are generous).")
    # --mode pulled from $MODE; no default to argparse. We validate the
    # CLI value (if supplied) matches the env value below.
    ap.add_argument("--mode", default=env_mode,
                    choices=["control", "conservative", "aggressive"],
                    help="Pinned to $MODE; pass explicitly only if you "
                         "really want to override (will fail if mismatched).")
    ap.add_argument("--trial-id", type=int, required=True)
    ap.add_argument("--master-seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--min-prompt-tokens", type=int, default=MIN_PROMPT_TOKENS,
                    help="Lower bound on prompt length (tokens). Bump up to "
                         "force prefill contention on fast models — pairs of "
                         "prompts whose sum exceeds max_num_batched_tokens "
                         "can't co-batch.")
    ap.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS,
                    help="Upper bound on prompt length (tokens).")
    args = ap.parse_args()

    # Reject CLI/env mismatch loudly — exactly the failure mode of trial_6
    # (server aggressive, loadgen conservative). Forces operator to think
    # twice before overriding.
    if args.mode != env_mode:
        print(
            f"ERROR: --mode={args.mode!r} but $MODE={env_mode!r}. The CSV "
            "label must match the server's actual policy. Restart the "
            "prefill node with the intended MODE, or drop --mode and trust "
            "the env.",
            file=sys.stderr,
        )
        return 1

    if not args.model:
        print("ERROR: --model not supplied and $MODEL not in env. "
              "Did config.py resolve correctly?", file=sys.stderr)
        return 1
    if not args.mode:
        print("ERROR: --mode not supplied and $MODE not in env.",
              file=sys.stderr)
        return 1

    print(f"[loadgen] resolved model={args.model} mode={args.mode} "
          f"endpoint={args.endpoint}")
    print(f"[loadgen] predictor: a={TTFT_COEFF_A_MS_PER_TOKEN} "
          f"c={TTFT_COEFF_C_MS} (range {MIN_PROMPT_TOKENS}-{MAX_PROMPT_TOKENS} tokens)")

    # 1. Load corpus + tokenizer.
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
        return 1
    prompts_pool = load_sharegpt(dataset_path)
    print(f"[loadgen] loaded {len(prompts_pool)} prompts from {dataset_path}")

    tokenizer = get_tokenizer(args.model)

    # 2. Build schedule (SLO sized via predictor, not via baseline lookup).
    schedule = build_request_schedule(
        master_seed=args.master_seed,
        trial_id=args.trial_id,
        rate=args.rate,
        warmup_s=args.warmup_s,
        measure_s=args.measure_s,
        tier_split=args.tier_split,
        prompts_pool=prompts_pool,
        tokenizer=tokenizer,
        min_prompt_tokens=args.min_prompt_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    # 3. Run.
    print(
        f"[loadgen] firing {len(schedule)} requests at rate={args.rate}/s "
        f"(mode={args.mode} trial={args.trial_id} seed={args.master_seed})",
        flush=True,
    )
    rows = asyncio.run(drive(args, schedule))

    # 4. Write outputs.
    write_outputs(args, rows, schedule, deploy_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
