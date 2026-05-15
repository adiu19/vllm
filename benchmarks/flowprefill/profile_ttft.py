# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlowPrefill TTFT profiler.

Measures prefill-only TTFT across prompt-length buckets and produces two
artifacts from the same data:

  - baselines.json: per-bucket median TTFT, used by the load gen to anchor
    per-tier SLO bands at benchmark time.
  - predictor_coeffs.json: (a, c) from a linear regression on the same
    data, plus R². Used to refit slo_monitor.py constants for this
    (model, hardware, TP) combination.

Pre-requisites (otherwise the profiler fails loud at pre-flight):

  - Prefill + decode + proxy stack up.
  - Proxy is benchmarks/flowprefill/proxy.py (relays timing headers).
  - vLLM build includes the arrival_time_ms response field.

See:
  - FlowPrefill - Decisions.md "TTFT profiling pass: one run, two outputs"
  - Benchmark Design.md Step 4

Usage:
  python benchmarks/flowprefill/profile_ttft.py \
    --endpoint http://localhost:10001 \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --buckets 500,1000,2000,4000 \
    --n-runs 10 \
    --warmup-runs 2 \
    --output-dir benchmarks/flowprefill/results/ \
    --tp 4
"""

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import numpy as np


# Vocabulary used to build random prefix tokens. Each word tokenizes to
# ~one token in Llama tokenizers, so an 8-word prefix yields ~8 random
# tokens at the front of every prompt — enough to defeat any prefix-cache
# lookup, regardless of how many tokens the cache keys on.
_PREFIX_VOCAB = [
    "word", "cat", "dog", "sky", "run", "open", "fire", "tree",
    "blue", "code", "rain", "moon", "song", "land", "lake",
    "bird", "wind", "snow", "salt", "wave",
]


def make_prompt(target_length: int, run_id: int) -> str:
    """Build a synthetic prompt approximately `target_length` tokens long.

    Structure: random integer suffix + 8 random vocabulary words + " word"
    padding to hit the target. The random prefix defeats prefix caching;
    the identical body is content-neutral (matmul cost depends on shape,
    not values). True tokenized length is read back from the response's
    `usage.prompt_tokens` and used as the regression x-axis.
    """
    rand_int = random.randint(0, 10**9)
    prefix_words = " ".join(random.choices(_PREFIX_VOCAB, k=8))
    prefix = f"prof_{run_id}_{rand_int} {prefix_words}"
    # Approximate body length. Tokenizer overhead (BOS, prefix words,
    # special tokens) eats ~30 tokens; the actual count gets read back
    # from response.usage.prompt_tokens and used for the regression.
    body_words = max(target_length - 30, 0)
    body = " word" * body_words
    return prefix + body


def send_request(
    session: requests.Session,
    endpoint: str,
    model: str,
    prompt: str,
    *,
    require_timing_headers: bool = True,
) -> tuple[int, float]:
    """Send one completion request, return (true_prompt_tokens, ttft_ms).

    `ttft_ms` is computed from the proxy-emitted timing headers:
        ttft_ms = X-Prefill-Done-Time-Ms - X-Server-Arrival-Time-Ms

    Raises RuntimeError if `require_timing_headers` is set and either
    header is missing — the profiler depends on them being honest.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "stream": False,
    }
    response = session.post(
        f"{endpoint}/v1/completions", json=payload, timeout=300.0
    )
    response.raise_for_status()

    body = response.json()
    true_prompt_tokens = body["usage"]["prompt_tokens"]

    # requests.Response.headers is case-insensitive dict-like.
    arrival_ms_str = response.headers.get("X-Server-Arrival-Time-Ms")
    prefill_done_ms_str = response.headers.get("X-Prefill-Done-Time-Ms")

    if arrival_ms_str is None or prefill_done_ms_str is None:
        if require_timing_headers:
            raise RuntimeError(
                "Missing timing headers in response. Expected "
                "X-Server-Arrival-Time-Ms and X-Prefill-Done-Time-Ms. "
                "Verify the request is going through "
                "benchmarks/flowprefill/proxy.py (not the upstream toy "
                "proxy) and that vLLM was built with the FlowPrefill "
                "arrival_time_ms response field."
            )
        return true_prompt_tokens, float("nan")

    arrival_ms = int(arrival_ms_str)
    prefill_done_ms = int(prefill_done_ms_str)
    ttft_ms = float(prefill_done_ms - arrival_ms)
    return true_prompt_tokens, ttft_ms


def preflight(session: requests.Session, endpoint: str, model: str) -> None:
    """Verify the stack is wired before launching the measurement loop.

    Two checks:
      1. Proxy /healthcheck responds.
      2. One throwaway request returns both timing headers.

    Either failure aborts the profiler — silently falling back to
    client-side timing would corrupt the SLO baseline downstream.
    """
    print("[preflight] checking proxy /healthcheck ...")
    r = session.get(f"{endpoint}/healthcheck", timeout=10.0)
    r.raise_for_status()
    print(f"[preflight] /healthcheck OK: {r.json()}")

    print("[preflight] sending one throwaway request to verify timing headers ...")
    prompt = make_prompt(target_length=100, run_id=-1)
    true_len, ttft = send_request(session, endpoint, model, prompt)
    print(f"[preflight] OK — tokens={true_len} ttft={ttft:.1f}ms")


def profile_bucket(
    session: requests.Session,
    endpoint: str,
    model: str,
    target_length: int,
    n_runs: int,
    warmup_runs: int,
) -> list[tuple[int, float]]:
    """Profile one bucket. Returns the measurement-phase (length, ttft_ms)
    pairs; warmup pairs are discarded."""
    for w in range(warmup_runs):
        prompt = make_prompt(target_length, run_id=-(w + 1))
        try:
            _ = send_request(session, endpoint, model, prompt)
        except Exception as e:
            print(
                f"[bucket {target_length}] warmup {w + 1}/{warmup_runs} failed: {e}"
            )
            raise
        print(f"[bucket {target_length}] warmup {w + 1}/{warmup_runs} done")

    results: list[tuple[int, float]] = []
    for i in range(n_runs):
        prompt = make_prompt(target_length, run_id=i)
        ttft = None
        true_len = None
        for attempt in range(3):
            try:
                true_len, ttft = send_request(session, endpoint, model, prompt)
                break
            except requests.RequestException as e:
                if attempt == 2:
                    print(
                        f"[bucket {target_length}] run {i + 1} failed "
                        f"after 3 attempts: {e}"
                    )
                    raise
                print(
                    f"[bucket {target_length}] run {i + 1} attempt "
                    f"{attempt + 1} failed: {e}, retrying..."
                )
                time.sleep(0.5)
        assert ttft is not None and true_len is not None
        results.append((true_len, ttft))
        print(
            f"[bucket {target_length}, run {i + 1}/{n_runs}] "
            f"tokens={true_len} ttft={ttft:.1f}ms"
        )

    return results


def aggregate_bucket(target_length: int, results: list[tuple[int, float]]) -> dict:
    lengths = [r[0] for r in results]
    ttfts = [r[1] for r in results]
    # statistics.quantiles needs n >= 4 samples for n=4. We default to
    # n_runs=10 so this is fine, but guard for tiny --n-runs values.
    if len(ttfts) >= 4:
        q = statistics.quantiles(ttfts, n=4)
        p25, p75 = q[0], q[2]
    else:
        p25 = min(ttfts)
        p75 = max(ttfts)
    return {
        "target_length": target_length,
        "actual_length_median": int(statistics.median(lengths)),
        "ttft_ms_median": round(statistics.median(ttfts), 2),
        "ttft_ms_min": round(min(ttfts), 2),
        "ttft_ms_max": round(max(ttfts), 2),
        "ttft_ms_p25": round(p25, 2),
        "ttft_ms_p75": round(p75, 2),
        "n_samples": len(ttfts),
    }


def fit_predictor(buckets_summary: list[dict]) -> dict:
    """Fit predicted_TTFT = a * num_tokens + c via linear regression.

    Regression uses per-bucket medians (one point per bucket). R² is
    computed from the residuals on those points.
    """
    lengths = np.array(
        [b["actual_length_median"] for b in buckets_summary], dtype=float
    )
    ttfts = np.array([b["ttft_ms_median"] for b in buckets_summary], dtype=float)

    # numpy.polyfit returns coefficients in decreasing order of degree.
    # For degree=1 → [slope, intercept].
    a, c = np.polyfit(lengths, ttfts, 1)

    predicted = a * lengths + c
    ss_res = float(np.sum((ttfts - predicted) ** 2))
    ss_tot = float(np.sum((ttfts - np.mean(ttfts)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return {
        "a_ms_per_token": round(float(a), 6),
        "c_ms": round(float(c), 3),
        "r_squared": round(float(r_squared), 4),
        "fit_points": [
            {"length": int(lengths[i]), "ttft_ms": float(ttfts[i])}
            for i in range(len(lengths))
        ],
    }


def _load_deploy_cfg() -> dict | None:
    """Import deploy.config.load() to resolve --model + --tp defaults from
    the same SoT the server uses. Returns None on import failure (allows
    --model + --tp to fall back to required CLI args)."""
    import sys as _sys
    repo_root = Path(__file__).resolve().parent.parent.parent
    if str(repo_root) not in _sys.path:
        _sys.path.insert(0, str(repo_root))
    try:
        from deploy.config import load as _load
        return _load()
    except Exception:
        return None


def main() -> int:
    _cfg = _load_deploy_cfg()
    if _cfg is not None:
        _default_model = _cfg["model"]["name"]
        _default_tp = _cfg["topology"]["prefill_tp"]
        _default_endpoint = f"http://localhost:{_cfg['ports']['proxy_http']}"
    else:
        _default_model = ""
        _default_tp = None
        _default_endpoint = "http://localhost:10001"

    parser = argparse.ArgumentParser(
        description=(
            "FlowPrefill TTFT profiler — measures per-bucket prefill TTFT and "
            "fits predictor coefficients for slo_monitor.py."
        )
    )
    parser.add_argument(
        "--endpoint",
        default=_default_endpoint,
        help="Proxy URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=_default_model,
        help="Defaults to $MODEL from deploy/config.sh.",
    )
    parser.add_argument(
        "--buckets",
        default="500,1000,2000,4000",
        help="Comma-separated prompt-length targets in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Measurement runs per bucket (default: %(default)d)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Warmup runs per bucket, discarded (default: %(default)d)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/flowprefill/results",
        help="Directory for baselines.json + predictor_coeffs.json",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=_default_tp,
        help="Tensor parallel size (defaults to $PREFILL_TP) — metadata only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt generation (default: %(default)d)",
    )
    args = parser.parse_args()

    if not args.model:
        print("ERROR: --model not supplied and $MODEL not in env. "
              "Did config.sh source correctly?", file=sys.stderr)
        return 1

    random.seed(args.seed)
    buckets = [int(b.strip()) for b in args.buckets.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== FlowPrefill TTFT profiler ===")
    print(f"endpoint:    {args.endpoint}")
    print(f"model:       {args.model}")
    print(f"tp:          {args.tp}")
    print(f"buckets:     {buckets}")
    print(f"n_runs:      {args.n_runs} (+ {args.warmup_runs} warmup)")
    print(f"output_dir:  {output_dir}")
    print(f"seed:        {args.seed}")
    print()

    with requests.Session() as session:
        preflight(session, args.endpoint, args.model)
        print()

        all_results: dict[int, list[tuple[int, float]]] = {}
        for bucket in buckets:
            print(f"--- bucket: target_length={bucket} ---")
            results = profile_bucket(
                session,
                args.endpoint,
                args.model,
                bucket,
                args.n_runs,
                args.warmup_runs,
            )
            all_results[bucket] = results
            print()

    buckets_summary = [
        aggregate_bucket(b, all_results[b]) for b in buckets
    ]
    baselines_map = {str(b["target_length"]): b["ttft_ms_median"] for b in buckets_summary}
    predictor = fit_predictor(buckets_summary)

    measured_at = datetime.now(timezone.utc).isoformat()
    common_meta = {
        "model": args.model,
        "tp": args.tp,
        "endpoint": args.endpoint,
        "measured_at": measured_at,
    }

    baselines_doc = {
        **common_meta,
        "n_runs_per_bucket": args.n_runs,
        "warmup_runs_per_bucket": args.warmup_runs,
        "buckets": buckets_summary,
        # Flat lookup table the load gen reads directly.
        "baselines_ms": baselines_map,
    }

    predictor_doc = {
        **common_meta,
        **predictor,
    }

    baselines_path = output_dir / "baselines.json"
    predictor_path = output_dir / "predictor_coeffs.json"

    with open(baselines_path, "w") as f:
        json.dump(baselines_doc, f, indent=2)
    with open(predictor_path, "w") as f:
        json.dump(predictor_doc, f, indent=2)

    print("=== Results ===")
    print(
        f"{'Bucket':<10}{'Tokens(med)':<14}{'TTFT(med)':<14}{'TTFT(p25 - p75)':<20}"
    )
    for b in buckets_summary:
        ttft_range = f"{b['ttft_ms_p25']:.1f} - {b['ttft_ms_p75']:.1f}"
        print(
            f"{b['target_length']:<10}{b['actual_length_median']:<14}"
            f"{b['ttft_ms_median']:<14.1f}{ttft_range:<20}"
        )

    print()
    a = predictor["a_ms_per_token"]
    c = predictor["c_ms"]
    r2 = predictor["r_squared"]
    print(f"Predictor fit: TTFT_ms = {a:.4f} * tokens + {c:.2f}   (R²={r2:.4f})")
    if r2 < 0.95:
        print(
            f"⚠  R² = {r2:.4f} is below 0.95 — linear model may be a poor fit. "
            f"Inspect per-bucket data for non-linearity at endpoints."
        )

    print()
    print(f"Wrote: {baselines_path}")
    print(f"Wrote: {predictor_path}")
    print()
    print("To apply the refit coefficients to slo_monitor.py, edit:")
    print(f"  TTFT_COEFF_A_MS_PER_TOKEN = {a}")
    print(f"  TTFT_COEFF_C_MS           = {c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
