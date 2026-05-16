# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlowPrefill deploy config — single source of truth.

Reads deploy/config.json (or config.smoke.json when SMOKE=True) and
resolves it for $MODE (defaulting to "conservative").

Two consumers:

- Bash start scripts: invoke via `eval "$(python3 deploy/config.py)"`.
  stdout is shell-ready `export FOO=bar` lines. Banner + config summary
  go to stderr so eval ignores them.

- Python benchmark scripts: `from deploy.config import load`.
  Returns a dict with the resolved values.

The SMOKE flag below is the one knob you flip to switch between prod
(config.json: 70B / TP=4+4) and smoke (config.smoke.json: 8B / TP=1+1).
"""

import json
import os
import shlex
import sys
from pathlib import Path

# ─── Knobs you can flip without re-reading the rest of this file ──────────
SMOKE = False  # flip to True for smoke testing (8B / TP=1+1 / 2 GPUs)
DEFAULT_MODE = "conservative"
# ──────────────────────────────────────────────────────────────────────────

_VALID_MODES = ("control", "conservative", "aggressive")
_HERE = Path(__file__).resolve().parent


def resolve_mode() -> str:
    mode = os.environ.get("MODE", DEFAULT_MODE)
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown MODE={mode!r}. Valid: {' | '.join(_VALID_MODES)}"
        )
    return mode


def config_json_path() -> Path:
    return _HERE / ("config.smoke.json" if SMOKE else "config.json")


def load() -> dict:
    """Resolve config for the current $MODE.

    Returns a dict with these top-level keys:
      mode, smoke, config_file
      model, topology, ports, logs, env
      vllm_flags (assembled string, including --max-model-len etc.)

    Used by Python scripts. Bash gets the equivalent via the CLI.
    """
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError(
            "HF_TOKEN not in env. Set it via the RunPod template "
            "(or `export HF_TOKEN=...` for local testing)."
        )

    mode = resolve_mode()
    cfg_path = config_json_path()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open() as f:
        all_buckets = json.load(f)
    if mode not in all_buckets:
        raise KeyError(f"Bucket {mode!r} not in {cfg_path}")
    bucket = all_buckets[mode]

    # Assemble VLLM_FLAGS string from the discrete model keys.
    m = bucket["model"]
    def b(name: str, val: bool) -> str:
        return f"--{name}" if val else f"--no-{name}"
    vllm_flags = " ".join([
        b("enable-chunked-prefill", m["enable_chunked_prefill"]),
        b("async-scheduling",        m["async_scheduling"]),
        b("enable-prefix-caching",   m["enable_prefix_caching"]),
        f"--max-model-len {m['max_model_len']}",
        f"--max-num-batched-tokens {m['max_num_batched_tokens']}",
    ])

    return {
        "mode": mode,
        "smoke": SMOKE,
        "config_file": str(cfg_path),
        "model": bucket["model"],
        "topology": bucket["topology"],
        "ports": bucket["ports"],
        "logs": bucket["logs"],
        "env": bucket["env"],
        "vllm_flags": vllm_flags,
    }


def _emit_exports(cfg: dict) -> None:
    """Print shell `export` lines for the bash side. stdout only."""
    m = cfg["model"]
    t = cfg["topology"]
    p = cfg["ports"]
    l = cfg["logs"]

    # Anything that goes to stdout becomes part of the bash eval, so be
    # strict: only `export FOO=bar` lines, all values shell-quoted.
    def out(k: str, v) -> None:
        print(f"export {k}={shlex.quote(str(v))}")

    out("MODE",                   cfg["mode"])
    out("MODEL",                  m["name"])
    out("MAX_MODEL_LEN",          m["max_model_len"])
    out("MAX_NUM_BATCHED_TOKENS", m["max_num_batched_tokens"])
    out("ENABLE_CHUNKED_PREFILL", str(m["enable_chunked_prefill"]).lower())
    out("ASYNC_SCHEDULING",       str(m["async_scheduling"]).lower())
    out("ENABLE_PREFIX_CACHING",  str(m["enable_prefix_caching"]).lower())

    out("PREFILL_TP",             t["prefill_tp"])
    out("DECODE_TP",              t["decode_tp"])
    out("STANDALONE_TP",          t["prefill_tp"])
    out("PREFILL_GPUS",           t["prefill_gpus"])
    out("DECODE_GPUS",            t["decode_gpus"])
    out("STANDALONE_GPUS",        t["standalone_gpus"])
    out("PREFILL_GPU_MEM_UTIL",   t["prefill_gpu_mem_util"])
    out("DECODE_GPU_MEM_UTIL",    t["decode_gpu_mem_util"])

    out("PREFILL_PORT",           p["prefill"])
    out("DECODE_PORT",            p["decode"])
    out("STANDALONE_PORT",        p["standalone"])
    out("PROXY_HTTP_PORT",        p["proxy_http"])
    out("PROXY_ZMQ_PORT",         p["proxy_zmq"])
    out("PREFILL_KV_PORT",        p["prefill_kv"])
    out("DECODE_KV_PORT",         p["decode_kv"])
    out("PREFILL_NIXL_PORT",      p["prefill_nixl"])
    out("DECODE_NIXL_PORT",       p["decode_nixl"])

    out("PREFILL_LOG",            l["prefill"])
    out("DECODE_LOG",             l["decode"])
    out("PROXY_LOG",              l["proxy"])
    out("STANDALONE_LOG",         l["standalone"])

    for k, v in cfg["env"].items():
        out(k, v)

    out("VLLM_FLAGS",             cfg["vllm_flags"])


def _emit_banner(cfg: dict) -> None:
    """Print bold-bordered banner + summary to stderr."""
    # ANSI escapes — terminals render bold/color; piped logs see harmless
    # garnish + a still-readable asterisk border.
    BOLD = "\033[1m"; RESET = "\033[0m"
    RED  = "\033[31m"; GREEN = "\033[32m"
    if cfg["smoke"]:
        text  = "SMOKE MODE  —  small pod, 8B, TP=1+1"
        color = RED
    else:
        short = cfg["model"]["name"].rsplit("/", 1)[-1]
        text  = f"PROD MODE  —  benchmark, {short}"
        color = GREEN

    W = 72
    border = "*" * W
    pad = (W - len(text) - 2) // 2
    lpad = " " * pad
    rpad = " " * (W - 2 - pad - len(text))

    e = sys.stderr.write
    e("\n")
    e(f"{color}{BOLD}{border}{RESET}\n")
    e(f"{color}{BOLD}*{lpad}{text}{rpad}*{RESET}\n")
    e(f"{color}{BOLD}{border}{RESET}\n\n")

    t = cfg["topology"]
    rows = [
        ("config file", cfg["config_file"]),
        ("mode",        cfg["mode"]),
        ("model",       cfg["model"]["name"]),
        ("prefill",     f"TP={t['prefill_tp']} GPUs={t['prefill_gpus']} mem_util={t['prefill_gpu_mem_util']}"),
        ("decode",      f"TP={t['decode_tp']} GPUs={t['decode_gpus']} mem_util={t['decode_gpu_mem_util']}"),
        ("tokens",      f"len={cfg['model']['max_model_len']} batched={cfg['model']['max_num_batched_tokens']}"),
        ("vllm flags",  cfg["vllm_flags"]),
    ]
    if "FLOWPREFILL_POLICY" in cfg["env"]:
        rows.append(("fp policy", cfg["env"]["FLOWPREFILL_POLICY"]))
    for k, v in rows:
        e(f"  {k:<15} {v}\n")
    e("\n")


def main_cli() -> int:
    """Entry point for `python3 deploy/config.py`. Emits bash exports to
    stdout + banner to stderr."""
    try:
        cfg = load()
    except Exception as e:
        # Fail loud, but to stderr so eval doesn't swallow garbage onto bash.
        sys.stderr.write(f"ERROR: {e}\n")
        return 1

    # Banner only when SUPPRESS_BANNER unset (so repeated source calls in
    # the same shell session can quiet down by setting it).
    if not os.environ.get("FLOWPREFILL_BANNER_SHOWN"):
        _emit_banner(cfg)
        # Tell the bash side to set this on its own env after eval.
        print("export FLOWPREFILL_BANNER_SHOWN=1")

    _emit_exports(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
