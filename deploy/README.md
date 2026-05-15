# deploy/

Pod image + service scripts for FlowPrefill experiments. Currently targets
RunPod but is provider-agnostic (any host that can run the Docker image works).

## TL;DR

```bash
# Pod env vars (set in the RunPod dashboard):
#   HF_TOKEN=hf_xxx
#   MODE=treatment        (or control)

# Pod starts → ENTRYPOINT runs entrypoint.sh → init.sh
# (validates env, git-syncs to the branch experiments.sh maps MODE to,
#  checks HF auth + GPU topology)

# SSH in, then:
bash /opt/vllm-fork/deploy/start_prefill_nixl.sh
# or
bash /opt/vllm-fork/deploy/start_standalone.sh
```

## File map

| File | Role |
|---|---|
| `Dockerfile` | Image definition. Rebuilt only when dependencies change. |
| `entrypoint.sh` | Container entrypoint. Calls `init.sh` and execs the CMD. Baked into the image. |
| `init.sh` | Pre-flight: validates env, git-syncs, HF auth + model access, GPU checks. Lives in git, pulled fresh on every container start. |
| `config.sh` | Branch-specific config: model, ports, GPUs, NCCL env, `EXTRA_VLLM_FLAGS`. Sourced by `init.sh` and every `start_*.sh`. |
| `experiments.sh` | `MODE → BRANCH` mapping. Single file controls which branch each mode checks out. |
| `start_prefill_nixl.sh` | Launch prefill with NixlConnector (kv_both). FlowPrefill prefill node. |
| `start_decode_nixl.sh` | Launch decode with NixlConnector (kv_both). FlowPrefill decode node. |
| `start_proxy_nixl.sh` | Launches `benchmarks/flowprefill/proxy.py` — request router with timing headers for benchmarks. |
| `start_standalone.sh` | Single-instance vLLM, no P/D, no KV connector. Useful for code sanity tests. |
| `kill_gpu.sh` | Kill all vLLM/proxy processes and wait for GPU memory to clear. |
| `verify_hf_auth.sh` | Standalone HF auth + model-access check (also called by `init.sh`). |
| `check_network.sh` | RunPod-specific network checks. |

## Required pod env vars

Set these in the pod environment (RunPod dashboard → Environment Variables):

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | HuggingFace token. Required for gated models (Llama, Mistral). |
| `MODE` | `control` or `treatment`. Selects branch (via `experiments.sh`). |

Optional:

| Variable | Effect |
|---|---|
| `SKIP_INIT=1` | Bypass `init.sh` on container start. Only used when debugging init.sh itself. |

## Mode semantics

| MODE | Branch checked out | Flags |
|---|---|---|
| `control` | `control` | Stock vLLM defaults (chunked prefill ON, async sched ON). Baseline for benchmarks. |
| `treatment` | `dev-treatment` (currently; flip to `treatment` after merge) | FlowPrefill flags from that branch's `config.sh`. |

`init.sh` reads `experiments.sh` to resolve `$BRANCH` from `$MODE`, then
git-checks-out and pulls that branch. Changing `MODE` and restarting the pod
is sufficient to flip experiments — no image rebuild needed.

Each branch's own `deploy/config.sh` defines the flag bundle that branch runs
with (`EXTRA_VLLM_FLAGS`). The branch is the config.

## Building the image

```bash
# From the repo root
docker buildx build --platform linux/amd64 \
    -t adiu19/vllm-flowprefill:latest \
    -f deploy/Dockerfile --push .
```

Rebuild only when:
- Dependencies change (`pip install` list, apt packages)
- `entrypoint.sh` changes (rare)
- vLLM precompiled wheel commit is bumped

For everything else (init.sh, config.sh, experiments.sh, start scripts, model
code in the fork), we push to the branch and `init.sh` pulls on the next
container start.

## Day-to-day workflow

```bash
# Edit code locally on dev-treatment (or treatment)
vim vllm/v1/core/sched/preempt_check.py
git commit -am "tweak preempt check"
git push origin dev-treatment

# On the running pod (same SSH session)
cd /opt/vllm-fork
bash deploy/init.sh         # pulls latest, re-validates
bash deploy/kill_gpu.sh     # if a previous service is still running
bash deploy/start_prefill_nixl.sh
```

For a clean run on a new pod: spin up the pod with `HF_TOKEN` + `MODE` env
vars set, SSH in, and `start_*` scripts work immediately (init.sh already ran
via ENTRYPOINT at container start).

## Troubleshooting

**Pod stuck "Waiting for engine core proc to start"** → 99% chance the model
is downloading from HF. Llama 3.1 8B is ~15 GiB. Llama 70B is ~140 GiB. Check
`du -sh $HF_HOME` (default `/workspace/hf_cache`).

**Init fails on "no NVLink detected"** → we're on a PCIe-only host (e.g.,
A40, A100 PCIe variant). Spin up a new pod with A100 SXM, H100 SXM, etc.
Verify with `nvidia-smi topo -m` (look for `NV#` entries between GPUs).

**Init fails on "GPU has orphan memory"** → previous run crashed without
cleanup. Run `bash deploy/kill_gpu.sh`. If memory is still stuck after the
kill, it's a driver-level orphan — only a pod restart from RunPod's dashboard
clears it.

**Init fails on "HF auth failed"** → confirm `HF_TOKEN` is set correctly. For
gated models, also visit `https://huggingface.co/$MODEL` and click
"Agree and access" once per HF account.

**vLLM hangs on NCCL bootstrap with TP > 1** → topology issue. If the
topology shows `PIX` / `PXB` / `SYS` between GPUs (no `NV#`), NCCL P2P fails
on PCIe in many container setups. Workaround: `export NCCL_P2P_DISABLE=1` in
`config.sh`, or get an SXM-variant pod.

## See also

- `Decisions.md` in the FlowPrefill vault — design decisions and trade-offs
- `Unknowns.md` — open questions and "read later" topics
