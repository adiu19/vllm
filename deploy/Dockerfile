# Pod image for the FlowPrefill / vLLM-fork experiments.
# Built once; the actual experiment code (deploy/ scripts, vLLM source) is
# pulled fresh from this repo by deploy/init.sh on every container start.
#
# Image lifecycle:
#   1. `docker build` against this file (rare — only on dep changes)
#   2. Push to registry
#   3. Spin a pod referencing the image
#   4. Container starts → ENTRYPOINT → entrypoint.sh → init.sh (git pull, env
#      validation, HF auth check, GPU topo/health check)
#   5. SSH in or `docker exec` → run deploy/start_*.sh

FROM vllm/vllm-openai:latest

# System tools:
#   git, vim                         → dev essentials
#   iputils-ping, lsof, netcat-openbsd, iproute2, net-tools, dnsutils
#                                    → networking debug (ss, ping, nc, netstat)
#   psmisc                           → fuser, pstree (used by kill_gpu.sh)
#   jq                               → JSON parsing for curl outputs
RUN apt-get update && apt-get install -y \
    git \
    vim \
    iputils-ping \
    lsof \
    netcat-openbsd \
    iproute2 \
    net-tools \
    dnsutils \
    psmisc \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Python tooling:
#   msgpack, quart, pyzmq            → P/D proxy + P2pNcclConnector deps
#   nixl[cu13]                       → NixlConnector KV transfer
#   py-spy                           → live-dump python stacks of stuck workers
RUN pip3 install \
    msgpack \
    quart \
    pyzmq \
    py-spy \
    "nixl[cu13]>=0.7.1,<=0.10.1"

# Clone the fork. init.sh pulls the FlowPrefill working branch on every
# container start (single-branch model — policy is a runtime knob in
# deploy/config.json, not a branch).
RUN git clone https://github.com/adiu19/vllm /opt/vllm-fork && \
    sed -i 's/^license = "Apache-2.0"/license = {text = "Apache-2.0"}/' /opt/vllm-fork/pyproject.toml && \
    sed -i '/^license-files/d' /opt/vllm-fork/pyproject.toml

# Install vLLM in editable mode so subsequent git pulls take effect on
# next Python import without a reinstall.
RUN VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=bc150f50299199599673614f80d12a196f377655 \
    python3 -m pip install -e /opt/vllm-fork --no-build-isolation

# Entrypoint wrapper. Thin glue that delegates to deploy/init.sh. Baked into
# the image; rarely changes. The actual init logic lives in deploy/init.sh
# which is pulled fresh on every container start.
# COPY path is relative to the build context — run docker build from the
# repo root with `-f deploy/Dockerfile`.
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /opt/vllm-fork

# Required pod-level env vars (NOT baked into image; set via RunPod dashboard
# or `docker run -e`):
#   HF_TOKEN  → HuggingFace token; required for gated models (Llama, Mistral)
#
# Optional:
#   MODE      → control | conservative | aggressive ; selects policy bucket
#               from deploy/config.json. Defaults to "conservative" if unset.
#   SKIP_INIT → set to "1" to bypass init.sh on startup (useful when iterating
#               on init.sh itself or debugging a stuck pod)

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
