FROM vllm/vllm-openai:latest

# System tools needed by runpod scripts:
#   iproute2 → ss (check open ports)
#   psmisc   → fuser (release ports in kill_gpu.sh)
RUN apt-get update && apt-get install -y \
    git \
    iproute2 \
    psmisc \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/adiu19/vllm /opt/vllm-fork && \
    sed -i 's/^license = "Apache-2.0"/license = {text = "Apache-2.0"}/' /opt/vllm-fork/pyproject.toml && \
    sed -i '/^license-files/d' /opt/vllm-fork/pyproject.toml

RUN VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_COMMIT=bc150f50299199599673614f80d12a196f377655 \
    python3 -m pip install -e /opt/vllm-fork --no-build-isolation

# Disaggregated prefill dependencies:
#   msgpack  → P2pNcclConnector ZMQ messages + proxy.py
#   quart    → proxy.py HTTP server (P2pNcclConnector path)
#   nixl     → NixlConnector KV transfer (CUDA 13 build)
RUN pip3 install \
    msgpack \
    quart \
    "nixl[cu13]>=0.7.1,<=0.10.1"

WORKDIR /opt/vllm-fork

ENTRYPOINT []
CMD ["sleep", "infinity"]
