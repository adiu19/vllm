#!/bin/bash
# Verify HuggingFace authentication before launching vLLM.
# Useful before pulling gated models (Llama 3.1 8B, Llama 3.3 70B, Mistral, etc.)
# Exits non-zero if HF_TOKEN is missing or invalid — fail fast rather than hang
# for 10 minutes waiting on a 401.
#
# Usage:
#   bash /opt/vllm-fork/runpod/verify_hf_auth.sh
#   bash /opt/vllm-fork/runpod/verify_hf_auth.sh meta-llama/Meta-Llama-3.1-70B-Instruct

set -e

# Source token if config has it (RunPod typically injects env vars directly)
if [ -f /tmp/config.sh ]; then
    source /tmp/config.sh
fi

# Fall back to saved CLI login if no env var set
if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
    echo "Using token from ~/.cache/huggingface/token"
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Gated models (Llama, Mistral, etc.) will hang on download."
    echo "Set it in RunPod's pod environment variables, or:"
    echo "  export HF_TOKEN=hf_YOUR_TOKEN_HERE"
    exit 1
fi

# Check token validity by calling /api/whoami-v2
RESPONSE=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer ${HF_TOKEN}" https://huggingface.co/api/whoami-v2)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n -1)

if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: HF auth failed (HTTP $HTTP_CODE)"
    echo "$BODY"
    exit 1
fi

USER=$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('name', 'unknown'))")
echo "HF auth OK — authenticated as: $USER"

# Optional: verify access to a specific gated model
MODEL="${1:-}"
if [ -n "$MODEL" ]; then
    echo "Verifying access to $MODEL..."
    MODEL_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        "https://huggingface.co/api/models/${MODEL}")
    if [ "$MODEL_CODE" != "200" ]; then
        echo "ERROR: Cannot access $MODEL (HTTP $MODEL_CODE)."
        echo "Likely you haven't accepted the model's license on HuggingFace yet."
        echo "Visit: https://huggingface.co/${MODEL} and click 'Agree and access'."
        exit 1
    fi
    echo "Model access OK: $MODEL"
fi
