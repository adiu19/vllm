#!/bin/bash
# Container ENTRYPOINT. Baked into the image at build time, rarely changes.
# Delegates to deploy/init.sh (pulled fresh from git on every start) for the
# actual initialization logic, then execs the CMD (sleep infinity by default,
# keeping the container alive for SSH / docker exec).
#
# Bypass init for debugging by setting `SKIP_INIT=1` in the pod's env.

set -e

if [ "${SKIP_INIT:-}" = "1" ]; then
    echo "[entrypoint] SKIP_INIT=1 — bypassing init.sh"
else
    echo "[entrypoint] Running deploy/init.sh"
    bash /opt/vllm-fork/deploy/init.sh
fi

exec "$@"
