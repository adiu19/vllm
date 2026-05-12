#!/bin/bash
# Shared SSH primitives for prefill/decode/proxy agents.
# Source this file, then call the functions.

PREFILL_SSH="a2jxofvx68dcrz-6441129a@ssh.runpod.io"
DECODE_SSH="7z1gckuweqh66u-64411280@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o RequestTTY=force"

PREFILL_IP="10.1.102.204"
DECODE_IP="10.0.5.208"

PROXY_LOG="/tmp/proxy.log"
PREFILL_LOG="/tmp/prefill.log"
DECODE_LOG="/tmp/decode.log"

# Run a command on a pod and return output.
# Usage: pod_run <prefill|decode> <command>
pod_run() {
    local pod="$1"; shift
    local cmd="$@"
    local host
    case "$pod" in
        prefill) host="$PREFILL_SSH" ;;
        decode)  host="$DECODE_SSH" ;;
        *) echo "Unknown pod: $pod" >&2; return 1 ;;
    esac
    ssh $SSH_OPTS "$host" <<EOF
$cmd
exit
EOF
}

# Start a command in the background on a pod, writing stdout+stderr to logfile.
# Usage: pod_start_bg <prefill|decode> <logfile> <command>
pod_start_bg() {
    local pod="$1"
    local logfile="$2"
    shift 2
    local cmd="$@"
    pod_run "$pod" "nohup bash -c '$cmd' > $logfile 2>&1 & echo \$!"
}

# Tail the last N lines of a log on a pod.
# Usage: pod_tail <prefill|decode> <logfile> [n]
pod_tail() {
    local pod="$1"
    local logfile="$2"
    local n="${3:-50}"
    pod_run "$pod" "tail -n $n $logfile 2>/dev/null || echo 'log not found'"
}

# Check if a process matching a pattern is running on a pod.
# Usage: pod_pgrep <prefill|decode> <pattern>
pod_pgrep() {
    local pod="$1"
    local pattern="$2"
    pod_run "$pod" "pgrep -fa '$pattern' | grep -v grep || echo 'NOT RUNNING'"
}

# Kill processes matching a pattern on a pod.
# Usage: pod_pkill <prefill|decode> <pattern>
pod_pkill() {
    local pod="$1"
    local pattern="$2"
    pod_run "$pod" "pkill -9 -f '$pattern' 2>/dev/null; echo done"
}

# Check TCP port reachability from a pod.
# Usage: pod_check_port <prefill|decode> <target_ip> <port>
pod_check_port() {
    local pod="$1"
    local ip="$2"
    local port="$3"
    pod_run "$pod" "python3 -c \"import socket; s=socket.socket(); s.settimeout(3); s.connect(('$ip',$port)); print('open')\" 2>&1 || echo 'FAIL'"
}

# GPU memory on a pod.
# Usage: pod_gpu <prefill|decode>
pod_gpu() {
    local pod="$1"
    pod_run "$pod" "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"
}
