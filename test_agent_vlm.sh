#!/bin/bash
# Agent VLM 诊断：先起 SSH 隧道（与 run_agent 相同），再跑 Python 探针
set -e
cd "$(dirname "$0")"

echo ">>> 启动 SSH 隧道 (localhost:8000 -> 202.120.36.186:8000) ..."
sshpass -p '#)!301epq' ssh -N -p 2145 -L 8000:127.0.0.1:8000 \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
    ps@202.120.36.186 &
SSH_PID=$!
trap 'kill $SSH_PID 2>/dev/null; wait $SSH_PID 2>/dev/null' EXIT INT TERM
sleep 2

source devel/setup.bash 2>/dev/null || true

echo ">>> 运行探针 (--ros --full) ..."
python3 -m agent.test_connectivity --ros --full "$@"
