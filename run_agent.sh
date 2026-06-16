#!/bin/bash
cd "$(dirname "$0")"
source devel/setup.bash 2>/dev/null

sshpass -p '#)!301epq' ssh -N -p 2145 -L 8000:127.0.0.1:8000 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
        ps@202.120.36.186 &
SSH_PID=$!

trap 'kill $SSH_PID 2>/dev/null; pkill -f "ssh -N -p 2145 -L 8000:127.0.0.1:8000.*202.120.36.186" 2>/dev/null' EXIT INT TERM

python3 -m agent.node "$@"
