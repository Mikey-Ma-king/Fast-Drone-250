#!/bin/bash
# Agent 仿真测试 tmux 布局（6 个 pane，命令预填不自动执行，各 pane 手动回车）
# 建议启动顺序：sim_fly → xtdrone → MPC → run_agent → agent_triger → tracker

SESSION="agent_test"
WORKDIR="/home/pc/Fast-Drone-250"

PATHS=(
    "$WORKDIR"
    "$WORKDIR"
    "$WORKDIR"
    "$WORKDIR"
    "$WORKDIR"
    "$WORKDIR"
)

CMDS=(
    "./sim_fly.sh"
    "./xtdrone.sh"
    "./MPC.sh"
    "./run_agent.sh"
    "./agent_triger.sh"
    "./tracker.sh"
)

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

tmux new-session -d -s $SESSION -c "${PATHS[0]}"

for i in {1..5}; do
    if [ $((i % 2)) -eq 1 ]; then
        tmux split-window -h -t $SESSION:0 -c "${PATHS[$i]}"
    else
        tmux split-window -v -t $SESSION:0 -c "${PATHS[$i]}"
    fi
done

tmux select-layout -t ${SESSION}:0 tiled

for i in {0..5}; do
    tmux send-keys -t ${SESSION}:0.$i "${CMDS[$i]}"
done

tmux attach-session -t $SESSION
