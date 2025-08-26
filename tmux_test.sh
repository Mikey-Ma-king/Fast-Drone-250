#!/bin/bash

SESSION="xtrone"

# 路径数组
PATHS=(
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
)

# 对应命令
CMDS=(
    "./xtdrone.sh"
    "python3 fake_target.py"
    "./MPC.sh"
    "./pub_triger.sh"
    "./land_triger.sh"
    "./tracker.sh"
)

# 如果会话已存在，则杀掉
tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

# 创建会话和窗口
tmux new-session -d -s $SESSION -c "${PATHS[0]}"

# 使用水平分割创建其余 5 个 pane
for i in {1..5}; do
    tmux split-window -h -t $SESSION -c "${PATHS[$i]}"
done

# 自动平铺布局（会变成网格）
tmux select-layout -t ${SESSION}:0 tiled

# 发送命令但不执行
for i in {0..5}; do
    tmux send-keys -t ${SESSION}:0.$i "${CMDS[$i]}"
done

# 附加会话
tmux attach-session -t $SESSION
