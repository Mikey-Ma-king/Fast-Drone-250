#!/bin/bash

SESSION="real_fly"

# 你的路径数组（可自定义）
PATHS=(
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
)

# 每个终端的预设命令（不要加 ./ 前缀也行）
CMDS=(
    "roscore"
    "./zydauto0.sh"
    "./zydauto1.sh"
    "./zydauto2.sh"
    "python3 rebort.py"
    ""
    "./pub_triger.sh"
    "./land_triger.sh"
)

# 如果会话存在，先杀掉
tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
    tmux kill-session -t $SESSION
fi

# 创建会话和第一个窗口
tmux new-session -d -s $SESSION -c "${PATHS[0]}"

# 逐个从上往下分出新的面板
for i in {1..4}; do
    # tmux select-pane -t $((i-1))       # 选择上一个面板
    tmux split-window -v -c "${PATHS[$i]}"   # 向下分割
    tmux split-window -h -c "${PATHS[$i]}"  # 垂直分割
done

# 设置竖向布局
tmux select-layout even-vertical

# 发送命令但不执行
for i in {0..7}; do
    tmux send-keys -t ${SESSION}:0.$i "${CMDS[$i]}"
done

# 附加会话
tmux attach-session -t $SESSION
