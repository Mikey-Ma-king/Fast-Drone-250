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
    "/home/pc/Fast-Drone-250"
)
# 对应命令
CMDS=(
    "./sim_fly.sh"
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

# 使用水平分割创建其余 6 个 pane
for i in {1..6}; do
    if [ $((i % 2)) -eq 1 ]; then
        # 奇数：水平分割
        tmux split-window -h -t $SESSION:0 -c "${PATHS[$i]}"
    else
        # 偶数：垂直分割
        tmux split-window -v -t $SESSION:0 -c "${PATHS[$i]}"
    fi
done

# 自动平铺布局（会变成网格）
tmux select-layout -t ${SESSION}:0 tiled

# 发送命令但不执行
for i in {0..6}; do
    tmux send-keys -t ${SESSION}:0.$i "${CMDS[$i]}"
done

# 附加会话
tmux attach-session -t $SESSION
