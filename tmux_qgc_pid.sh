#!/bin/bash

SESSION="qgc_pid"

# 你的路径数组（可自定义）
PATHS=(
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
    "/home/pc/Fast-Drone-250"
)

# 每个终端的预设命令（不要加 ./ 前缀也行）
CMDS=(
    "source ~/Fast-Drone-250/devel/setup.bash;
roslaunch realsense2_camera rs_camera.launch"
    "./read.sh"
    "roslaunch mavros px4.launch fcu_url:=/dev/USB_px4:921600 gcs_url:='udp://@127.0.0.1:14550'"
    "source ~/schurvins/devel/setup.bash;
roslaunch svo_ros euroc_vio_stereo.launch"
    "python3 vins2odom.py"
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
    tmux split-window -h -c "${PATHS[$i]}"   # 向下分割
done

# 设置竖向布局
tmux select-layout even-vertical

# 发送命令但不执行
for i in {0..4}; do
    tmux send-keys -t ${SESSION}:0.$i "${CMDS[$i]}"
done

# 附加会话
tmux attach-session -t $SESSION
