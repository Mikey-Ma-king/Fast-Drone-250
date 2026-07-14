#!/bin/bash
# dog_pos 四方案对比 tmux 布局（4 个 pane，命令预填不自动执行，各 pane 手动回车）
# 建议启动顺序：fake_target → tracker → compare(4 processors) → compare_viz
# 前提：仿真/真机基础栈（sim_fly、xtdrone 等）已在其它终端跑好

SESSION="dogpos_cmp"
WORKDIR="/home/pc/Fast-Drone-250"

PATHS=(
    "$WORKDIR"
    "$WORKDIR"
    "$WORKDIR"
    "$WORKDIR"
)

CMDS=(
    "python3 fake_target.py"
    "./tracker.sh"
    "source devel/setup.bash; roslaunch planning dog_pos_processor_compare.launch"
    "source devel/setup.bash; python3 dog_pos_compare_viz.py"
)

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

tmux new-session -d -s $SESSION -c "${PATHS[0]}"

for i in {1..3}; do
    tmux split-window -t $SESSION:0 -c "${PATHS[$i]}"
    tmux select-layout -t ${SESSION}:0 tiled
done

for i in {0..3}; do
    tmux send-keys -t ${SESSION}:0.$i "${CMDS[$i]}"
done

tmux attach-session -t $SESSION
