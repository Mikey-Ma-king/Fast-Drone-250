#!/bin/bash
# 启动 agent 模式：MPC 追踪 /command_pos（orientation.w = -2）
rostopic pub /mode_manager geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: -2.0"
