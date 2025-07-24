#!/bin/bash

# 确保脚本具有可执行权限
# 运行前使用 `chmod +x restart_vins.sh` 来赋予可执行权限

# 设置ROS环境变量
source devel/setup.bash  # 根据你的ROS版本修改路径

# 发布/vins_restart话题，消息类型为std_msgs/Bool，数据为true
# rostopic pub /vins_imu_switch std_msgs/Bool "data: true" -1
rostopic pub /vins_restart std_msgs/Bool "data: true" -1
# while true; do
#     rostopic pub /vins_restart std_msgs/Bool "data: true" -1
#     sleep 0.05  # 延时0.1秒，达到10Hz频率
# done

echo "Published /vins_restart message to restart the estimator!"
