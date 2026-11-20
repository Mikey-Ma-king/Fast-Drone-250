# SchurVINS 无人机里程计转发

真机相机入口 `realsense2_camera/launch/rs_camera.launch` 的双目 IR 默认 **30 Hz**。`read` 仍向 `/vins_fusion/imu_propagate` 提供 **200 Hz** 接口；额外频次来自有时间限制的模型外推，不代表增加了原始观测。

无人机支路直接接受 `/svo/pose_imu` 的有效位置与姿态，使用相邻两条接受观测的原始 `header.stamp` 时间差计算世界系线速度，再按同一匀速模型将位置外推到输出时间。姿态沿用最新有效观测。该支路不再经过 30 点 Bezier 速度拟合，也不再假定输入固定 60 Hz。ArUco 目标检测及目标/狗的 Bezier 预测流程保留。

`svo_odometry.h` 的默认限制：

- 原始观测龄期超过 0.10 s，不接受新样本，也不继续发布旧状态。
- 输入间隔超过 0.20 s 后重新等待有效速度样本对。
- 非有限位置、无效姿态、零/重复/回退时间戳和未来超过 0.01 s 的观测不接受。
- 相邻观测速度范数达到 4 m/s 时，该状态不输出。
- 订阅队列为 1，回调与发布共享状态有互斥保护。

时间限制依赖相机时间戳与主机 ROS 时间的一致性。输出时间戳为当前模型预测时间，原始时间不会被丢弃：`/read/odometry_diagnostics` 以约 10 Hz 记录接受观测的时间/龄期、最新输入的采集时间/接收龄期及发布/拒绝原因，录包器默认保留此话题。上游 SchurVINS 排队造成的延迟不会被转发节点修复；超过龄期限制时会无输出。

只构建此包（从工作区根目录）：

```bash
source devel/setup.bash
cmake -S src -B build/read_odometry -DCATKIN_WHITELIST_PACKAGES=read -DCATKIN_DEVEL_PREFIX="$PWD/devel" -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCATKIN_ENABLE_TESTING=ON
cmake --build build/read_odometry --target read test_svo_odometry read_odometry_replay -j2
devel/lib/read/test_svo_odometry
```

单元测试覆盖 15/30/60 Hz 真实时间差、200 Hz 外推、速度反向、源数据超时、时间戳异常及断流恢复。`read_odometry_replay` 为无 ROS I/O 的离线测试工具，接收 CSV：`receipt,source_stamp,x,y,z,qx,qy,qz,qw`。

已在旧日志上检查速度反馈与陈旧输入拒绝，不能据此证明新的闭环飞行已经稳定。需要相机和 read 的下一次启动才应用修改；未自动重启进程。`zydauto2.sh` 包含自动起飞命令，不应用作仅重启/验证的脚本。
