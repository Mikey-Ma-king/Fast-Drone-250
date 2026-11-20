# 原始 Fast-Drone-250 真机线接入说明

本包对应本机原始工程：

- `/home/pc/Fast-Drone-250/src/planning/src/dog_pos_processor.cpp`
- `/home/pc/Fast-Drone-250/src/planning/launch/dog_pos_processor.launch`
- `/home/pc/Fast-Drone-250/linktarck.py`

我检查到当前这份 `dog_pos_processor.cpp` 构造函数先把 `simulate_mode_` 初始化为
`false`，但后面又明确写了 `simulate_mode_ = true`，随后把 `camera_offset_` 置零。
这不是 Ours 融合改动的一部分，但是真机运行前必须由接收方确认；否则“真机线”仍会沿用仿真相机杆臂设置。

真机侧 agent 应以原工程为主体，只替换 `dog_pos_processor.cpp` 中从 `// 维护AOA位置偏移（单距离单角度方案）` 开始的融合段。原来的输入回调、状态机、起飞重置、偏置初始化、输出消息字段、定时器频率和 planner 接口都保留。

## 保留项

以下逻辑不改：

- `linktarck.py` 的串口解析、距离/角度滤波和 `/AOA_Tag_data` 发布；
- `/AOA_Tag_data` 字段约定：`pose.position.x` 是距离，`pose.orientation.x` 是角度；
- 原 `aoa_count_` / `aoa_received_` 的五次状态检查节奏；
- 原 `vins_received_`、`precise_yaw_offset_ready_` 和起飞重置流程；
- 原 dog yaw、位置偏置、控制输出和 `/dog_pos_processed` 字段编码；
- 原 planner、traj_server、PX4 和其它底层节点。

原工程与新的 Ours 节点不能同时发布同一个 `/dog_pos_processed`，否则会出现两个融合结果互相覆盖。当前包默认只发布 `/quaddock/target_odom` 和诊断用 `/quaddock/legacy_preview`，不会抢占原 `/dog_pos_processed`。

## 唯一融合变化

AOA 不再先在上层转换成一个世界系坐标再和视觉坐标做位置融合。每个匹配时刻向 Ours-SW3 送两条独立事件：

1. `uwb_range`：收到有限距离就送；不再用距离最小值或角度门控丢弃距离。
2. `uwb_bearing`：沿用原上层 AOA 可用状态和朝向门控后再送。

原源码中实际执行的朝向门控是 `abs(heading_diff) <= M_PI/4.0`，也就是 ±45°；保留这个条件。原来的 `aoa_min_distance_` 和高度差提前 return 只对新的 range 事件取消，因为当前约定是“有距离就放进去”，距离约束交给 Ours 的标量 range 残差处理。包内适配器镜像原 `aoa_received_`/`vins_received_` 的 100 ms、五次失联节奏；正式集成时仍应直接使用原 C++ 状态。

Ours 的视觉输入仍使用原 `/target_ekf_odom`，UAV 使用 `/vins_fusion/imu_propagate`，dog 使用 `/dog_pos`。最终结果按原 `/dog_pos_processed` 消息语义回填，不能把 Ours 内部协方差或标准四元数写入原工程约定的 yaw/ready 字段。

## 不需要改的文件

在这条真机线上不需要改 `linktarck.py`、VINS、PX4、traj_server、planner、底层串口或仿真 launch。若采用本包的 Python 核心，CMake/launch 只负责把它作为唯一的融合实现启动；不能同时启动原 `dog_pos_processor` 和新的融合发布节点。
