# Vicon 评价接收与真机录包

真机入口沿用 `tmux_real_fly.sh → zydauto1.sh → tracker.sh`。
不需要另外开一个录包窗口。tmux 仍只预填命令，各 pane 手动回车。
如需填写本次场景，可把该 pane 的命令补成 `./zydauto1.sh --label run01 --scene 场景描述`；参数会传到录包器。

- `zydauto1.sh` 新增独立 Vicon 接收端，服务器为 **192.168.39.162**。
- `tracker.sh` 并行运行原 `traj_server` 与增强录包器，修复原先前台 roslaunch 阻塞录包的问题。
- 从 zydauto1 所在 pane 按 Ctrl+C，逐层通知录包器关闭 bag 索引，等待 `Recorder closed` 后再关窗口。
- 录包器不修改 SchurVINS 或融合算法；相机和 read 的 30 Hz 适配见 [read 说明](../read/README.md)。

## Vicon 输出

无人机 Subject 为 **drone00**，机器狗为 **quadruped00**。当前接收全部刚体：

| 话题 | 内容 |
| --- | --- |
| `/evaluation/vicon/<Subject>/pose` | 原始 Vicon 位姿，评价优先使用 |
| `/evaluation/vicon/<Subject>/odom` | 接收端 KF 后位姿和速度 |
| `/evaluation/vicon/diagnostics` | SDK 帧号、SDK 总延迟、逐刚体原始位姿有效性 |

原 launch 通往 `/vins_fusion/imu_propagate` 的重映射已移除；Vicon 不接入定位或控制，TF 发布关闭。
位置由 mm 转为 m，轴为 Forward/Left/Up，参考帧 `mocap`。

每帧各刚体共享主机 ROS 接收时间。诊断明确记录时间戳来源；它不是已同步的采集时间。
SDK 总延迟不保证包含网络及全部端到端延迟。本次没有猜测时钟偏移或自动减去延迟。

## 录制内容与目录

清单集中在 [config/recording.yaml](config/recording.yaml)：

- 双方里程计、视觉输出、UWB、光流，以及初始化/重置所需的少量事件。
- SW3 实际输入、seed、bearing gate、alignment、估计输出、融合采用/拒绝计数和计算耗时状态。
- 仅 `drone00`、`quadruped00` 的原始 Vicon 位姿和 Vicon 诊断；后启动的同名话题也会录到。
- SchurVINS pose、info、read 原始观测龄期/有效性诊断，MAVROS 时钟状态和飞控连接状态。
- 起飞/接管排查必需的小话题：`/position_cmd`、`/mode_manager`、`/debugPx4ctrl`、MAVROS 姿态指令、RC、带姿态的 IMU 与 landed 状态。
- 静态 TF、一份 rosout，以及实验 metadata、health、events。

默认不录双目图像、原始 IMU、Vicon KF 输出/其他刚体、重复 preview、调试 offset、动态 TF 和完整规划轨迹。
接收端仍可发布全部刚体，精简仅影响录包。视觉有效性、UWB、估计状态和融合输入继续保留。

默认保存到 `~/perching_bag/日期_标签_唯一编号/`，每 300 秒或约 2 GiB 分包，全部保留。
默认不压缩，减少 CPU 开销。磁盘余量不足 1 GiB 时 rosbag 会停止写入，需看录包日志。
目录同时保存 `run.json`、起止参数和 ROS 图快照、相关源码及配置快照、`health.jsonl`、`rosbag_record.log`。
参数快照包含 `/px4ctrl`、`/traj_server`；源码快照包含起飞状态机、控制器、里程计接口和控制配置，供故障复查。
源码快照不等于运行二进制版本的证明。方法名称是人工标签，不能据此认定该方法已实际产出结果。

Health 每秒记录消息计数/频率、接收龄期、最大接收间隔、重复/回退/零时间戳等。
Health 反映监测订阅者收到的消息，完整性仍应以 bag 内容和录包日志为准。
没有输出的话题保持零计数，不阻止保留其他数据；收到 UWB 也不等于融合采用，采用数量看 SW3 status。

默认可以回放融合输入，无法从日志重跑 SchurVINS。如需重跑前端，显式追加 `--with-images`，同时录双目 IR 图像、camera_info 和原始 IMU。
例如 `./zydauto1.sh --with-images` 或 `./shfiles/record_vicon.sh --with-images`。旧的 `--no-images` 参数仍兼容，等同默认行为。
ArUco 前端目前没有原始 USB 图像发布话题，本次只录其已有输出，没有修改前端。
可选基线若发布 `/quaddock/joint/.*` 或 `/quaddock/h0/.*` 也会录制，其他名称需补进清单；本工具不启动基线。

## 单独接收或录包

通常使用原 tmux 流程即可。需要仅收流/录包时，可在两个终端分别执行：

```bash
./shfiles/vicon_eval.sh
./shfiles/record_vicon.sh --label run01 --scene '平台运动和场景说明'
```

这两个脚本不启动规划器、定位或运动。不要与原启动链路重复运行接收端或录包器。
更改服务器可传 `server_address:=IP`；更改日志位置可传 `--output-dir PATH`。
以后修改接收端代码时执行 `./shfiles/build_vicon_logging.sh`，只构建接收与录包相关包。

## 人工事件标记

以下命令仅记录注释，不会屏蔽视觉或操纵平台；请在实际操作发生时调用：

```bash
./shfiles/mark_vicon_event.sh vision_mask_start '人工遮挡开始'
./shfiles/mark_vicon_event.sh vision_mask_end '撤去遮挡'
./shfiles/mark_vicon_event.sh maneuver_start '狗开始转弯'
./shfiles/mark_vicon_event.sh maneuver_end '转弯结束'
./shfiles/mark_vicon_event.sh note '其他备注'
```

自然失观需结合视觉消息分析，不能把所有消息缺口当成人工屏蔽。

## 待补评价信息与验证

清单 evaluation 段已记录双方 Subject；刚体到机体/任务点外参、时钟同步及标定来源暂为 null。
这些信息应按实际标定补齐，录制成功不等于已经校准。

本机已编译接收端。独立 ROS master 下的短测可以用以下命令重做：

```bash
source devel/setup.bash
/usr/bin/python3 src/quaddock_logging/test/smoke_recording.py
```

合成测试走实际 `tracker.sh`，把其规划器启动替换为无动作的空进程，验证延迟出现的刚体、注释、时间戳及关包。
追加 `--hardware` 会仅连接实际 Vicon 并短时录包，不启动规划器或真机运动。
测试日志保存在打印的 `/tmp/quaddock_recording_smoke_*` 目录。
