# 双层 VLM Agent — 实现文档

> **权威参考**：与当前代码一致（`python3 -m agent.node`）。  
> 设计背景见 `agent_hierarchical_framework.md`。旧单层备份在 `ori/`。

---

## 1. 启动与模式

```bash
# 仓库根目录
./run_agent.sh                    # 默认 config.EXECUTOR_MODE
python3 -m agent.node --executor-mode 3d   # 3D 执行器
python3 -m agent.node --executor-mode 2d   # 2D 执行器
```

仿真：`sim_fly.sh` 使用 `vehicle_sdf:=iris_realsense_camera`；`SIMULATE=True` 时 RGB/深度为 `/realsense/depth_camera/color|depth/image_raw`，内参见 `config.py`（来自 `realsense_camera.sdf`）。

---

## 2. 总数据流

```
RGB + VINS(门控/坐标变换，不进 VLM)
  → node.py
  → Planner（PLANNER_PERIOD_S，默认 4s）维护子任务 list
  → Executor（EXECUTOR_PERIOD_S=0.2s，~5Hz）→ /command_pos
  → f=1 时发布 mode_manager.w=0
```

```
agent_prompt.txt（总任务）
        │
        ▼
┌─────────────────┐   双图 + JSON u/f    ┌──────────────────────────┐
│ Planner         │ ◄─────────────────── │ SUBTASK_START + LIVE_NOW │
│ prompts.py      │                      │ 子任务 list 文本          │
└────────┬────────┘                      └──────────────────────────┘
         │ u=1 → append_subtask + 存 RGB 快照
         │ u=0 → list 不变
         │ f=1 → finished
         ▼
┌─────────────────┐   单图 LIVE_NOW      ┌──────────────────────────┐
│ Executor 3D/2D  │ ◄─────────────────── │ 当前子任务 description    │
└────────┬────────┘                      └──────────────────────────┘
         │ 3D: fwd/lat/vert/yaw → 机体系增量
         │ 2D: waypoint_2d + distance_m → 反投影 + yaw
         ▼
   /command_pos（VINS 快照做 body→world）
```

---

## 3. 模块一览

| 文件 | 职责 |
|------|------|
| `node.py` | ROS 入口、订阅、Timer、asyncio、组装 Planner/Executor |
| `config.py` | 话题、VLM、限幅、executor 模式、相机内参默认 |
| `task_state.py` | 子任务 list、`list_version`、`force_switch()` |
| `planner.py` | 上层 VLM：双图请求、解析 u/f、append/finish |
| `executor.py` | 3D 下层：compact `fwd/lat/vert/yaw` JSON |
| `executor_2d.py` | 2D 下层：像素点 + 距离 JSON |
| `prompts.py` | 三层 prompt（代码内联，非 txt 拼接） |
| `motion_parse.py` | Planner/3D Executor JSON 解析 |
| `parse_2d.py` | 2D Executor JSON 解析 |
| `attitude.py` | VINS 四元数：机体系向量/bearing → 世界系 |
| `camera_geom.py` | 2D：像素→body bearing→世界系对比与位移；再转机体系给 command |
| `camera_intrinsics.py` | 2D 内参；仅 `config` 启动时注入 |
| `command.py` | 限幅、`body_delta_to_world`、发布 `/command_pos` |
| `cmd_velocity.py` | 维护目标点轨迹速度（相邻目标差分 + EMA，限幅） |
| `vlm_utils.py` | 缩放、PNG base64、单图/多图 messages、异步 VLM |
| `overlay.py` | 预览/录制面板 |

---

## 4. ROS 话题

| 话题 | 方向 | 说明 |
|------|------|------|
| `RGB_IMAGE_TOPIC` | 订阅 | 仿真 `/realsense/depth_camera/color/image_raw`；真机 `/camera/color/image_raw` |
| `/vins_fusion/imu_propagate` | 订阅 | 门控 + `/command_pos` 变换，**不进 VLM** |
| `/command_pos` | 发布 | `nav_msgs/Odometry`，MPC 跟踪 |
| `/mode_manager` | 发布 | `f=1` 时 `orientation.w=0` |

---

## 5. 子任务 List（task_state.py）

```python
@dataclass
class SubTask:
    description: str
    rgb_snapshot: np.ndarray   # append 时刻 RGB（供下次 Planner 的 SUBTASK_START）
    started_at: rospy.Time
    ended_at: Optional[rospy.Time]
```

- `subtasks[-1]` = 当前子任务（下层只读其 `description`）
- `append_subtask`：关闭上一条计时 → 写入新条 + 当前 RGB 快照 → `list_version += 1`
- 超过 `MAX_LIST_LEN`（10）从头部 FIFO 丢弃
- **无** status 字段；是否更新 list 完全由 Planner 的 **`u`** 决定

---

## 6. Planner（上层）

### 6.1 频率与图像

- 周期：`PLANNER_PERIOD_S`（默认 **4.0 s**）
- **固定 2 张图**（`build_messages_multi_image`）：
  1. **SUBTASK_START**：当前子任务开始时的前视 RGB（无子任务时与 LIVE_NOW 相同）
  2. **LIVE_NOW**：当前前视 RGB
- 图像 letterbox 到 **`IMG_MAX_SIZE×IMG_MAX_SIZE`**（448×448，等比+黑边），`uint8` PNG base64
- Prompt 文本在 `prompts.build_planner_prompt()`（含双图说明、list、总任务）

### 6.2 VLM JSON 输出（`USE_JSON_RESPONSE_FORMAT=True`）

**必须带 `u` 和 `f`：**

| 字段 | 含义 | 本地行为 |
|------|------|----------|
| `u=0` | 不更新 subtask list | `map` → `{"u":0}`，下层继续当前条 |
| `u=1` | 追加下一条子任务 | 需非空 `subtask` → `append_subtask(subtask, 当前RGB)` |
| `f=1` | 总任务结束 | `mark_finished()` + `mode_manager.w=0` |
| `f=0` | 任务未结束 | — |

无 reasoning 时格式：`{"u":0|1,"f":0|1,"subtask":"..."}`（`u=1` 时必填 subtask）。  
有 reasoning 时可含 `target_visible`、`scene_summary`、`reasoning`。

解析：`motion_parse.parse_planner_vlm_output` → `map_planner_to_list_action`。  
**不再**用 subtask 文本相似度判断更新；只看 **`u`**。

### 6.3 强制切换（force_switch）

当 `current_subtask_elapsed_s() >= SUBTASK_MAX_DURATION_S`（10s）：

- Prompt 追加 `[FORCED SWITCH]`：禁止 `u=0`，必须 `u=1` 新 subtask 或 `f=1`
- `map`：若模型仍 `u=0`/缺 `u` → 强制 `u=1`（subtask 或 `SUBTASK_TIMEOUT_FALLBACK_DESC`）
- `handle_response`：若解析后仍为 `u=0` 且 `force_switch` → 兜底 append

---

## 7. Executor（下层）

### 7.1 公共

- 周期：`EXECUTOR_PERIOD_S = 0.2` s（~5 Hz）
- 前置：list 非空、未 `finished`、`vins_received && rgb_received`
- **固定 1 张图**：LIVE_NOW（当前前视 RGB）
- 只读 `subtasks[-1].description`，**不传**总任务、VINS
- 回复时若 `list_version` 已变则丢弃（防切换后旧回复）

### 7.2 3D 模式（`executor.py`，`EXECUTOR_MODE="3d"`）

**VLM JSON：**

```json
{"fwd":"F0.3","lat":"L0.1|R0|0","vert":"U0|D0|0","yaw":"L10|R0|0"}
```

- 机体系：+X 前、+Y 左、+Z 上
- `motion_parse.parse_executor_output` → `clamp_body_delta` → `body_delta_to_world` → `/command_pos`

### 7.3 2D 模式（`executor_2d.py`，`EXECUTOR_MODE="2d"`）

**VLM JSON：**

```json
{"waypoint_2d":[x,y],"distance_m":1.5}
```

- `[x,y]`：LIVE_NOW 坐标（左上原点，x 右 y 下）；优先 **0~1000** 相对坐标 → `u_vlm=x*rw/1000`，再映射到相机像素 + 内参反投影
- `distance_m`：相机到该 3D 点的距离（米）；未给则按 **0**（`parse_2d`）
- **本地不算 yaw 输出**；由 `camera_geom.waypoint_to_body_delta`：
  - 像素 + 距离 → 相机系 → 机体系 **位置增量** `x_m,y_m,z_m`
  - **yaw 增量**：航点 `u` 相对画面水平中心偏多少 → `yaw_deg = -YAW_2D_GAIN_DEG * (u-center)/half_width`（右偏右转/负，左偏左转/正）；`distance_m=0` 时仅 yaw
  - 发布：`target_yaw = vins_yaw + yaw_deg`
- 内参：`config` 中 `CAM_*` / `CAM_*_SIM|REAL`，`node` 启动时写入 `camera_intrinsics`

---

## 8. 硬限幅（command.py，不进 prompt）

| 参数 | 默认 | 作用 |
|------|------|------|
| `BODY_DELTA_MAX_M` | 1.0 | 机体系 x/y/z 单步上限 |
| `YAW_DELTA_MAX_DEG` | 30 | yaw 增量钳制上限（度） |
| `YAW_2D_GAIN_DEG` | 30 | 2D 航点偏到画面左右边缘时的 yaw 增量幅度（度） |
| `CMD_Z_MIN_M` / `CMD_Z_MAX_M` | 0 / 1.0 | 世界系 command.z |

---

## 9. config.py 要点

| 参数 | 默认 | 说明 |
|------|------|------|
| `PLANNER_PERIOD_S` | 4.0 | Planner 周期 |
| `EXECUTOR_PERIOD_S` | 0.2 | Executor 周期 |
| `EXECUTOR_MODE` | `"3d"` | `"3d"` 或 `"2d"` |
| `MAX_LIST_LEN` | 10 | 子任务 list 上限 |
| `SUBTASK_MAX_DURATION_S` | 10.0 | 超时 force_switch |
| `IMG_MAX_SIZE` | 448 | VLM 正方形边长（固定 448×448） |
| `PLANNER_REASONING` / `EXECUTOR_REASONING` | False | 是否要求 reasoning 字段 |
| `EXECUTOR_2D_BBOX` | False | 2D executor 是否要求 bbox（仅可视化） |
| `CMD_V_MAX_XY_MPS` / `CMD_V_MAX_Z_MPS` | 0.3 | 目标轨迹速度上限 |
| `CMD_V_EMA_ALPHA` | 0.35 | 目标速度一阶低通系数 |
| `SIMULATE` | True | 仿真/真机话题与内参切换 |

---

## 10. VLM 调用（vlm_utils.py）

- OpenAI 兼容 API（vLLM），`response_format={"type":"json_object"}`
- 多图 message：`[image_1, image_2, ..., text]`（Planner）
- 单图 message：`[image, text]`（Executor）
- 异步请求 + seq；超时丢弃

---

## 11. Prompt 来源

| 组件 | 来源 |
|------|------|
| 总任务 | `agent/agent_prompt.txt` |
| Planner / Executor | `prompts.py` 函数动态生成（**非** `agent_prompt_planner.txt`） |

---

## 12. 与旧文档差异（勿再参考）

| 旧说法 | 当前实现 |
|--------|----------|
| Planner 1Hz | 默认 **4s**（`PLANNER_PERIOD_S`） |
| Planner 3 图 / `PLANNER_RGB_WINDOW` | **固定 2 图**：SUBTASK_START + LIVE_NOW |
| Executor 双图 | **单图** LIVE_NOW |
| Executor 输出 `x_m,y_m,z_m,yaw_deg` | **3D**：`fwd/lat/vert/yaw`；**2D**：`waypoint_2d`+`distance_m` |
| 用 subtask 文本判断是否更新 | 用 JSON **`u`**；**`f`** 结束 |
| `agent_prompt_planner.txt` | 已不用，见 `prompts.py` |

---

## 13. ori/

`ori/agent.py`、`example_executor&planner.py` 等为参考/备份，**不参与** `python3 -m agent.node` 运行。
