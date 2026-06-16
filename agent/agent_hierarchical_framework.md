# 双层 VLM Agent — 框架与设计说明

> **实现细节（模块、话题、JSON、配置）以 [`MODULES.md`](MODULES.md) 为准。**  
> 本文档说明设计动机与语义；代码入口：`python3 -m agent.node`。

---

## 1. 动机

单层「总任务 + 单图 → 控制量」难以稳定完成多阶段任务。拆成两层：

| 层级 | 职责 | 默认频率 |
|------|------|----------|
| **Planner** | 总任务分解、维护子任务 list、`u`/`f` 决策 | `PLANNER_PERIOD_S`（默认 4s） |
| **Executor** | 只执行 `subtasks[-1]`，输出短期控制 | `EXECUTOR_PERIOD_S`（0.2s，~5Hz） |

Executor 两种模式（`config.EXECUTOR_MODE`）：

- **3d**：VLM 直接输出机体系 compact 运动量 `fwd/lat/vert/yaw`
- **2d**：VLM 输出图像像素点 + `distance_m`；本地反投影为机体系位置增量，**yaw = 射线相对 +X 前方的水平角**

---

## 2. 核心语义

### 2.1 更新 list = 切换子任务

无单独「切换」API。Planner 解析到 **`u=1`** 时 `append_subtask`：

1. 新 `description` 成为下层输入  
2. 保存当前 RGB 为 `rgb_snapshot`（供后续 Planner 的 **SUBTASK_START** 图）  
3. `started_at` 重置，`list_version++`

**`u=0`**：list 不变，下层继续执行当前子任务（常态）。

### 2.2 结束任务

**`f=1`** → `TaskState.mark_finished()` → 发布 `mode_manager.orientation.w=0`（同 `pub_triger.sh`）。

### 2.3 子任务文案

下层只见一条短句：`action + object + until <stop condition>`，由 Planner 在 `u=1` 时写入 `subtask` 字段。

---

## 3. 数据流（当前实现）

```
总任务 (agent_prompt.txt)
        │
        ▼
┌──────────────────┐  2 张前视 RGB          ┌─────────────────────────────┐
│ Planner          │ ◄────────────────────── │ 图1 SUBTASK_START           │
│ JSON: u, f,      │                         │ 图2 LIVE_NOW                │
│      subtask?    │                         │ + list 文本                 │
└────────┬─────────┘                         └─────────────────────────────┘
         │ u=1 append / u=0 不变 / f=1 finish
         ▼
┌──────────────────┐  1 张 LIVE_NOW          ┌─────────────────────────────┐
│ Executor 3D/2D   │ ◄────────────────────── │ 当前子任务 + 当前前视图      │
└────────┬─────────┘                         └─────────────────────────────┘
         │ /command_pos（VINS 仅本地变换）
         ▼
      MPC (mode -2)
```

**VINS 不进 VLM**：仅 `vins_received` 门控与 `body_delta_to_world`。

---

## 4. 图像输入约定

| 角色 | 张数 | 含义 |
|------|------|------|
| **Planner** | **2** | ① 当前子任务开始时的前视照片；② 当前时刻前视照片 |
| **Executor** | **1** | 当前时刻前视照片（LIVE_NOW） |

Prompt 中对图像的说明统一写在 `prompts.py`（前视相机、机体轴向、像素坐标系），不按 Camera/Image 拆多段。

尚无子任务时，Planner 两张图均为当前帧（代码用 LIVE_NOW 填充 SUBTASK_START）。

---

## 5. Planner JSON（u / f）

全局 `USE_JSON_RESPONSE_FORMAT=True`。模型**每次**应输出 **`u`** 与 **`f`**：

| 输出 | 含义 |
|------|------|
| `u=0` | **不**更新 subtask list |
| `u=1` | 追加下一条子任务（必填 `subtask` 文案） |
| `f=1` | 总任务完成 |
| `f=0` | 未完成 |

本地 `map_planner_to_list_action` **只认 `u`**，不再比较 subtask 字符串是否相同。

### 5.1 强制切换（force_switch）

单条子任务执行超过 `SUBTASK_MAX_DURATION_S`（10s）：

- Prompt 注入 `[FORCED SWITCH]`，禁止 `u=0`
- 解析层与 `planner.handle_response` 双重兜底：仍 `u=0` 或解析失败时强制 append（见 `MODULES.md` §6.3）

---

## 6. Executor JSON

### 6.1 3D

```json
{"fwd":"F<m>","lat":"L<m>|R<m>|0","vert":"U<m>|D<m>|0","yaw":"L<deg>|R<deg>|0"}
```

### 6.2 2D

```json
{"waypoint_2d":[x,y],"distance_m":<meters>}
```

本地：`camera_geom.waypoint_to_body_delta` → 机体系 `x_m,y_m,z_m,yaw_deg` → 与 3D 相同发布链。

`distance_m` 缺失时视为 **0**。2D 内参固定为 `config` 中 `CAM_*`（不订阅 `camera_info`）。

---

## 7. 子任务 List 结构

见 `task_state.py`：`SubTask(description, rgb_snapshot, started_at, ended_at)`，`MAX_LIST_LEN=10` FIFO。

---

## 8. 配置与启动

见 `config.py` 与 [`MODULES.md` §9](MODULES.md#9-configpy-要点)。

```bash
./run_agent.sh
python3 -m agent.node --executor-mode 2d
python3 -m agent.node --planner-reasoning --executor-reasoning
```

仿真：`sim_fly.sh` + `SIMULATE=True`；真机：`SIMULATE=False`，RealSense 话题与内参。

---

## 9. 并发模型

- ROS 主线程 + 后台 `asyncio` 线程跑 VLM  
- Planner / Executor 各维护 `request_seq`；Executor 校验 `list_version`  
- `TaskState` 使用 `threading.Lock`

---

## 10. 与单层 agent 的差异

| 维度 | 单层 `ori/agent.py` | 双层（当前） |
|------|---------------------|--------------|
| VLM 次数 | 1 | Planner + Executor |
| 任务记忆 | 无 list | 子任务 list + 起始 RGB 快照 |
| 上层输出 | — | `u` / `f` + 可选 `subtask` |
| 下层输入 | 总任务 + 图 | 仅当前子任务 + 1 图 |
| 下层输出 | 四键米制 | 3D compact 或 2D 像素+距离 |
| 结束 | 无 | `f=1` → mode_manager |

---

## 11. 实现状态

- [x] TaskState、双图 Planner、单图 Executor  
- [x] `u`/`f` JSON 与 force_switch 兜底  
- [x] 3D / 2D 双执行器  
- [x] `prompts.py` 内联 prompt  
- [x] 2D `camera_info` 在线内参  

`ori/` 仅作参考，不接入运行。
