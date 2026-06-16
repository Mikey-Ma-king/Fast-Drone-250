#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双层 VLM agent 全局配置（按模块分块，改参数请只动对应块）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# =============================================================================
# 路径与全局提示
# =============================================================================

AGENT_DIR = Path(__file__).resolve().parent  # agent 包根目录
GLOBAL_PROMPT_FILE = AGENT_DIR / "agent_prompt.txt"  # 叠加到 Planner/Executor 系统提示的任务背景

# =============================================================================
# 运行环境：仿真 / 真机、图像与 VINS 话题
# =============================================================================

SIMULATE = True  # True=Gazebo iris_realsense_camera；False=真机 RealSense

# --- 真机 RealSense（realsense2_camera 默认命名）---
RGB_IMAGE_TOPIC_REAL = "/camera/color/image_raw"
DEPTH_IMAGE_TOPIC_REAL = "/camera/depth/image_rect_raw"

# --- 仿真 PX4 iris_realsense_camera + model://realsense_camera ---
# 见 PX4_Firmware/Tools/sitl_gazebo/models/realsense_camera/realsense_camera.sdf
#   cameraName=realsense/depth_camera → color/image_raw、depth/image_raw
RGB_IMAGE_TOPIC_SIM = "/iris/realsense/depth_camera/color/image_raw"
DEPTH_IMAGE_TOPIC_SIM = "/iris/realsense/depth_camera/depth/image_raw"

RGB_IMAGE_TOPIC = RGB_IMAGE_TOPIC_SIM if SIMULATE else RGB_IMAGE_TOPIC_REAL
DEPTH_IMAGE_TOPIC = DEPTH_IMAGE_TOPIC_SIM if SIMULATE else DEPTH_IMAGE_TOPIC_REAL

VINS_TOPIC = "/vins_fusion/imu_propagate"  # 位置、速度、姿态（四元数）
COMMAND_POS_TOPIC = "/command_pos"  # 发布目标位姿；MPC triger==2 跟踪
MODE_MANAGER_TOPIC = "/mode_manager"  # 任务结束发布 orientation.w=0

STATUS_CHECK_PERIOD_S = 0.2  # 定时检查 RGB/VINS 是否断流（秒）
SENSOR_LOSS_TICKS = 2  # 连续多少个检查周期无新帧则判传感器丢失

# =============================================================================
# VLM 服务（OpenAI 兼容 vLLM）
# =============================================================================

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"  # vLLM OpenAI 兼容根地址
VLLM_API_KEY = "EMPTY"  # 本地 vLLM 占位 key
VLLM_MODEL: Optional[str] = "/home/ps/ltc/Qwen3-VL-8B-Instruct-FP8/"  # 模型路径或名；None 时由 vlm_utils 自动选择
VLLM_HTTP_TIMEOUT_S = 300.0  # 单次 HTTP 请求超时（秒）
USE_JSON_RESPONSE_FORMAT = True  # 请求 response_format=json_object，约束输出为 JSON
VLM_TEMPERATURE = 0.0  # 采样温度，0=确定性，与 example_executor&planner 一致

IMG_MAX_SIZE = 640  # 送 VLM 前长边上限（等比缩放，不拉成正方形）
VLM_IMAGE_UINT8 = True  # True=编码 PNG 前转 uint8；False=保持原 dtype

# =============================================================================
# Planner（上层子任务）
# =============================================================================

PLANNER_PERIOD_S = 4.0  # 两次 Planner 请求最小间隔（秒）
PLANNER_MAX_TOKENS = 256  # Planner 补全 max_tokens
PLANNER_MAX_TOKENS_REASONING = 512  # 若开启 reasoning 时的 max_tokens 上限
PLANNER_RESPONSE_TIMEOUT_S = 30.0  # 等待 Planner 响应超时（秒）
PLANNER_REASONING = False  # True=要求 JSON 含 reasoning 字段（更长、更慢）

# =============================================================================
# Executor（下层执行）
# =============================================================================

EXECUTOR_MODE = "2d"  # "2d"=像素航点+本地几何；"3d"=compact fwd/lat/vert/yaw JSON
EXECUTOR_PERIOD_S = 0.2  # 两次 Executor 请求最小间隔（秒）
EXECUTOR_MAX_TOKENS = 128  # Executor 补全 max_tokens
EXECUTOR_RESPONSE_TIMEOUT_S = 8.0  # 等待 Executor 响应超时（秒）
EXECUTOR_REASONING = False  # True=要求 JSON 含 reasoning 字段
EXECUTOR_2D_BBOX = False  # True=2D JSON 含 bbox（仅面板可视化，不参与控制）

# =============================================================================
# 子任务状态与超时
# =============================================================================

MAX_LIST_LEN = 10  # 已完成子任务描述列表最大条数（送入 Planner 上下文）
SUBTASK_MAX_DURATION_S = 10.0  # 单条子任务最长执行时间，超时强制换策略
SUBTASK_TIMEOUT_FALLBACK_DESC = "Change the strategy."  # 超时写入已完成列表的占位描述

# =============================================================================
# 预览面板、终端统计、录像
# =============================================================================

SHOW_RGB = True  # True=cv2 弹窗（左图 + 右侧文字面板）
SAVE_VIDEO = True  # True=把合成面板写入 mp4
VIDEO_SAVE_DIR = AGENT_DIR / "recordings"  # 录像输出目录
VIDEO_SAVE_FPS = 5.0  # 录像帧率（Hz）；在 rgb 回调里节流写入

PANEL_LEFT_PAD = 20  # 面板左侧原图四周留白（像素）
PANEL_TEXT_WIDTH_RATIO = 1.8  # 右侧文字区宽度 = 左图宽度 × 该比例
PANEL_TEXT_MIN_WIDTH = 520  # 文字区最小宽度（像素）
PANEL_FONT_SIZE = 14  # 面板字体大小
PANEL_LINE_H = 20  # 面板行高（像素）
PANEL_RATE_WINDOW_S = 2.0  # 面板显示的 Planner/Executor 发送频率统计窗口（秒）
STATS_PRINT_PERIOD_S = 4.0  # 终端打印 RGB/VINS/发送计数周期（秒）

# =============================================================================
# /command_pos 目标速度（cmd_velocity，供 MPC 跟踪）
# =============================================================================

CMD_V_MAX_XY_MPS = 0.3  # 目标点水平速度模长上限（m/s），建议与 MPC_V_MAX_AGENT 一致
CMD_V_MAX_Z_MPS = 0.3  # 目标点竖直速度绝对值上限（m/s）
CMD_V_MIN_STEP_M = 0.005  # 相邻目标位移小于该值则本帧速度置 0（抑制抖动）
CMD_V_EMA_ALPHA = 0.35  # 差分速度一阶低通系数（1=不平滑，越小越平滑）
CMD_V_MAX_DT_S = 1.0  # 距上次发布目标超过该间隔则重置差分，避免间断后速度尖峰

# =============================================================================
# 机体系 / 世界系指令限幅（command.py，与 VLM 无关的硬钳制）
# =============================================================================

BODY_DELTA_MAX_M = 2.0  # 单次机体系 x_m/y_m/z_m 绝对值上限（米）
YAW_DELTA_MAX_DEG = 30.0  # 单次 yaw_deg 绝对值上限（度）
CMD_Z_MIN_M = 0.1  # 发布后世界系目标高度下限（米）
CMD_Z_MAX_M = 10.0  # 发布后世界系目标高度上限（米）

# =============================================================================
# 2D 几何控制（camera_geom：像素 → 世界 bearing → 位移 / yaw / z）
# =============================================================================

NORM_SCALE_2D = 1000.0  # VLM 归一化坐标满量程（0~1000 映射到相机原图宽高）
EXECUTOR_2D_MAX_DISTANCE_M = 1.0  # 水平前进距离上限（米）；随 3D 夹角衰减的峰值
BEARING_DEV_MAX_RAD = 1.5707963267948966  # 航点与标准方向 3D 夹角达到该值（默认 π/2）时 distance=0
Z_2D_REF_PITCH_DEG = -0.0  # 标准俯仰（度）：0=水平，-90=垂直向下；与机头水平方向配合算 pitch 差
YAW_2D_GAIN_DEG = 0.15  # 偏航 K：yaw_deg = K × degrees(yaw_航点 − yaw_标准)
Z_2D_GAIN_M = 0.3  # 高度 K：z_m_world = K × (世界系 pitch_航点 − pitch_标准)，单位 m/rad
BEARING_BODY_FLIP_LR = False  # 仅像素→机体系→世界系 bearing；后续控制只用世界系 bearing
EXECUTOR_2D_DEBUG_PRINT = False  # True 时在 executor_2d 打印 bearing/几何/指令中间量

# =============================================================================
# 2D 相机内参（固定使用本段，不订阅 camera_info）
# =============================================================================

# --- 真机 RealSense color（标定值）---
CAM_WIDTH_REAL = 640
CAM_HEIGHT_REAL = 480
CAM_FX_REAL = 607.3521118164062
CAM_FY_REAL = 607.6259765625
CAM_CX_REAL = 317.3368835449219
CAM_CY_REAL = 251.7844696044922

# --- 仿真 Gazebo realsense_camera 深度插件 color 通道（与深度同分辨率）---
# SDF: width=640 height=480 horizontal_fov=1.047198 rad
# fx = fy = W / (2*tan(hfov/2)) ≈ 554.256，cx=W/2 cy=H/2
CAM_WIDTH_SIM = 640
CAM_HEIGHT_SIM = 480
CAM_FX_SIM = 554.254691191187
CAM_FY_SIM = 554.254691191187
CAM_CX_SIM = 320.5
CAM_CY_SIM = 240.5

CAM_WIDTH = CAM_WIDTH_SIM if SIMULATE else CAM_WIDTH_REAL
CAM_HEIGHT = CAM_HEIGHT_SIM if SIMULATE else CAM_HEIGHT_REAL
CAM_FX = CAM_FX_SIM if SIMULATE else CAM_FX_REAL
CAM_FY = CAM_FY_SIM if SIMULATE else CAM_FY_REAL
CAM_CX = CAM_CX_SIM if SIMULATE else CAM_CX_REAL
CAM_CY = CAM_CY_SIM if SIMULATE else CAM_CY_REAL

# --- MPC 深度栅格避障用内参（固定配置，不订阅 camera_info）---
# 真机：深度 camera_info 标定（与 color 不同）
DEPTH_WIDTH_REAL = 640
DEPTH_HEIGHT_REAL = 480
DEPTH_FX_REAL = 391.0926513671875
DEPTH_FY_REAL = 391.0926513671875
DEPTH_CX_REAL = 323.90478515625
DEPTH_CY_REAL = 235.0640411376953

# --- 仿真 Gazebo depth camera_info K（640×480, plumb_bob, D≈0）---
# fx=fy=554.254691191187, cx=320.5, cy=240.5
DEPTH_WIDTH_SIM = 640
DEPTH_HEIGHT_SIM = 480
DEPTH_FX_SIM = 554.254691191187
DEPTH_FY_SIM = 554.254691191187
DEPTH_CX_SIM = 320.5
DEPTH_CY_SIM = 240.5

DEPTH_WIDTH = DEPTH_WIDTH_SIM if SIMULATE else DEPTH_WIDTH_REAL
DEPTH_HEIGHT = DEPTH_HEIGHT_SIM if SIMULATE else DEPTH_HEIGHT_REAL
DEPTH_FX = DEPTH_FX_SIM if SIMULATE else DEPTH_FX_REAL
DEPTH_FY = DEPTH_FY_SIM if SIMULATE else DEPTH_FY_REAL
DEPTH_CX = DEPTH_CX_SIM if SIMULATE else DEPTH_CX_REAL
DEPTH_CY = DEPTH_CY_SIM if SIMULATE else DEPTH_CY_REAL

# =============================================================================
# 2D Executor 预留 / 检测模式（当前主路径未使用，保留便于扩展）
# =============================================================================

MIN_DEPTH_M = 0.0  # 预留：反投影最小深度（米），当前 camera_geom 未读取

# =============================================================================
# command_pos 发布前避障（深度图 → 3D 点 + 线段硬安全距离）
# =============================================================================

ENABLE_COMMAND_POS_OBSTACLE_AVOIDANCE = True  # True=发布 /command_pos 前按深度障碍 SLSQP 微调目标 xyz
OA_OBS_DEPTH_MAX_M = 2.0  # 深度反投影障碍点最远距离（m），超过则忽略
OA_OBS_MIN_CLEARANCE_M = 0.5  # 线段 [当前位置, 修正目标] 与每个障碍点的最小距离（m）
OBS_GRID_N = 8  # 深度图降采样栅格边长 N，共 N×N 格
OBS_GRID_DEPTH_PERCENTILE = 2  # 每格深度分位数（%）；越小越保守、越偏近端障碍
OBS_GRID_MIN_VALID_DEPTH_M = 0.01  # 有效深度下限（m），过近/无效像素不参与
OBS_POINT_HISTORY_FRAMES = 2  # 障碍 3D 点历史帧数，用于多帧融合抗闪烁

# =============================================================================
# 非 move 子任务直接控制（direct_control：rotate_scan / stop）
# =============================================================================

SCAN_YAW_DELTA_DEG = 8.0  # rotate_scan 单次原地偏航增量（度）；右转为负、与 compact yaw 约定一致
