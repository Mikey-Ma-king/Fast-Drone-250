/**
 * ===========================================================================
 * traj_server.cpp — 轨迹跟踪主控制器 (无人机飞行指令生成器)
 * ===========================================================================
 *
 * 【在整个系统中的角色】
 *   本模块是整个控制管道的最后一环,也是飞控系统的"大脑"。它接收 MPC 规划
 *   的轨迹(或直接的狗位姿数据),进行坐标变换和 PID 控制,最终生成无人机
 *   位置/速度/加速度命令(/position_cmd),发送给底层的 px4ctrl 飞控节点执行。
 *
 *   traj_server 是系统中运行频率最高的节点之一:
 *   - cmdCallback @ 67 Hz (0.015s) — 发布位置指令
 *   - mpc_callback @ 100 Hz (0.01s) — MPC 轨迹插值
 *   - flag_and_hc14_process_callback @ 10 Hz (0.1s) — 模式切换逻辑
 *
 * 【三模式控制架构】
 *
 *   本模块的核心设计是"三模式控制":
 *
 *   Mode 0 (MPC 模式):
 *     - 控制来源: MPC.py 发布的 /drone2/planning/traj 轨迹
 *     - 适用场景: 无人机距离目标较远(>2m),需要避障或大范围机动
 *     - 控制策略: 从 MPC 轨迹中按时间插值期望状态(mpc_callback),
 *                 在狗体坐标系下做 PID 跟踪
 *     - Yaw: 无人机朝向狗(atan2 目标位置)
 *
 *   Mode 1 (精确跟踪模式):
 *     - 控制来源: dog_pos_processor 发布的 /dog_pos_processed 实时位姿
 *     - 适用场景: 无人机距离目标较近(<2m),需要精确跟踪狗的运动
 *     - 控制策略: 直接在狗体坐标系下做 PID + 前馈(速度+加速度+jerk)
 *     - Yaw: 与目标狗航向一致(方便相机对准)
 *
 *   Mode 2 (降落模式):
 *     - 控制来源: 同 Mode 1 + 额外的下降速度
 *     - 适用场景: 满足降落条件后自动触发
 *     - 控制策略: 同 Mode 1 + vz = 狗垂直速度 + land_vel(-0.6 m/s)
 *     - 触发条件: 速度差<0.3, 位置差<0.4, 角度差<30°, 持续 10 帧
 *     - 自动着陆: 光流高度<0.1m 且帧高差<0.5m, 持续 8 帧 → 飞控着陆
 *
 * 【狗体坐标系 PID 的原理】
 *
 *   不在世界系直接做 PID,而选择在狗体坐标系下做,原因:
 *   1. 狗的运动有方向性 — 狗头方向和狗侧方向的动力学特性完全不同
 *   2. dog_pos_processor 已经提供了准确的狗航向(yaw)估计
 *   3. 在狗体系下,可以独立调节"狗头方向"(x_body)和"狗侧方向"(y_body)
 *     的 PID 参数,获得更好的跟踪效果
 *
 *   变换: [error_world] → R(yaw) → [error_x_body, error_y_body]
 *         狗头前方 (x_body): 主要跟踪方向,PID 用了较大的 Ki(0.83)
 *         狗侧面   (y_body): 次要跟踪方向,Ki(0.4) 较小(狗侧移量有限)
 *
 * 【加速度和 Jerk 限制链】
 *
 *   为避免电机饱和和保护机械结构,输出命令经过三级限制:
 *   1. Jerk 限制: |Δaccel| ≤ max_jerk * dt (加速度变化率限制)
 *   2. 加速度限制: |accel| ≤ max_accel (最大加速度限制)
 *   3. 速度限制: |vel| ≤ [1.5, 1.5, 0.8] (最大速度限制)
 *   4. 位置变化率限制: 配套速度限制
 *
 * 【依赖项】
 *   - nav_msgs/Odometry, Path (ROS 标准消息)
 *   - quadrotor_msgs/PositionCommand, TakeoffLand (自定义消息)
 *   - Eigen (线性代数)
 *   - dog_pos_processor (经由 /dog_pos_processed 话题)
 *   - px4ctrl (经由 /position_cmd 话题)
 */

#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/TakeoffLand.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>

#include <geometry_msgs/PoseStamped.h>

#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <array>
#include <tuple>
#include <fstream>

#include "callback.h"

// ============================================================================
// 一、ROS 发布器和全局状态
// ============================================================================

ros::Publisher pos_cmd_pub_;       // 发布位置命令给 px4ctrl
ros::Publisher land_pub_;          // 发布降落命令给 px4ctrl
ros::Publisher mode_pub_;          // 发布模式切换信号给 MPC.py
int land_lock_timer = 0;           // 降落锁定计数器
ros::Publisher debug_pub_;         // 调试信息发布器
ros::Time heartbeat_time_;         // 心跳时间
ros::Publisher yaw_offset_pub;     // 发布 yaw offset 差值(降落时保存)

// ===== 命令限幅相关 =====
// 保存上一次发布的命令,用于加速度和 jerk 限制
// 原理: 限制两次命令之间的变化率,防止电机指令突变
Eigen::Vector3d last_cmd_velocity{0, 0, 0};
Eigen::Vector3d last_cmd_position{0, 0, 0};
Eigen::Vector3d last_cmd_acceleration{0, 0, 0};
bool last_cmd_initialized = false;

// ===== 距离日志 =====
// 记录无人机和目标的距离历史,用于离线分析
static std::ofstream distance_log_stream;
static const char kDistanceLogPath[] = "/home/pc/Fast-Drone-250/distance_log.txt";
static std::vector<std::tuple<double, double, double>> distance_history;  // {time, distance, height}

static inline void logDistance(double distance, double height) {
  if (!distance_log_stream.is_open()) {
    distance_log_stream.open(kDistanceLogPath, std::ios::app);
    if (!distance_log_stream.is_open()) {
      ROS_WARN_STREAM("Failed to open distance log file: " << kDistanceLogPath);
      return;
    }
  }
  const double now_sec = ros::Time::now().toSec();
  distance_history.emplace_back(now_sec, distance, height);
  distance_log_stream << std::fixed << now_sec << "," << distance << std::endl;
}

// ===== 三模式控制标志 =====
// triger_mode 的含义:
//   -1: 未初始化(等待)
//    0: MPC 模式(使用 /drone2/planning/traj)
//    1: 精确跟踪模式(使用 /dog_pos_processed)
//    2: 降落模式(同精确跟踪 + 下降速度)
int triger_mode = -1;

// ===== VINS 无人机状态 =====
Eigen::Vector3d vins_p{0,0,0};     // 无人机世界系位置
Eigen::Vector3d vins_v{0,0,0};     // 无人机世界系速度
double vins_yaw = 0;               // 无人机 yaw 角

// ===== 目标视觉状态 (/target_ekf_odom) =====
Eigen::Vector3d target_p{0,0,0};   // 目标世界系位置
Eigen::Vector3d target_v{0,0,0};   // 目标世界系速度
double target_dog_yaw = 0;         // 目标世界系 yaw
int last_target_timer = 0;
unsigned int target_count = 0;
unsigned int last_target_count = 0;
unsigned int last_precise_target_count = 0;
bool target_receive = false;       // 是否正在接收目标数据
int last_target_loss_timer = 0;
unsigned int last_target_loss_count = 0;

// ===== 狗处理后位姿 (/dog_pos_processed) =====
bool hc14_dog_pos_received = false;
unsigned int hc14_dog_pos_count = 0;
unsigned int last_hc14_dog_pos_count = 0;
int last_hc14_dog_pos_timer = 0;

// 处理后的狗位姿(世界系):
Eigen::Vector3d hc14_dog_vel{0,0,0};        // 速度
Eigen::Vector3d hc14_dog_pos{0,0,0};        // 位置
double hc14_dog_yaw = 0.0;                   // yaw 角
double hc14_dog_yaw_rate = 0.0;              // 狗旋转角速度(来自 dog_pos_processor)
Eigen::Vector2d hc14_dog_acc{0,0};           // 狗加速度(世界系 xy,来自 dog_pos_processor)
Eigen::Vector2d hc14_dog_jerk_filtered{0,0}; // 滤波后的狗 jerk(加速度的导数)
Eigen::Vector2d last_hc14_dog_acc{0,0};      // 上一帧狗加速度(用于计算 jerk)
ros::Time last_hc14_dog_acc_time;
bool hc14_dog_acc_initialized = false;
bool hc14_offset_yaw_ready = false;          // dog_pos_processor 的 yaw offset 是否 ready
bool hc14_offset_pos_ready = false;          // dog_pos_processor 的 pos offset 是否 ready

double command_pos_yaw = 0.0;
bool command_pos_received = false;

// ============================================================================
// 二、控制参数
// ============================================================================

double kp = 1.2;                       // 世界系 PID 的比例增益(备用)
double tracking_dist_ = 1.5;           // 跟踪触发距离阈值(m)
double target_receive_triger = 1;
double mode_vins_z = 0;                // 降落模式下 VINS 高度参考
double mode_vins_vel_z = 0;            // 降落模式下 VINS 垂直速度参考
bool reflight_complete = false;        // Mode 1 中重飞到指定高度带是否完成
ros::Time target_lost_time;
Eigen::Vector3d target_lost_p = {0,0,0};
Eigen::Vector3d target_lost_v = {0,0,0};
Eigen::Vector3d last_target_v = {0,0,0};

// ===== AOA 相关(备用) =====
double AOA_x = 10;
double AOA_w = 0;
double flow_z = -1;                    // 光流高度(m)
double last_error_x = 0;
double last_error_y = 0;
double last_error_z = 0;
double alpha = 0.3;
double filtered_d_error_x = 0;
double filtered_d_error_y = 0;
double filtered_d_error_z = 0;
double intergral_x = 0;
double intergral_y = 0;
double intergral_z = 0;

// ===== PID 积分项 =====
// 世界系积分项(Mode 0/MPC 模式使用):
double intergral_targetz = 0;          // z 轴积分
double intergral_targetx = 0;          // x 轴积分(世界系)
double intergral_targety = 0;          // y 轴积分(世界系)
// 狗体坐标系积分项(Mode 1/2 使用):
double intergral_targetx_body = 0;     // x_body 积分(狗头方向)
double intergral_targety_body = 0;     // y_body 积分(狗侧方向)
double last_error_targetx = 0;
double last_error_targety = 0;
double last_error_targetz = 0;

// 均值滤波器: 对跟踪误差做滑动窗口均值,抑制高频噪声
std::deque<double> error_targetx_buffer;
std::deque<double> error_targety_buffer;
std::deque<double> error_targetz_buffer;

int target_land_flag = 0;

// ===== 偏航角控制参数 =====
double yaw_kp = 0.3;                   // 偏航 P 增益
double max_yaw_rate = 0.8;             // 最大偏航角速度(rad/s)

// 角速度前馈增益: 将狗旋转速度转换为无人机位置/速度补偿
// 原理: 狗在转弯时,如果不加前馈,无人机会因为纯位置反馈而滞后
double yaw_rate_pos_gain = 0.0;        // 侧面位置前馈增益
double yaw_rate_vel_gain = 0.0;        // 侧面速度前馈增益
double yaw_rate_pos_gain_forward = 0.0; // 前方位置前馈增益
double yaw_rate_vel_gain_forward = 0.0; // 前方速度前馈增益

// ===== 加速度和 Jerk 限制参数 =====
// 硬限制: 保护电机,防止指令突变导致控制失稳或电机饱和
const double max_accel = 1.2;          // 最大加速度(m/s²)
const double max_jerk = 100.0;         // 最大 jerk(m/s³)
const double accel_dt = 0.01;          // 控制周期(秒)
const double max_accel_z = 1.0;        // z 方向最大加速度(m/s²)
const double land_vel = -0.6;          // 降落速度(m/s,负表示下降)


// ============================================================================
// 三、Mode 1/2 PID 参数 (精确跟踪模式,狗体坐标系)
// ============================================================================
// 这些参数在狗体坐标系下工作:
//   x_body = 狗头方向(前方)
//   y_body = 狗侧面方向(左侧)

// ----- X 轴 (狗头方向) PID -----
double x_p = 0.3;                        // 比例增益
double x_i = 0.83;                       // 积分增益(较大,用于消除稳态误差)
double x_d = 0.0;                        // 微分增益(当前为 0,速度前馈替代了 D 的作用)
double x_d_max = 0.12;                   // 微分项上限(限制噪声放大)
double v_offset_x = 0.0;                 // 速度偏移(手动微调用)
double integral_limit_x = 0.7;           // 积分项上限(防 windup)
double integral_decay = 0.0;             // 积分衰减系数(1/s),防止积分 windup
double acc_gain_x = 0.0;                 // 加速度前馈增益(当前关闭)
double jerk_filter_alpha_x = 0.02;       // jerk 低通滤波系数(0=不滤波, 1=完全信任新值)
double jerk_gain_x = 0.0;                // jerk 前馈增益(当前关闭)
double acc_feedforward_sat_threshold_x = 0.1;  // 加速度前馈饱和阈值(小于此值线性)
double acc_feedforward_sat_value_x = 1.0;      // 加速度前馈饱和值
double jerk_feedforward_sat_threshold_x = 0.3;  // jerk 前馈饱和阈值
double jerk_feedforward_sat_value_x = 1.0;      // jerk 前馈饱和值

// ----- Y 轴 (狗侧方向) PID -----
// 与 X 轴参数结构相同,但积分增益较小(因为侧面运动量小,不需要大积分)
double y_p = 0.3;
double y_i = 0.83;
double y_d = 0.0;
double y_d_max = 0.12;
double v_offset_y = 0.0;
double integral_limit_y = 0.4;           // Y 轴积分限制比 X 小(侧面不需要大积分)
double acc_gain_y = 0.0;
double jerk_filter_alpha_y = 0.02;
double jerk_gain_y = 0.0;
double acc_feedforward_sat_threshold_y = 0.1;
double acc_feedforward_sat_value_y = 1.0;
double jerk_feedforward_sat_threshold_y = 0.3;
double jerk_feedforward_sat_value_y = 1.0;

double z_p = 0.1;
double z_i = 0.0;
double z_d = 0.0;
double z_d_max = 0.1;
double integral_limit_z = 0.1;

// ===== Mode 0 PID 参数 (MPC 模式,世界坐标系) =====
// 与精确跟踪模式参数独立,可分别调优
double mpc_x_p = 0.3;
double mpc_x_i = 0.83;
double mpc_x_d = 0.0;
double mpc_x_d_max = 0.12;

double mpc_y_p = 0.3;
double mpc_y_i = 0.83;
double mpc_y_d = 0.0;
double mpc_y_d_max = 0.12;

double mpc_z_p = 0.3;
double mpc_z_i = 0.0;
double mpc_z_d = 0.0;
double mpc_z_d_max = 0.12;

// ===== 位置控制 PID 参数 (狗体坐标系,备用/未启用) =====
double pos_x_p = 0.0;
double pos_x_i = 0.0;
double pos_x_d = 0.0;
double pos_x_d_max = 0.1;
double pos_integral_limit_x = 0.5;

double pos_y_p = 0.0;
double pos_y_i = 0.0;
double pos_y_d = 0.0;
double pos_y_d_max = 0.1;
double pos_integral_limit_y = 0.5;

double mpc_pos_x_p = 0.0;
double mpc_pos_x_i = 0.0;
double mpc_pos_x_d = 0.0;
double mpc_pos_x_d_max = 0.1;

double mpc_pos_y_p = 0.0;
double mpc_pos_y_i = 0.0;
double mpc_pos_y_d = 0.0;
double mpc_pos_y_d_max = 0.1;

#ifdef SIMULATE
v_offset_x = 0.0;
v_offset_y = 0.0;
#endif

// ===== 高度限制区间 [下界, 上界] =====
std::vector<double> land_height_limit = {1.2, 1.5};

std::vector<Eigen::Vector3d> target_p_list, target_v_list;

// ===== MPC 轨迹相关 =====
ros::Time traj_start_time;
bool traj_initialized = false;                              // MPC 轨迹是否已初始化
std::vector<Eigen::Vector3d> trajectory_points;             // 位置轨迹点序列
std::vector<Eigen::Vector3d> trajectory_v_points;           // 速度轨迹点序列
std::vector<Eigen::Vector3d> trajectory_a_points;           // 加速度轨迹点序列

Eigen::Vector3d mpc_p{0,0,0};    // MPC 轨迹插值后的期望位置
Eigen::Vector3d mpc_v{0,0,0};    // MPC 轨迹插值后的期望速度
Eigen::Vector3d mpc_a{0,0,0};    // MPC 轨迹插值后的期望加速度

int BPNN_count = 60;
ros::Time traj_sub_time;          // 轨迹接收时间

double land_timer = 0;            // 降落条件保持计数器

// ===== BLSC 滑模控制器参数 =====
// BLSC (Boundary Layer Sliding Control): 基于滑模变结构 + 边界层的鲁棒控制器
// 滑模面: s = vel_error + λ * pos_error
// 边界层: 在滑模面附近做线性化,消除高频抖振
const double blsc_lambda = 1.2;          // 收敛率(λ 越大收敛越快)
const double blsc_boundary_layer = 0.5;  // 边界层厚度(越大抖振越小,但精度降低)
const double blsc_max_horiz_vel = 1.5;   // 水平速度上限(m/s)
const double blsc_max_vert_vel = 0.8;    // 垂直速度上限(m/s)

using namespace Eigen;


// ============================================================================
// 四、PID 参数结构体
// ============================================================================

struct PIDParams {
  double Kp;
  double Ki;
  double Kd;
};


// ============================================================================
// 五、BP 神经网络自适应 PID 控制器 (BPNeuralNetwork + BPNeuralNetworkPIDController)
// ============================================================================
// 注意: 以下 BP 神经网络类已定义但实际未启用(当前使用固定参数的传统 PID)。
// 保留代码供未来自适应控制实验使用。
//
// 原理:
//   BP 神经网络在线调节 PID 的三个参数(Kp, Ki, Kd),
//   输入: [误差, 积分, 微分, 目标速度]
//   输出: [ΔKp, ΔKi, ΔKd]
//   训练目标: 减少当前误差(误差→0, 积分→0, 微分→0)
//
// 网络结构: 4 输入 → 5 隐含层 (sigmoid) → 3 输出 (sigmoid)

class BPNeuralNetwork {
  private:
      int input_size, hidden_size, output_size;
      double learning_rate;
      // 权重矩阵(双层全连接)
      std::vector<std::vector<double>> input_to_hidden_weights;   // [input_size × hidden_size]
      std::vector<std::vector<double>> hidden_to_output_weights;  // [hidden_size × output_size]
      std::vector<double> hidden_bias, output_bias;
      std::vector<double> hidden_output, network_output;

  public:
      BPNeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate = 2)
          : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) {
          // 随机初始化权重 [0, 1]
          input_to_hidden_weights = std::vector<std::vector<double>>(input_size, std::vector<double>(hidden_size));
          hidden_to_output_weights = std::vector<std::vector<double>>(hidden_size, std::vector<double>(output_size));
          hidden_bias = std::vector<double>(hidden_size, 0.0);
          output_bias = std::vector<double>(output_size, 0.0);
          hidden_output = std::vector<double>(hidden_size, 0.0);
          network_output = std::vector<double>(output_size, 0.0);

          for (int i = 0; i < input_size; ++i)
              for (int j = 0; j < hidden_size; ++j)
                  input_to_hidden_weights[i][j] = (rand() % 1000) / 1000.0;
          for (int i = 0; i < hidden_size; ++i)
              for (int j = 0; j < output_size; ++j)
                  hidden_to_output_weights[i][j] = (rand() % 1000) / 1000.0;
      }

      // 激活函数: sigmoid(x) = 1/(1+e^(-x))
      // 将任意实数映射到 (0,1),提供非线性能力
      double sigmoid(double x) {
          return 1.0 / (1.0 + exp(-x));
      }

      // sigmoid 的导数: σ'(x) = σ(x)·(1-σ(x))
      double sigmoid_derivative(double x) {
          return x * (1.0 - x);
      }

      // 前向传播: input → hidden → output
      std::vector<double> forward(const std::vector<double>& input) {
          // 输入→隐藏层: h = σ(W1·x + b1)
          for (int i = 0; i < hidden_size; ++i) {
              hidden_output[i] = 0.0;
              for (int j = 0; j < input_size; ++j)
                  hidden_output[i] += input[j] * input_to_hidden_weights[j][i];
              hidden_output[i] += hidden_bias[i];
              hidden_output[i] = sigmoid(hidden_output[i]);
          }

          // 隐藏层→输出: y = σ(W2·h + b2)
          for (int i = 0; i < output_size; ++i) {
              network_output[i] = 0.0;
              for (int j = 0; j < hidden_size; ++j)
                  network_output[i] += hidden_output[j] * hidden_to_output_weights[j][i];
              network_output[i] += output_bias[i];
              network_output[i] = sigmoid(network_output[i]);
          }

          return network_output;
      }

      // 反向传播: 用误差 BP 算法更新权重
      // 目标: 使输出误差最小化
      void backward(const std::vector<double>& input, const std::vector<double>& target) {
          // 输出层误差: δ_out = (target - output)  (简化,不含 sigmoid 导数)
          std::vector<double> output_errors(output_size);
          for (int i = 0; i < output_size; ++i)
              output_errors[i] = target[i] - network_output[i];

          // 隐藏层误差: δ_hidden = (Σ δ_out · W2) ⊙ σ'(h)
          std::vector<double> hidden_errors(hidden_size);
          for (int i = 0; i < hidden_size; ++i) {
              hidden_errors[i] = 0.0;
              for (int j = 0; j < output_size; ++j)
                  hidden_errors[i] += output_errors[j] * hidden_to_output_weights[i][j];
              hidden_errors[i] *= sigmoid_derivative(hidden_output[i]);
          }

          // 更新输出层权重: ΔW2 = lr · δ_out · hᵀ
          for (int i = 0; i < output_size; ++i) {
              for (int j = 0; j < hidden_size; ++j) {
                  hidden_to_output_weights[j][i] += learning_rate * output_errors[i] * hidden_output[j];
              }
              output_bias[i] += learning_rate * output_errors[i];
          }

          // 更新隐藏层权重: ΔW1 = lr · δ_hidden · xᵀ
          for (int i = 0; i < hidden_size; ++i) {
              for (int j = 0; j < input_size; ++j) {
                  input_to_hidden_weights[j][i] += learning_rate * hidden_errors[i] * input[j];
              }
              hidden_bias[i] += learning_rate * hidden_errors[i];
          }
      }
};

/**
 * BPNeuralNetworkPIDController — 神经网络自适应 PID
 *
 * 核心思路: 用 BP 神经网络在线调整 PID 的三个参数,
 *          使控制器适应目标运动特性变化。
 *
 * 输入向量: [error, integral, delta_error, target_v] (4 维)
 * 输出向量: [ΔKp, ΔKi, ΔKd] (3 维)
 *
 * 训练策略(简单的梯度下降):
 *   目标 = [-error*0.1, -integral*0.1, -delta_error*0.1]
 *   含义: 让输出朝着减少误差的方向更新
 */
class BPNeuralNetworkPIDController {
  private:
      double Kp, Ki, Kd;
      BPNeuralNetwork nn;
      double prev_error = 0;
      double integral = 0;

      double clamp(double value, double min_val, double max_val) {
          return std::max(min_val, std::min(value, max_val));
      }

  public:
      BPNeuralNetworkPIDController(double Kp_init, double Ki_init, double Kd_init)
          : Kp(Kp_init), Ki(Ki_init), Kd(Kd_init), nn(4, 5, 3) {}

      PIDParams compute(double error, double delta_error, double target_v) {
          // 积分累加(带限幅)
          integral = clamp(integral + 0.01*error, -100.0, 100.0);

          // 神经网络前向传播: 输入误差特征 → 输出 PID 参数调整量
          std::vector<double> input = { error, integral, delta_error, target_v };
          std::vector<double> output = nn.forward(input);

          // 用神经网络输出调整 PID 参数(带范围限制)
          Kp = clamp(Kp + output[0] * 0.06, 0.0, 1.5);
          Ki = clamp(Ki + output[1] * 0.05, 0.0, 3.0);
          Kd = clamp(Kd + output[2] * 0.1, 0.0, 1.5);

          // 训练目标: 朝减少误差方向更新
          std::vector<double> target(3, 0.0);
          target[0] = -error * 0.1;
          target[1] = -integral * 0.1;
          target[2] = -delta_error * 0.1;

          // 反向传播更新网络权重
          nn.backward(input, target);

          // 计算控制信号(注: 这里 Ki*error 实际上是 Ki*积分的形式,但代码用 Ki*error)
          double control_signal = Kp * error + Ki * error + Kd * delta_error;
          return {Kp, Ki, Kd};
      }

      PIDParams get_parm() {
        return {Kp, Ki, Kd};
      }
};

// 创建 BP 神经网络 PID 实例(当前未启用,替代方案见 cmdCallback 中的传统 PID)
BPNeuralNetworkPIDController pid_x(0.6, 0.0, 0.3), pid_y(0.6, 0.0, 0.3), pid_z(0.3, 0.1, 0.01);


// ============================================================================
// 六、光滑微分误差计算器 (SmoothedDeltaError)
// ============================================================================
// 用于计算误差的平滑导数,替代传统的离散差分。
// 使用 5 点四次差分 + 低通滤波器,在噪声抑制和响应速度之间取得平衡。

class SmoothedDeltaError {
  private:
      const double delta_t;
      std::deque<double> error_history;
      const int window_size = 5;
      // 五点四次差分系数: [-2, -1, 0, 1, 2] / (10 * dt)
      // 这是数值微分的 Lagrange 公式在 5 点上的特例
      const std::vector<double> coefficients = { -2.0, -1.0, 0.0, 1.0, 2.0 };

      // 低通滤波器参数
      double alpha;           // 一阶 RC 低通滤波器系数
      double filtered_delta;  // 上一帧滤波输出
      double error;
      double last_error;
      double last_last_error;

      const double max_delta_error = 0.5;  // 微分输出限幅

  public:
      /**
       * @param dt 控制周期(秒),默认 0.015
       * @param lpf_cutoff_freq 低通截止频率(Hz),默认 12Hz
       *
       * 低通滤波原理: 一阶离散 RC 滤波器
       *   alpha = delta_t / (delta_t + RC), RC = 1/(2π*f_cutoff)
       *   output = alpha * input + (1-alpha) * last_output
       */
      SmoothedDeltaError(double dt = 0.015, double lpf_cutoff_freq = 12.0)
          : delta_t(dt), filtered_delta(0.0) {
          if (delta_t <= 0) {
              throw std::invalid_argument("delta_t must be positive");
          }
          double rc = 1.0 / (2 * M_PI * lpf_cutoff_freq);
          alpha = delta_t / (delta_t + rc);
      }

      /**
       * update_error — 输入新误差,返回误差的平滑时间导数
       *
       * 步骤:
       *   1. 误差自身先做平滑: error = 0.3*new + 0.4*last + 0.3*last_last
       *   2. 维护 5 点滑动窗口
       *   3. 计算四点差分: Σ(coeff[i] * error[i]) / (10 * dt)
       *   4. 低通滤波: output = α*raw + (1-α)*last_output
       *   5. 输出限幅在 [-0.5, 0.5]
       */
      double update_error(double new_error) {
          // 误差平滑: 带滞后的三点加权平均(比简单的新旧平均更稳定)
          error = 0.3 * new_error + 0.4 * last_error + 0.3 * last_error;
          last_last_error = last_error;
          last_error = error;

          // 维护滑动窗口
          error_history.push_back(error);
          if (error_history.size() > window_size) {
              error_history.pop_front();
          }
          if (error_history.size() < window_size) {
              return 0.0;  // 数据不足时返回 0
          }

          // 四点差分(五点 Lagrange 数值微分)
          double raw_delta = calculate_quad_differential();

          // 低通滤波
          filtered_delta = alpha * raw_delta + (1 - alpha) * filtered_delta;
          filtered_delta = std::max(-max_delta_error, std::min(filtered_delta, max_delta_error));
          return filtered_delta;
      }

      void reset() {
          error_history.clear();
          filtered_delta = 0.0;
      }

  private:
      /**
       * 四点加权微分公式:
       *   f'(t) ≈ Σ(w_i * f(t-i*dt)) / (10 * dt)
       * 其中 w = [-2, -1, 0, 1, 2]
       * 这等价于用二阶多项式拟合 5 个点后求导
       */
      double calculate_quad_differential() {
          double weighted_sum = 0.0;
          for (size_t i = 0; i < window_size; ++i) {
              weighted_sum += coefficients[i] * error_history[i];
          }
          return weighted_sum / (10.0 * delta_t);
      }
};

// 为 x/y/z 三个轴分别创建微分器
SmoothedDeltaError diff_x, diff_y, diff_z;


// ============================================================================
// 七、Saturation 函数 + BLSC 滑模控制器
// ============================================================================

/**
 * sat — Saturation 函数(含边界层)
 *
 * 标准 sat 函数:
 *   sat(s, ε) = s/ε          if |s| ≤ ε (边界层内线性)
 *             = sign(s)       if |s| > ε (边界层外饱和)
 *
 * 边界层的作用: 在滑模面附近做线性化,消除 sign 函数引起的抖振。
 *   ε 越大 → 抖振越小 → 但跟踪精度越低
 *   ε 越小 → 精度越高 → 但抖振风险增加
 */
double sat(double value, double epsilon) {
  if (std::abs(value) > epsilon)
      return (value > 0 ? 1.0 : -1.0);
  else
      return value / epsilon;  // 边界层内线性化
}

/**
 * BLSCController — 边界层滑模控制器 (Boundary Layer Sliding Control)
 *
 * 理论基础:
 *   滑模控制的核心思想是将系统状态"驱动"到滑模面 s=0 上,
 *   然后在滑模面上滑动到目标。
 *
 *   本实现使用:
 *     滑模面: s = vel_error + λ * pos_error
 *     控制律: u = v_des + λ * (p_des - p) - sat(s/ε)
 *
 *   解释:
 *     - v_des + λ*pos_error: 前馈 + 比例项
 *     - sat(s): 鲁棒项,当 s 很大时饱和(等于±1),当 s 很小时线性
 *     - 这种结构在理论上等价于 PI + 鲁棒项
 *
 * 与 PID 的关系:
 *   - 传统 PID: u = Kp*e + Ki*∫e + Kd*ė
 *   - BLSC:     u = v_des + λ*(p_des-p) - sat((v_des-v) + λ*(p_des-p))
 *   可以看出 BLSC 就是 PID 的另一种形式,但具有更好的鲁棒性保证
 */
void BLSCController(
  const Eigen::Vector3d& current_pos,
  const Eigen::Vector3d& current_vel,
  const Eigen::Vector3d& desired_pos,
  const Eigen::Vector3d& desired_vel,
  Eigen::Vector3d& output_vel) {

  // 位置误差和速度误差
  Eigen::Vector3d pos_error = desired_pos - current_pos;
  Eigen::Vector3d vel_error = desired_vel - current_vel;

  // 滑模面: s = vel_error + λ * pos_error
  // 当 s→0 时, vel_error = -λ * pos_error, 即指数收敛
  Eigen::Vector3d s = vel_error + blsc_lambda * pos_error;

  // Saturation 项(逐元素处理,因为三个轴的饱和情况可能不同)
  Eigen::Vector3d sat_s;
  for (int i = 0; i < 3; ++i) {
      sat_s[i] = sat(s[i], blsc_boundary_layer);
  }

  // 控制律: u = v_des + λ*(p_des-p) - sat(s)
  // 这是滑模控制的等效控制 + 切换控制
  output_vel = desired_vel + blsc_lambda * pos_error - sat_s;

  // 速度限幅
  output_vel.x() = std::max(-blsc_max_horiz_vel, std::min(output_vel.x(), blsc_max_horiz_vel));
  output_vel.y() = std::max(-blsc_max_horiz_vel, std::min(output_vel.y(), blsc_max_horiz_vel));
  output_vel.z() = std::max(-blsc_max_vert_vel, std::min(output_vel.z(), blsc_max_vert_vel));
}


// ============================================================================
// 八、轨迹可视化器 (TrajectoryVisualizer) + 速度估计器 (VelocityEstimator)
// ============================================================================

class TrajectoryVisualizer {
    public:
        explicit TrajectoryVisualizer(const std::string& frame_id = "world") : frame_id_(frame_id) {}

        void visualizeTraj(const Eigen::MatrixXd& position_array, double dt, const std::string& topic = "traj") {
            publishPath(position_array, topic);
            publishWaypoints(position_array, topic + "_wayPts");
        }

    private:
        ros::NodeHandle nh_;
        std::string frame_id_;
        std::unordered_map<std::string, ros::Publisher> publishers_;

        template<typename T>
        ros::Publisher& getPublisher(const std::string& topic, uint32_t queue_size = 10) {
            if (publishers_.find(topic) == publishers_.end()) {
                publishers_[topic] = nh_.advertise<T>(topic, queue_size);
            }
            return publishers_[topic];
        }

        void publishPath(const Eigen::MatrixXd& positions, const std::string& topic) {
            nav_msgs::Path path_msg;
            path_msg.header.frame_id = frame_id_;
            path_msg.header.stamp = ros::Time::now();

            for (int i = 0; i < positions.rows(); ++i) {
                geometry_msgs::PoseStamped pose;
                pose.header = path_msg.header;
                pose.pose.position.x = positions(i, 0);
                pose.pose.position.y = positions(i, 1);
                pose.pose.position.z = positions(i, 2);
                path_msg.poses.push_back(pose);
            }

            getPublisher<nav_msgs::Path>(topic).publish(path_msg);
        }

        void publishWaypoints(const Eigen::MatrixXd& positions, const std::string& topic) {
            sensor_msgs::PointCloud2 cloud_msg;
            cloud_msg.header.frame_id = frame_id_;
            cloud_msg.header.stamp = ros::Time::now();

            cloud_msg.height = 1;
            cloud_msg.width = positions.rows();
            cloud_msg.is_bigendian = false;
            cloud_msg.is_dense = true;

            sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            modifier.resize(positions.rows());

            sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

            for (int i = 0; i < positions.rows(); ++i, ++iter_x, ++iter_y, ++iter_z) {
                *iter_x = static_cast<float>(positions(i, 0));
                *iter_y = static_cast<float>(positions(i, 1));
                *iter_z = static_cast<float>(positions(i, 2));
            }

            getPublisher<sensor_msgs::PointCloud2>(topic).publish(cloud_msg);
        }
};

/**
 * VelocityEstimator — 从连续位置测量估计速度
 *
 * 简单差分估计: v = (p_new - p_old) / dt
 *   超时保护: dt > 0.5s 时重置(认为数据中断)
 */
class VelocityEstimator {
  private:
      Eigen::Vector3d last_position;
      ros::Time last_time;
      bool initialized;
      double timeout_threshold;  // 0.5s 超时清除

  public:
      VelocityEstimator(double timeout_sec = 0.5)
          : initialized(false), timeout_threshold(timeout_sec) {
          last_position.setZero();
      }

      Eigen::Vector3d update(const Eigen::Vector3d& new_position, const ros::Time& current_time) {
          if (!initialized) {
              last_position = new_position;
              last_time = current_time;
              initialized = true;
              return Eigen::Vector3d::Zero();
          }

          double dt = (current_time - last_time).toSec();

          if (dt <= 1e-6 || dt > timeout_threshold) {
              // 时间无效或超时,重置状态
              last_position = new_position;
              last_time = current_time;
              return Eigen::Vector3d::Zero();
          }

          Eigen::Vector3d velocity = (new_position - last_position) / dt;

          last_position = new_position;
          last_time = current_time;
          return velocity;
      }

      void reset() {
          initialized = false;
          last_position.setZero();
          last_time = ros::Time(0);
      }
};

VelocityEstimator drone_v_estimator, target_v_estimator;


// ============================================================================
// 九、目标轨迹预测 + 轨迹状态插值 (辅助函数)
// ============================================================================

/**
 * predictTargetTrajectory — 匀速直线预测(简化版,用于 MPC 轨迹的简单预测)
 *
 * 此函数在 traj_server 中定义但主要被 MPC.py 中的同名函数替代。
 * 保留作为备用方案。
 */
Eigen::MatrixXd predictTargetTrajectory(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, int N, double dt) {
      Eigen::MatrixXd trajectory(N + 1, 6);
      for (int t = 0; t <= N; ++t) {
          trajectory.block<1,3>(t, 0) = pos + vel * (t * dt);
          trajectory.block<1,3>(t, 3) = vel;
      }
      return trajectory;
}

/**
 * trajGetState — 二次插值从 MPC 轨迹中获取状态
 *
 * 与 MPC.py 的 traj_get_state 对应,但使用二次插值代替线性插值。
 *
 * 二次插值公式 (Lagrange 三点):
 *   f(α) = (1/2)α(α-1)f₀ - (α+1)(α-1)f₁ + (1/2)(α+1)αf₂
 *   其中 0≤α<1, f₀,f₁,f₂ 是三个连续采样点
 */
std::pair<Eigen::VectorXd, Eigen::Vector3d> trajGetState(const std::vector<Eigen::VectorXd>& x_traj,
                                                           const std::vector<Eigen::VectorXd>& u_traj,
                                                           double future_time, double dt) {
      double steps = future_time / dt;
      int idx_base = static_cast<int>(std::floor(steps));
      double alpha = steps - idx_base;

      idx_base = std::min(std::max(0, idx_base), static_cast<int>(x_traj.size()) - 3);

      const Eigen::VectorXd& x0 = x_traj[idx_base];
      const Eigen::VectorXd& x1 = x_traj[idx_base + 1];
      const Eigen::VectorXd& x2 = x_traj[idx_base + 2];

      // 二次插值
      Eigen::VectorXd state = (0.5 * alpha * (alpha - 1.0)) * x0
                            + (0.5 * alpha * (alpha + 1.0) - 0.5 * (alpha - 1.0) * (alpha + 1.0)) * x1
                            + (0.5 * (alpha - 1.0) * alpha) * x2;

      int accel_idx = std::min(idx_base, static_cast<int>(u_traj.size()) - 1);
      Eigen::Vector3d accel = u_traj[accel_idx].segment<3>(0);

      return {state, accel};
}


// ============================================================================
// 十、核心 — 位置命令生成回调 (cmdCallback @ 67Hz)
// ============================================================================
// 这是本节点的核心函数,每个控制周期执行一次。
//
// 主要流程:
//   1. 模式预检: 确认 triger_mode 有效
//   2. 目标获取: 根据模式选择 MPC 轨迹插值结果或实时狗位姿
//   3. Yaw 控制: 计算期望 yaw 角(朝向狗 或 与狗同向)
//   4. 坐标变换: 世界系误差 → 狗体坐标系 (R(yaw) 旋转)
//   5. 狗体系 PID: 独立计算 vx_body, vy_body, vz
//   6. 前馈补偿: 加速度前馈 + jerk 前馈 (Mode 1/2)
//   7. 逆坐标变换: 狗体系速度/位置 → 世界系
//   8. 速度/加速度/Jerk 限幅: 保护电机和结构
//   9. 积分更新(带衰减): 防止 windup
//  10. 调试信息发布 + 位置命令发布

void cmdCallback(const ros::TimerEvent &e) {
  // if (!(hc14_dog_pos_received && hc14_offset_yaw_ready))
  //   return;
  if (triger_mode != -1 && triger_mode != 0 && triger_mode != 1 && triger_mode != 2 && triger_mode != -2)
    return;
  if (triger_mode == -1)
    return;
  if ((triger_mode == 0 || triger_mode == -2) && !traj_initialized)
    return;
  // Mode 1/2: 需要狗位姿数据 ready
  if ((triger_mode == 1 || triger_mode == 2) && !(hc14_dog_pos_received && hc14_offset_yaw_ready))
    return;

  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = 0;

  double targetx, targety, targetz;          // 目标世界系位置
  double target_vx, target_vy, target_vz;    // 目标世界系速度
  double error_targetx, error_targety, error_targetz;  // 世界系跟踪误差
  double target_yaw;
  double angle_diff = 0;


  // MPC/agent 模式下的目标 yaw
  if (triger_mode == -2) {
    target_yaw = command_pos_yaw;
  } else {
    target_yaw = hc14_dog_yaw;
  }

  static int yaw_mode = 0;
  // yaw_mode 的含义:
  //   0: 无人机朝向狗 (atan2 目标位置)
  //   1: 无人机与狗同向 (跟随狗的航向)
  //   2: 保持当前 yaw (不旋转, Mode 2 降落时使用)
  if (triger_mode == 2){
    // yaw_mode = 2;
    yaw_mode = 1;
  } else if (triger_mode == -2) {
  // agent 模式：始终跟随 /command_pos 的 yaw
    yaw_mode = 1;
  } else if (triger_mode == 0 && (Eigen::Vector2d(vins_p.x() - hc14_dog_pos.x(), vins_p.y() - hc14_dog_pos.y()).norm() > 5.0)){
    yaw_mode = 0;
  } else if (Eigen::Vector2d(vins_p.x() - hc14_dog_pos.x(), vins_p.y() - hc14_dog_pos.y()).norm() < 3.0){
    yaw_mode = 1;    // 距离 < 3m: 与狗同向
  }

  // 计算 yaw 误差(当前 yaw 和期望 yaw 的差)
  if (yaw_mode == 0) {
    // 朝向狗: yaw_diff = atan2(dy, dx) - vins_yaw
    double yaw_to_dog = std::atan2(hc14_dog_pos.y() - vins_p.y(), hc14_dog_pos.x() - vins_p.x());
    angle_diff = yaw_to_dog - vins_yaw;
  } else if (yaw_mode == 1) {
    // 与狗同向: yaw_diff = dog_yaw - vins_yaw
    angle_diff = target_yaw - vins_yaw;
  } else if (yaw_mode == 2) {
    angle_diff = 0.0;  // 保持当前朝向
  }

  // Yaw 误差归一化到 [-π, π]
  if (angle_diff > M_PI) {
    angle_diff -= 2 * M_PI;
  } else if (angle_diff < -M_PI) {
    angle_diff += 2 * M_PI;
  }

  if ((triger_mode == 0 || triger_mode == -2)) {
    target_vx = mpc_v.x();
    target_vy = mpc_v.y();
    target_vz = mpc_v.z();

    targetx = mpc_p.x();
    targety = mpc_p.y();
    targetz = mpc_p.z();

  } else if (triger_mode == 1 || triger_mode == 2) {
    // Mode 1/2 (精确跟踪模式): 使用实时狗位姿
    target_vx = hc14_dog_vel.x();
    target_vy = hc14_dog_vel.y();

    // ----- 角速度前馈补偿 -----
    // 原理: 狗在转弯时,无人机需要预测狗下一个位置,提前移动
    // 侧面方向: 狗的角速度 × 前馈增益 = 转弯方向的提前量
    // 前方方向: 狗的角速度 × 前馈增益 = 线速度方向的提前量
    double dog_vel_norm = std::sqrt(hc14_dog_vel.x() * hc14_dog_vel.x() + hc14_dog_vel.y() * hc14_dog_vel.y());

    // 侧面分量补偿: 狗转弯时,无人机在侧面方向提前偏移
    double compensation_pos_lateral = hc14_dog_yaw_rate * yaw_rate_pos_gain;
    double compensation_vel_lateral = hc14_dog_yaw_rate * yaw_rate_vel_gain;

    // 前方分量补偿: 狗转弯时,前方线速度变化的前馈
    double compensation_pos_forward = std::fabs(hc14_dog_yaw_rate) * yaw_rate_pos_gain_forward;
    double compensation_vel_forward = std::fabs(hc14_dog_yaw_rate) * yaw_rate_vel_gain_forward;

    // 计算狗速度的法向(侧面)和切向(前方)单位向量
    double normal_x = 0.0, normal_y = 0.0;
    double forward_x = 0.0, forward_y = 0.0;

    if (dog_vel_norm > 0.001) {
      // 法向: 速度方向逆时针 90° (← 侧面)
      normal_x = -hc14_dog_vel.y() / dog_vel_norm;
      normal_y = hc14_dog_vel.x() / dog_vel_norm;

      // 切向: 速度方向(↑ 前方)
      forward_x = hc14_dog_vel.x() / dog_vel_norm;
      forward_y = hc14_dog_vel.y() / dog_vel_norm;
    }

    // 计算侧面/前方的位置和速度补偿
    double pos_compensation_lateral_x = normal_x * compensation_pos_lateral;
    double pos_compensation_lateral_y = normal_y * compensation_pos_lateral;
    double pos_compensation_forward_x = forward_x * compensation_pos_forward;
    double pos_compensation_forward_y = forward_y * compensation_pos_forward;
    double vel_compensation_lateral_x = normal_x * compensation_vel_lateral;
    double vel_compensation_lateral_y = normal_y * compensation_vel_lateral;
    double vel_compensation_forward_x = forward_x * compensation_vel_forward;
    double vel_compensation_forward_y = forward_y * compensation_vel_forward;

    targetx = hc14_dog_pos.x();
    targety = hc14_dog_pos.y();

    // 加上侧面和前方的位置/速度补偿
    targetx += pos_compensation_lateral_x + pos_compensation_forward_x;
    targety += pos_compensation_lateral_y + pos_compensation_forward_y;
    target_vx += vel_compensation_lateral_x + vel_compensation_forward_x;
    target_vy += vel_compensation_lateral_y + vel_compensation_forward_y;

    // ----- 高度控制 (Mode 1 vs Mode 2) -----
    if (triger_mode == 1)
    {
      if (reflight_complete){
        // 已在目标高度带内: 正常跟踪
        targetz = std::min(hc14_dog_pos.z() + land_height_limit[1],
                          std::max(hc14_dog_pos.z() + land_height_limit[0], vins_p.z()));
        target_vz = hc14_dog_vel.z();
      } else {
        // 尚未进入目标高度带: 缓慢爬升/下降
        if (mode_vins_z < hc14_dog_pos.z() + land_height_limit[0]){
          mode_vins_vel_z = 0.05;   // 低于下界: 爬升
          if (vins_p.z() > hc14_dog_pos.z() + land_height_limit[0]){
            reflight_complete = true;  // 进入高度带
          }
        } else if (mode_vins_z > hc14_dog_pos.z() + land_height_limit[1]){
          mode_vins_vel_z = -0.05;  // 高于上界: 下降
          if (vins_p.z() < hc14_dog_pos.z() + land_height_limit[1]){
            reflight_complete = true;
          }
        }
        else {
          reflight_complete = true;  // 已在高度带内
        }

        mode_vins_z += mode_vins_vel_z * accel_dt;
        target_vz = hc14_dog_vel.z() + mode_vins_vel_z;
        targetz = mode_vins_z;
      }
    }
    else if (triger_mode == 2)
    {
      // 降落模式: 叠加下降速度
      mode_vins_z += land_vel * accel_dt;   // land_vel = -0.6 m/s
      target_vz = hc14_dog_vel.z() + land_vel;
      targetz = mode_vins_z;
    }
  }

  // ===== 计算世界系跟踪误差 =====
  error_targetx = targetx - vins_p.x();
  error_targety = targety - vins_p.y();
  error_targetz = targetz - vins_p.z();

  if(last_error_targetx == 0){
    last_error_targetx = error_targetx;
    last_error_targety = error_targety;
    last_error_targetz = error_targetz;
  }

  // ===== 坐标系变换: 世界系 → 狗体坐标系 =====
  // 用目标狗航向 (target_yaw) 做旋转变换
  double cos_yaw, sin_yaw;
  if ((triger_mode == 0 || triger_mode == -2)) {
    cos_yaw = cos(0.0);
    sin_yaw = sin(0.0);
  } else {
    cos_yaw = cos(target_yaw);
    sin_yaw = sin(target_yaw);
  }

  // 将世界系的位置误差旋转到狗体坐标系:
  //   x_body = 狗头方向(前方)
  //   y_body = 狗侧面(左侧,右手定则)
  //
  //   [error_x_body]   [ cos_yaw  sin_yaw] [error_x_world]
  //   [error_y_body] = [-sin_yaw  cos_yaw] [error_y_world]
  double error_x_body =  error_targetx * cos_yaw + error_targety * sin_yaw;
  double error_y_body =  - error_targetx * sin_yaw + error_targety * cos_yaw;
  double last_error_x_body = last_error_targetx * cos_yaw + last_error_targety * sin_yaw;
  double last_error_y_body = - last_error_targetx * sin_yaw + last_error_targety * cos_yaw;
  double target_vx_body = target_vx * cos_yaw + target_vy * sin_yaw;
  double target_vy_body = - target_vx * sin_yaw + target_vy * cos_yaw;
  double target_x_body = targetx * cos_yaw + targety * sin_yaw;
  double target_y_body = - targetx * sin_yaw + targety * cos_yaw;

  // ===== 根据模式选择 PID 参数 =====
  double current_x_p, current_y_p, current_z_p, current_pos_x_p, current_pos_y_p;
  double current_x_i, current_y_i, current_z_i, current_pos_x_i, current_pos_y_i;
  double current_x_d, current_y_d, current_z_d, current_pos_x_d, current_pos_y_d;
  double current_x_d_max, current_y_d_max, current_z_d_max, current_pos_x_d_max, current_pos_y_d_max;
  
  if ((triger_mode == 0 || triger_mode == -2)) {
    current_x_p = mpc_x_p;
    current_y_p = mpc_y_p;
    current_z_p = mpc_z_p;
    current_pos_x_p = mpc_pos_x_p;
    current_pos_y_p = mpc_pos_y_p;
    current_x_i = mpc_x_i;
    current_y_i = mpc_y_i;
    current_z_i = mpc_z_i;
    current_pos_x_i = mpc_pos_x_i;
    current_pos_y_i = mpc_pos_y_i;
    current_x_d = mpc_x_d;
    current_y_d = mpc_y_d;
    current_z_d = mpc_z_d;
    current_pos_x_d = mpc_pos_x_d;
    current_pos_y_d = mpc_pos_y_d;
    current_x_d_max = mpc_x_d_max;
    current_y_d_max = mpc_y_d_max;
    current_z_d_max = mpc_z_d_max;
    current_pos_x_d_max = mpc_pos_x_d_max;
    current_pos_y_d_max = mpc_pos_y_d_max;
  } else if (triger_mode == 1 || triger_mode == 2) {  
    current_x_p = x_p;
    current_y_p = y_p;
    current_z_p = z_p;
    current_pos_x_p = pos_x_p;
    current_pos_y_p = pos_y_p;
    current_x_i = x_i;
    current_y_i = y_i;
    current_z_i = z_i;
    current_pos_x_i = pos_x_i;
    current_pos_y_i = pos_y_i;
    current_x_d = x_d;
    current_y_d = y_d;
    current_z_d = z_d;
    current_pos_x_d = pos_x_d;
    current_pos_y_d = pos_y_d;
    current_x_d_max = x_d_max;
    current_y_d_max = y_d_max;
    current_z_d_max = z_d_max;
    current_pos_x_d_max = pos_x_d_max;
    current_pos_y_d_max = pos_y_d_max;
  }

  // ===== 积分项选择 =====
  // Mode 0: 在世界系积分(因不需要坐标变换)
  // Mode 1/2: 在狗体坐标系积分
  double integral_x_used, integral_y_used;
  if ((triger_mode == 0 || triger_mode == -2)) {
    // trigger_mode=0/-2时使用世界系积分项，需要转换到body系
    integral_x_used = intergral_targetx * cos_yaw + intergral_targety * sin_yaw;
    integral_y_used = - intergral_targetx * sin_yaw + intergral_targety * cos_yaw;
  } else {
    integral_x_used = intergral_targetx_body;
    integral_y_used = intergral_targety_body;
  }

  // ===== 狗体坐标系 PID 计算 =====
  // PID 公式: u = target_v + Kp*e + Ki*∫e + Kd*Δe
  // 其中:
  //   - target_v 是前馈项(狗本身的速度)
  //   - Kp*e 是比例项(根据位置误差立即反应)
  //   - Ki*∫e 是积分项(消除稳态误差)
  //   - Kd*Δe 是微分项(阻尼,抑制超调,带限幅)

  // 狗头方向 (x_body,前进方向) PID:
  double vx_body = target_vx_body
                   + current_x_p * error_x_body
                   + current_x_i * integral_x_used
                   + std::max(std::min(current_x_d * (error_x_body - last_error_x_body), current_x_d_max), -current_x_d_max);

  // 狗侧方向 (y_body,侧面) PID:
  double vy_body = target_vy_body
                   + current_y_p * error_y_body
                   + current_y_i * integral_y_used
                   + std::max(std::min(current_y_d * (error_y_body - last_error_y_body), current_y_d_max), -current_y_d_max);

  // 高度 (z,世界系) PID:
  double vz = target_vz
              + current_z_p * error_targetz
              + current_z_i * intergral_targetz
              + std::max(std::min(current_z_d * (error_targetz - last_error_targetz), current_z_d_max), -current_z_d_max);

  // 狗体系位置计算(PID 作用于位置,生成位置参考)
  double x_body = target_x_body
                  + current_pos_x_p * error_x_body
                  + current_pos_x_i * integral_x_used
                  + std::max(std::min(current_pos_x_d * (error_x_body - last_error_x_body), current_pos_x_d_max), -current_pos_x_d_max);
  double y_body = target_y_body
                  + current_pos_y_p * error_y_body
                  + current_pos_y_i * integral_y_used
                  + std::max(std::min(current_pos_y_d * (error_y_body - last_error_y_body), current_pos_y_d_max), -current_pos_y_d_max);

  // ===== 狗体坐标系 → 世界坐标系 (逆变换) =====
  // R(-yaw) 逆变换: Rᵀ(yaw)
  cmd.velocity.x = vx_body * cos_yaw - vy_body * sin_yaw;
  cmd.velocity.y = vx_body * sin_yaw + vy_body * cos_yaw;
  cmd.velocity.z = vz;

  cmd.position.x = x_body * cos_yaw - y_body * sin_yaw;
  cmd.position.y = x_body * sin_yaw + y_body * cos_yaw;
  cmd.position.z = targetz;

  // ===== 加速度前馈 (Mode 1/2 精确跟踪模式) =====
  // 使用狗加速度做前馈: 预判狗的运动趋势,让无人机提前响应
  if ((triger_mode == 1 || triger_mode == 2)) {
    // 加速度前馈: 来自 dog_pos_processor 估计的狗加速度
    double acc_x = hc14_dog_acc.x() * acc_gain_x;
    double acc_y = hc14_dog_acc.y() * acc_gain_y;
    // Jerk 前馈: 狗加速度的导数(来自滤波后的 jerk)
    double jerk_x = hc14_dog_jerk_filtered.x() * jerk_gain_x;
    double jerk_y = hc14_dog_jerk_filtered.y() * jerk_gain_y;
    // 饱和映射: 输入较小时线性,较大时饱和(防止噪声放大)
    double sat_acc_x = std::min(std::fabs(acc_x) / acc_feedforward_sat_threshold_x, 1.0) * acc_feedforward_sat_value_x;
    double sat_acc_y = std::min(std::fabs(acc_y) / acc_feedforward_sat_threshold_y, 1.0) * acc_feedforward_sat_value_y;
    double sat_jerk_x = std::min(std::fabs(jerk_x) / jerk_feedforward_sat_threshold_x, 1.0) * jerk_feedforward_sat_value_x;
    double sat_jerk_y = std::min(std::fabs(jerk_y) / jerk_feedforward_sat_threshold_y, 1.0) * jerk_feedforward_sat_value_y;
    cmd.acceleration.x = sat_acc_x * acc_x + sat_jerk_x * jerk_x;
    cmd.acceleration.y = sat_acc_y * acc_y + sat_jerk_y * jerk_y;
  }

  // ===== Yaw 命令 =====
  // 限制单帧转角不超过 0.5 rad (≈28.6°),防止急剧旋转
  cmd.yaw = std::max(std::min(angle_diff, 0.5), -0.5) + vins_yaw;
  cmd.yaw_dot = 0.0;  // 不使用角速度控制(依赖飞控内部的 yaw 控制)

  // ===== 降落逻辑 (Mode 2) =====
  if(triger_mode == 2){
    // 降落锁定: 当光流高度 < 0.1m 且无人机与狗高度差 < 0.5m,
    //            持续 8 帧(约 0.12s)后触发降落
    if ((flow_z < 0.1) && (flow_z > 0.0) && (std::fabs(vins_p.z() - hc14_dog_pos.z()) < 0.5))
        land_lock_timer += 1;
    else
        land_lock_timer = std::max(land_lock_timer - 0.5, 0.0);

    // 降落触发: land_lock_timer > 8 (约 0.12s 持续满足条件)
    if (land_lock_timer > 8)
    {
      // 发布降落命令给 px4ctrl
      quadrotor_msgs::TakeoffLand land;
      land.takeoff_land_cmd = 2;
      land_pub_.publish(land);

      // 保存 yaw offset 差值(供下次起飞时初始化)
      std_msgs::Float64 yaw_diff_msg;
      yaw_diff_msg.data = angle_diff;
      yaw_offset_pub.publish(yaw_diff_msg);
      std::cout << "Published yaw diff: " << angle_diff * 180.0 / M_PI << " degrees" << std::endl;

      // 模式退回到 -1 (所有节点停止)
      geometry_msgs::PoseStamped mode_msg;
      mode_msg.header.stamp = ros::Time::now();
      mode_msg.pose.orientation.w = -1;
      mode_pub_.publish(mode_msg);
    }
  }

  // ===== 速度偏移(手动微调,仅仿真环境启用) =====
  cmd.velocity.x += v_offset_x * cos(vins_yaw) - v_offset_y * sin(vins_yaw);
  cmd.velocity.y += v_offset_x * sin(vins_yaw) + v_offset_y * cos(vins_yaw);

  // ===== 速度上下限限制 =====
  cmd.velocity.x = std::max(-1.5, std::min(1.5, cmd.velocity.x));
  cmd.velocity.y = std::max(-1.5, std::min(1.5, cmd.velocity.y));
  cmd.velocity.z = std::max(-0.8, std::min(0.8, cmd.velocity.z));

  // ===== 加速度和 Jerk 限制链 =====
  // 三级限制(依次执行):
  //   1. Jerk(加加速度)限制: |accel_change| ≤ max_jerk * dt
  //   2. 加速度限制:         |accel| ≤ max_accel
  //   3. 位置变化率限制:     配套速度限制
  if (last_cmd_initialized) {
    // 计算期望速度变化 = 当前指令速度 - 上一帧指令速度
    Eigen::Vector3d vel_change(
        cmd.velocity.x - last_cmd_velocity.x(),
        cmd.velocity.y - last_cmd_velocity.y(),
        cmd.velocity.z - last_cmd_velocity.z());

    // 期望加速度 = 速度变化 / dt
    Eigen::Vector3d desired_accel = vel_change / accel_dt;

    // ---- 第三级: Jerk 限制 ----
    // 加速度变化量
    Eigen::Vector3d accel_change = desired_accel - last_cmd_acceleration;
    double max_accel_change = max_jerk * accel_dt;

    // 限制加速度变化率: |Δa| ≤ max_jerk * dt
    if (accel_change.norm() > max_accel_change) {
      accel_change = accel_change.normalized() * max_accel_change;
    }

    // ---- 第二级: 加速度限制 ----
    Eigen::Vector3d limited_accel = last_cmd_acceleration + accel_change;
    if (limited_accel.norm() > max_accel) {
      limited_accel = limited_accel.normalized() * max_accel;
    }

    // ---- 反算速度变化 ----
    Eigen::Vector3d limited_vel_change = limited_accel * accel_dt;

    // 更新指令速度
    cmd.velocity.x = last_cmd_velocity.x() + limited_vel_change.x();
    cmd.velocity.y = last_cmd_velocity.y() + limited_vel_change.y();
    cmd.velocity.z = last_cmd_velocity.z() + limited_vel_change.z();

    // ---- 第一级: 位置变化率限制 ----
    Eigen::Vector3d pos_change(
        cmd.position.x - last_cmd_position.x(),
        cmd.position.y - last_cmd_position.y(),
        cmd.position.z - last_cmd_position.z());
    Eigen::Vector3d limited_vel_vec(cmd.velocity.x, cmd.velocity.y, cmd.velocity.z);
    double max_pos_change = limited_vel_vec.norm() * accel_dt * 3.5;  // 3.5× 速度作为位置变化上限

    if (max_pos_change > 0.001 && pos_change.norm() > max_pos_change) {
      pos_change = pos_change.normalized() * max_pos_change;
      cmd.position.x = last_cmd_position.x() + pos_change.x();
      cmd.position.y = last_cmd_position.y() + pos_change.y();
      cmd.position.z = last_cmd_position.z() + pos_change.z();
    }

    last_cmd_acceleration = limited_accel;
  } else {
    // 首次初始化: 用当前 VINS 状态初始化命令
    last_cmd_initialized = true;
    cmd.velocity.x = vins_v.x();
    cmd.velocity.y = vins_v.y();
    cmd.velocity.z = vins_v.z();
    cmd.position.x = vins_p.x();
    cmd.position.y = vins_p.y();
    cmd.position.z = vins_p.z();
    last_cmd_acceleration = Eigen::Vector3d(0, 0, 0);
  }

  // 保存本轮命令(供下一帧的限幅计算使用)
  last_cmd_velocity = Eigen::Vector3d(cmd.velocity.x, cmd.velocity.y, cmd.velocity.z);
  last_cmd_position = Eigen::Vector3d(cmd.position.x, cmd.position.y, cmd.position.z);

  // 保存上一帧值
  last_target_v.x() = target_vx;
  last_target_v.y() = target_vy;
  last_target_v.z() = target_vz;
  last_error_targetx = error_targetx;
  last_error_targety = error_targety;
  last_error_targetz = error_targetz;
  last_precise_target_count = target_count;

  // ===== 积分项更新(带衰减防止 Windup) =====
  // 积分衰减原理: I_new = decay * I_old + dt * error
  //   当 decay=1 时(integral_decay=0): 不衰减,无限累积(易 windup)
  //   当 decay<1 时: 旧积分逐渐衰减,防止 windup(但会丢失稳态校正能力)
  double decay_factor = 1.0 - integral_decay * accel_dt;
  decay_factor = std::max(0.0, std::min(1.0, decay_factor));

  if ((triger_mode == 0 || triger_mode == -2)) {
    // trigger_mode=0/-2时在世界系累积x和y的积分项
    intergral_targetx = intergral_targetx * decay_factor + accel_dt * (cmd.position.x - vins_p.x());
    intergral_targety = intergral_targety * decay_factor + accel_dt * (cmd.position.y - vins_p.y());
  } else {
    // Mode 1/2: 在狗体系累积积分
    intergral_targetx_body = intergral_targetx_body * decay_factor + accel_dt * ((cmd.position.x - vins_p.x()) * cos_yaw + (cmd.position.y - vins_p.y()) * sin_yaw);
    intergral_targety_body = intergral_targety_body * decay_factor + accel_dt * (- (cmd.position.x - vins_p.x()) * sin_yaw + (cmd.position.y - vins_p.y()) * cos_yaw);
  }

  // Z 轴积分(始终在世界系,与航向无关)
  intergral_targetz = intergral_targetz * decay_factor + accel_dt * (cmd.position.z - vins_p.z());

  // 限制积分项
  if ((triger_mode == 0 || triger_mode == -2)) {
    intergral_targetx = std::max(std::min(intergral_targetx, integral_limit_x), -integral_limit_x);
    intergral_targety = std::max(std::min(intergral_targety, integral_limit_y), -integral_limit_y);
  } else {
    intergral_targetx_body = std::max(std::min(intergral_targetx_body, integral_limit_x), -integral_limit_x);
    intergral_targety_body = std::max(std::min(intergral_targety_body, integral_limit_y), -integral_limit_y);
  }
  intergral_targetz = std::max(std::min(intergral_targetz, integral_limit_z), -integral_limit_z);

  // ===== 发布调试信息 =====
  quadrotor_msgs::PositionCommand debug_msg;
  debug_msg.header.stamp = ros::Time::now();
  debug_msg.header.frame_id = "world";
  debug_msg.position.x = error_targetx;                         // 世界系 x 误差
  debug_msg.position.y = error_targety;                         // 世界系 y 误差
  debug_msg.position.z = std::sqrt(error_targetx * error_targetx + error_targety * error_targety); // xy 误差模长
  // 狗体系下向量角度(用于分析跟踪方向偏差)
  debug_msg.velocity.x = std::atan2(
      (cmd.position.x - vins_p.x()) * cos_yaw + (cmd.position.y - vins_p.y()) * sin_yaw,
      - (cmd.position.x - vins_p.x()) * sin_yaw + (cmd.position.y - vins_p.y()) * cos_yaw
  );
  debug_msg.velocity.y = hc14_dog_jerk_filtered.x();            // 滤波后狗 jerk x
  debug_msg.velocity.z = hc14_dog_jerk_filtered.y();            // 滤波后狗 jerk y
  debug_msg.yaw = last_error_targetx;                           // 上帧误差(对比)
  debug_pub_.publish(debug_msg);

  // ===== 发布位置命令 =====
  pos_cmd_pub_.publish(cmd);
  return;
}


// ============================================================================
// 十一、MPC 轨迹回调 + 模式切换回调
// ============================================================================

/**
 * traj_callback — 接收 MPC 位置轨迹 (/drone2/planning/traj)
 * 来自 MPC.py, 格式为 Path 消息, 每个 pose 代表一个轨迹点的位置
 */
void traj_callback(const nav_msgs::Path::ConstPtr& msg) {
    traj_sub_time = ros::Time::now();
    trajectory_points.clear();
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d pos;
        pos << pose.pose.position.x,
               pose.pose.position.y,
               pose.pose.position.z;
        trajectory_points.push_back(pos);
    }
}

/**
 * traj_v_callback — 接收 MPC 速度轨迹 (/traj_v)
 */
void traj_v_callback(const nav_msgs::Path::ConstPtr& msg) {
    if (!traj_initialized && !msg->poses.empty()) {
        traj_initialized = true;
    }
    trajectory_v_points.clear();
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d pos;
        pos << pose.pose.position.x,
               pose.pose.position.y,
               pose.pose.position.z;
        trajectory_v_points.push_back(pos);
    }
}

/**
 * traj_a_callback — 接收 MPC 加速度轨迹 (/traj_a)
 */
void traj_a_callback(const nav_msgs::Path::ConstPtr& msg) {
    trajectory_a_points.clear();
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d acc;
        acc << pose.pose.position.x,
               pose.pose.position.y,
               pose.pose.position.z;
        trajectory_a_points.push_back(acc);
    }
}


// ============================================================================
// 十二、模式切换逻辑 (flag_and_hc14_process_callback @ 10Hz)
// ============================================================================
// 负责:
//   1. 检查各数据源健康状态(数据是否在持续更新)
//   2. 根据距离、速度差、角度差等条件自动切换模式

void flag_and_hc14_process_callback(const ros::TimerEvent &event) {
    // ----- 目标数据状态检查 -----
    if (target_count != last_target_count) {
        last_target_timer++;
        if (last_target_timer >= 5) {
            target_receive = true;   // 连续收到 5 帧 → 标记为 active
        }
        last_target_count = target_count;
        last_target_loss_timer = 0;
    } else {
        last_target_loss_timer++;
        if (last_target_loss_timer >= 5) {
            target_receive = false;  // 连续 5 帧无新数据 → 标记为丢失
        }
        last_target_timer -= 0.2;   // 缓慢衰减
    }

    // ----- 狗数据状态检查 -----
    if (hc14_dog_pos_count != last_hc14_dog_pos_count) {
        hc14_dog_pos_received = true;
        last_hc14_dog_pos_count = hc14_dog_pos_count;
        last_hc14_dog_pos_timer = 0;
    } else {
        last_hc14_dog_pos_timer++;
        if (last_hc14_dog_pos_timer >= 5)
        hc14_dog_pos_received = false;
    }

    // ----- 计算无人机和狗的 yaw 差 -----
    double angle_diff = target_dog_yaw - vins_yaw;
    if (angle_diff > M_PI) {
      angle_diff -= 2 * M_PI;
    } else if (angle_diff < -M_PI) {
      angle_diff += 2 * M_PI;
    }

    // 处理triger_mode切换逻辑
    if ((triger_mode == 0 || triger_mode == -2)){
      if (hc14_offset_pos_ready && 
          hc14_offset_yaw_ready && 
          target_receive && 
          vins_p.z() > hc14_dog_pos.z() + land_height_limit[0] &&
          Eigen::Vector2d(vins_p.x() - hc14_dog_pos.x(), vins_p.y() - hc14_dog_pos.y()).norm() < 0.8 &&
          angle_diff < 30.0/180.0 * M_PI)
      {
          geometry_msgs::PoseStamped mode_msg;
          mode_msg.header.stamp = ros::Time::now();
          mode_msg.pose.orientation.w = 1;  // 通知 MPC.py 切换到精确模式
          mode_pub_.publish(mode_msg);
          std::cout << "precise_mode: true" << std::endl;
      }
    }
    // ===== Mode 1 → Mode 2 切换条件 =====
    // 从精确跟踪切换到降落模式的条件(全部满足,持续 10 帧):
    //   1. hc14_offset 就绪
    //   2. 速度差 < 0.3 m/s
    //   3. 位置差 < 0.4 m
    //   4. 角度差 < 30°
    else if (triger_mode == 1){
      if (hc14_offset_yaw_ready &&
          hc14_offset_pos_ready &&
          std::fabs(hc14_dog_vel.x() - vins_v.x()) < 0.3 && 
          std::fabs(hc14_dog_vel.y() - vins_v.y()) < 0.3 && 
          std::fabs(hc14_dog_pos.x() - vins_p.x()) < 0.4 &&
          std::fabs(hc14_dog_pos.y() - vins_p.y()) < 0.4 &&
          angle_diff < 30.0/180.0 * M_PI)
      {
        land_timer ++;
        if (land_timer > 10)  // 持续 10 帧确认
        {
          geometry_msgs::PoseStamped mode_msg;
          mode_msg.header.stamp = ros::Time::now();
          mode_msg.pose.orientation.w = 2;  // 通知 MPC.py 切换到降落模式
          mode_pub_.publish(mode_msg);
          std::cout << "land_mode: true" << std::endl;
          land_timer = 0;
        }
      }else if (Eigen::Vector2d(vins_p.x() - hc14_dog_pos.x(), vins_p.y() - hc14_dog_pos.y()).norm() > 2.0){
        // 可能遇到障碍物 → 切回 MPC 模式避障
        geometry_msgs::PoseStamped mode_msg;
        mode_msg.header.stamp = ros::Time::now();
        mode_msg.pose.orientation.w = 0;
        mode_pub_.publish(mode_msg);
        std::cout << "precise_mode: false" << std::endl;
        land_timer = std::max(0.0, land_timer - 0.5);
      }else{
        land_timer = std::max(0.0, land_timer - 0.5);
      }
    }
    // ===== Mode 2 → Mode 1 切换条件(降落中断) =====
    // 降落过程中如果条件不满足(偏离过大),退回到精确跟踪模式
    else if (triger_mode == 2){
      if (!(
            hc14_offset_pos_ready &&
            hc14_offset_yaw_ready &&
            std::fabs(hc14_dog_vel.x() - vins_v.x()) < 0.65 && 
            std::fabs(hc14_dog_vel.y() - vins_v.y()) < 0.65 && 
            std::fabs(hc14_dog_pos.x() - vins_p.x()) < 0.3 &&
            std::fabs(hc14_dog_pos.y() - vins_p.y()) < 0.3 &&
            vins_p.z() > hc14_dog_pos.z() - 0.2 &&
            angle_diff < 45.0/180.0 * M_PI ))
      {
        geometry_msgs::PoseStamped mode_msg;
        mode_msg.header.stamp = ros::Time::now();
        mode_msg.pose.orientation.w = 1;
        mode_pub_.publish(mode_msg);
        std::cout << "land_mode: false" << std::endl;
      }
    }
}


// ============================================================================
// 十三、MPC 轨迹时间插值回调 (mpc_callback @ 100Hz)
// ============================================================================
// 根据当前时间和轨迹接收时间的差,从 MPC 轨迹序列中插值出当前位置、速度和加速度。

void mpc_callback(const ros::TimerEvent &event){
  if (!traj_initialized)
  {
    return;
  }
  ros::Time now = ros::Time::now();
  double dt = (now - traj_sub_time).toSec();  // 距离轨迹接收的经过时间
  double dt_step = 0.1;                        // 轨迹点时间步长(对应 MPC 的 dt=0.1s)

  dt += 0.0;  // 预留的提前量参数(可调整为正值以取未来点)

  // 计算浮点索引(可能带小数)
  double idx_float = dt / dt_step;

  // 找到相邻的两个整数索引
  int idx_down = static_cast<int>(std::floor(idx_float));
  int idx_up = idx_down + 1;

  // 边界处理
  int max_idx = static_cast<int>(trajectory_points.size() - 1);
  idx_down = std::max(0, std::min(idx_down, max_idx));
  idx_up = std::max(0, std::min(idx_up, max_idx));

  // 超出轨迹范围: 保持在最后一点
  if (idx_down >= max_idx) {
    mpc_p = trajectory_points[max_idx];
    mpc_v = trajectory_v_points[max_idx];
    if (!trajectory_a_points.empty() && max_idx < static_cast<int>(trajectory_a_points.size())) {
      mpc_a = trajectory_a_points[max_idx];
    }
    return;
  }

  // 线性插值: 根据小数权重在两个相邻点之间加权
  double weight_up = idx_float - idx_down;      // 上权重(离上一个点多远)
  double weight_down = 1.0 - weight_up;          // 下权重(离下一个点多远)

  // 位置和速度的加权插值
  mpc_p = weight_down * trajectory_points[idx_down] + weight_up * trajectory_points[idx_up];
  mpc_v = weight_down * trajectory_v_points[idx_down] + weight_up * trajectory_v_points[idx_up];

  // 加速度同理
  if (!trajectory_a_points.empty() &&
      idx_down < static_cast<int>(trajectory_a_points.size()) &&
      idx_up < static_cast<int>(trajectory_a_points.size())) {
    mpc_a = weight_down * trajectory_a_points[idx_down] + weight_up * trajectory_a_points[idx_up];
  }
}


// ============================================================================
// 十四、主函数 — ROS 节点入口
// ============================================================================

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle nh("~");

  // ===== 订阅各种传感器和控制数据 =====
  ros::Subscriber heartbeat_sub = nh.subscribe("heartbeat", 10, heartbeatCallback);
  ros::Subscriber triger_sub_ = nh.subscribe("/mode_manager", 10, mode_callback);
  ros::Subscriber odom_sub_ = nh.subscribe("/vins_fusion/imu_propagate", 10, odom_callback);
  ros::Subscriber target_sub_ = nh.subscribe("/target_ekf_odom", 10, target_callback);
  ros::Subscriber AOA_sub_ = nh.subscribe("/AOA_Tag_data", 10, AOA_callback);
  ros::Subscriber flow_sub_ = nh.subscribe("/flow_data", 10, flow_callback);

  // 订阅 MPC 轨迹(来自 MPC.py):
  //   /drone2/planning/traj → 位置轨迹
  //   /traj_v               → 速度轨迹
  //   /traj_a               → 加速度轨迹
  ros::Subscriber traj_sub_ = nh.subscribe<nav_msgs::Path>("/drone2/planning/traj", 10, traj_callback);
  ros::Subscriber traj_v_sub_ = nh.subscribe<nav_msgs::Path>("/traj_v", 10, traj_v_callback);
  ros::Subscriber traj_a_sub_ = nh.subscribe<nav_msgs::Path>("/traj_a", 10, traj_a_callback);
  
  // 订阅处理后的dog_pos话题
  ros::Subscriber dog_pos_sub_ = nh.subscribe<nav_msgs::Odometry>("/dog_pos_processed", 10, dog_pos_callback);
  ros::Subscriber command_pos_sub_ = nh.subscribe<nav_msgs::Odometry>("/command_pos", 10, command_pos_callback);

  // 订阅坐标系对齐后的狗位姿(来自 dog_pos_processor)
  ros::Subscriber dog_pos_sub_ = nh.subscribe<nav_msgs::Odometry>("/dog_pos_processed", 10, dog_pos_callback);

  // ===== 发布器注册 =====
  pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
  land_pub_ = nh.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land",1);
  mode_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/mode_manager", 10);
  debug_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/debug_info", 50);
  yaw_offset_pub = nh.advertise<std_msgs::Float64>("/yaw_diff_preset", 10);

  // ===== 定时器 =====
  // 初始化定时器: 2s 后执行,检查各节点是否就绪
  ros::Timer init_timer = nh.createTimer(ros::Duration(2.0), initCallback);
  // 主控制定时器: 67Hz (0.015s 周期)
  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.015), cmdCallback);
  // MPC 插值定时器: 100Hz (0.01s 周期)
  ros::Timer mpc_timer = nh.createTimer(ros::Duration(0.01), mpc_callback);
  // 模式切换定时器: 10Hz (0.1s 周期)
  ros::Timer flag_and_hc14_process_timer = nh.createTimer(ros::Duration(0.1), flag_and_hc14_process_callback);

  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}
