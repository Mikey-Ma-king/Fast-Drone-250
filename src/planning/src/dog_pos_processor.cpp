/**
 * ===========================================================================
 * dog_pos_processor.cpp — 狗位姿坐标对齐处理器
 * ===========================================================================
 *
 * 【在整个系统中的角色】
 *   本模块是感知管道的第二环节,位于 read.cpp (视觉检测)和 traj_server.cpp (轨迹跟踪)
 *   之间。它的核心任务是解决两个坐标系的"对齐"问题:
 *
 *   - 狗身上的 UWB/通信模块报告的位姿数据在"狗坐标系"下(以狗自身航向为前方)
 *   - 无人机飞行控制和 MPC 规划需要"世界坐标系"下的目标位姿
 *   - 本模块通过维护 yaw_offset (旋转)和 pos_offset (平移),实时将狗坐标系数据
 *     转换为世界坐标系数据,发布 /dog_pos_processed 供下游消费
 *
 * 【核心管道】
 *   /dog_pos (狗原始数据) ─┐
 *   /target_ekf_odom (视觉)─┤
 *   /vins_fusion (无人机)  ─┤
 *   /AOA_Tag_data (UWB)   ─┤
 *   /flow_data (光流高度)  ─┤
 *                            ├→ [processCallback 50Hz]
 *                            │    1. Yaw offset 迭代 (狗航向 → 世界航向)
 *                            │    2. Pos offset 迭代 (狗位置 → 世界位置)
 *                            │    3. AOA 几何修正 (UWB 单锚点辅助定位)
 *                            │    4. 非线性坐标变换 (狗系→世界系)
 *                            │    5. 卡尔曼滤波 (6 维 CV 模型)
 *                            │
 *                            └→ 发布 /dog_pos_processed
 *
 * 【坐标变换原理】
 *   给定狗系下的 raw_dog_pos 和 raw_dog_yaw,要得到世界系下的位置:
 *
 *   1. 减去位置偏移:  corrected_pos = raw_pos - pos_offset   (狗系下偏移校正)
 *   2. 旋转 yaw 偏移:  rotated = R(-yaw_offset) * corrected_pos
 *
 *   其中:
 *   - yaw_offset = raw_dog_yaw - target_dog_yaw (狗航向和视觉航向的差)
 *   - pos_offset = raw_dog_pos - R(yaw_offset) * vins_pos (平移偏差)
 *
 *   最终发布的世界系位置是:
 *     dog_pos_world = R(-yaw_offset) * (raw_pos - pos_offset)
 *
 * 【AOA 辅助修正原理】(详见 processCallback 中的几何推导)
 *   当 UWB AOA 传感器可用时,利用 "yaw 平面 ∩ 高度平面 = 直线" 的约束,
 *   从单距离+单角度测量解算目标的世界系位置,用于修正 pos_offset。
 *
 * 【依赖项】
 *   - Eigen (线性代数)
 *   - ROS nav_msgs::Odometry (数据通信)
 *   - scipy.spatial.transform (Python 版使用 scipy,C++ 版使用 Eigen 四元数)
 */

#include "dog_pos_processor.h"
#include <ros/ros.h>
#include <Eigen/Dense>
#include <cmath>

// ============================================================================
// 一、卡尔曼滤波器实现 (KalmanFilter)
// ============================================================================
// 恒定速度模型 (CV Model) 的 6 维卡尔曼滤波:
// 状态向量: [x, y, z, vx, vy, vz]
// 状态转移: x_{k+1} = x_k + vx_k * dt (位置 = 位置 + 速度*时间)
// 观测模型: 直接观测位置和速度 (H = I₆)

/**
 * 构造函数 — 初始化卡尔曼滤波器参数
 * @param process_noise   过程噪声协方差 (Q 矩阵的对角值)
 *                        越大 → 滤波器越信任观测,响应越快但噪声敏感
 * @param measurement_noise 测量噪声协方差 (R 矩阵的对角值)
 *                          越小 → 滤波器越信任观测,响应越快但噪声敏感
 *
 * 默认值权衡: process_noise=0.1 (适中的过程噪声,对恒定速度模型有一定容错)
 *            measurement_noise=0.02 (较小的测量噪声,信任 raw dog pos 质量)
 */
KalmanFilter::KalmanFilter(double process_noise, double measurement_noise)
    : initialized_(false) {
    // 状态向量: [x, y, z, vx, vy, vz]
    state_ = Eigen::VectorXd::Zero(6);
    // 初始协方差: 对角 100 (高不确定性,初始几帧快速收敛)
    covariance_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;

    Q_ = Eigen::MatrixXd::Identity(6, 6) * process_noise;
    R_ = Eigen::MatrixXd::Identity(6, 6) * measurement_noise;

    // F_base_ 是基础状态转移矩阵,实际使用时会填充 dt:
    //   F = [ I₃ₓ₃   dt*I₃ₓ₃ ]
    //       [ 0₃ₓ₃   I₃ₓ₃    ]
    F_base_ = Eigen::MatrixXd::Identity(6, 6);

    // 观测矩阵: 直接观测全部 6 维 (可以直接测量位置和速度)
    H_ = Eigen::MatrixXd::Identity(6, 6);
}

/**
 * 预测步骤 — 根据时间间隔 dt 做状态预测
 *
 * 恒定速度模型假设:
 *   - 位置变化由速度驱动: p_new = p_old + v_old * dt
 *   - 速度不变:          v_new = v_old  (过程噪声会逐渐增加速度的不确定性)
 *
 * 协方差传播: P_new = F * P * Fᵀ + Q
 *   第一项: 状态不确定性通过运动模型传播
 *   第二项: dt 期间的过程噪声
 */
void KalmanFilter::predict(double dt) {
    if (!initialized_) return;

    // 构建状态转移矩阵: 只有 [0,3][1,4][2,5] 三个位置是 dt (p = p + v*dt)
    Eigen::MatrixXd F = F_base_;
    F(0, 3) = dt;  // x = x + vx * dt
    F(1, 4) = dt;  // y = y + vy * dt
    F(2, 5) = dt;  // z = z + vz * dt

    // 状态预测
    state_ = F * state_;

    // 协方差预测
    covariance_ = F * covariance_ * F.transpose() + Q_;
}

/**
 * 更新步骤 — 用测量值修正预测结果
 *
 * 标准 Kalman 更新公式:
 *   1. 残差:           y = z - H·x     (测量值 - 预测测量值)
 *   2. 残差协方差:      S = H·P·Hᵀ + R (测量不确定性 + 预测不确定性)
 *   3. 卡尔曼增益:      K = P·Hᵀ·S⁻¹   (最优加权系数)
 *   4. 状态更新:        x = x + K·y     (用测量修正预测)
 *   5. 协方差更新:      P = (I-K·H)·P   (不确定性降低)
 *
 * 卡尔曼增益 K 的本质:
 *   - 测量噪声大(R 大) → K 小 → 更信任预测 → 平滑但延迟
 *   - 预测不确定性大(P 大) → K 大 → 更信任测量 → 响应快但噪声敏感
 */
void KalmanFilter::update(const Eigen::VectorXd& measurement) {
    if (!initialized_) {
        state_ = measurement;
        initialized_ = true;
        return;
    }

    // 残差
    Eigen::VectorXd residual = measurement - H_ * state_;

    // 残差协方差
    Eigen::MatrixXd S = H_ * covariance_ * H_.transpose() + R_;

    // 卡尔曼增益
    Eigen::MatrixXd K = covariance_ * H_.transpose() * S.inverse();

    // 状态更新
    state_ = state_ + K * residual;

    // 协方差更新 (Joseph 形式)
    covariance_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * covariance_;
}

/**
 * 滤波主函数 — Predict + Update 的封装
 * @param position 测量位置 [x, y, z]
 * @param velocity 测量速度 [vx, vy, vz]
 * @param dt       距上次滤波的时间间隔
 * @return         {滤波后位置, 滤波后速度}
 */
std::pair<Eigen::Vector3d, Eigen::Vector3d> KalmanFilter::filter(
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity,
    double dt) {
    // 构建 6 维测量向量
    Eigen::VectorXd measurement(6);
    measurement << position, velocity;

    predict(dt);
    update(measurement);

    // 从状态向量中提取位置(前3维)和速度(后3维)
    return std::make_pair(state_.head<3>(), state_.tail<3>());
}

/**
 * 重置滤波器 — 清空初始化标志,下次调用 filter 时重新初始化
 */
void KalmanFilter::reset() {
    initialized_ = false;
}

// ============================================================================
// 二、狗位姿处理器实现 (DogPosProcessor)
// ============================================================================

static const double default_dt = -1.0;  // 默认时间间隔标记(-1 表示自动计算)

/**
 * 构造函数 — 初始化所有订阅者、发布者、定时器和状态变量
 *
 * 关键参数说明 (与 traj_server.cpp 一致):
 *   yaw_filter_gain_ = 0.1        → yaw offset 迭代步长(越小越保守)
 *   yaw_stable_threshold_ = 5°    → yaw offset 收敛判据
 *   yaw_exceed_threshold_ = 45°   → yaw offset 发散判据
 *   pos_stable_threshold_ = 5cm   → pos offset 收敛判据
 *   pos_exceed_threshold_ = 30cm  → pos offset 发散判据
 *   camera_offset_ = 0.36m        → 相机安装位置到目标几何中心的前馈补偿距离
 *   aoa_min_distance_ = 3.0m      → AOA 最小有效距离(太近时角度噪声大)
 */
DogPosProcessor::DogPosProcessor()
    : nh_("~"),
      saved_yaw_diff_(nullptr),
      yaw_offset_(0.0),
      pos_offset_(Eigen::Vector3d::Zero()),
      precise_yaw_offset_ready_(false),
      precise_pos_offset_ready_(false),
      yaw_exceed_timer_(0),
      pos_exceed_timer_(0),
      initialized_(false),
      simulate_mode_(false),
      raw_dog_vel_(Eigen::Vector3d::Zero()),
      raw_dog_yaw_(0.0),
      raw_dog_pitch_(0.0),
      raw_dog_roll_(0.0),
      raw_dog_pos_received_(false),
      raw_dog_pos_count_(0),
      last_raw_dog_pos_count_(0),
      last_dog_pos_timer_(0),
      vins_yaw_(0.0),
      R_wb_(Eigen::Matrix3d::Identity()),
      vins_pos_(Eigen::Vector3d::Zero()),
      vins_received_(false),
      vins_count_(0),
      last_vins_count_(0),
      last_vins_timer_(0),
      target_dog_yaw_(0.0),
      target_dog_pitch_(0.0),
      target_dog_roll_(0.0),
      target_dog_pos_(Eigen::Vector3d::Zero()),
      target_receive_(false),
      target_count_(0),
      last_target_count_(0),
      last_target_timer_(0),
      last_target_loss_timer_(0),
      last_target_loss_count_(0),
      aoa_received_(false),
      aoa_count_(0),
      last_aoa_count_(0),
      last_aoa_timer_(0),
      aoa_distance_(0.0),
      aoa_angle_(0.0),
      flow_z_(0.0),
      final_dog_pos_(Eigen::Vector3d::Zero()),
      final_dog_vel_(Eigen::Vector3d::Zero()),
      final_dog_yaw_(0.0),
      dog_vel_initialized_(false),
      final_dog_yaw_rate_(0.0),
      last_dog_yaw_time_(0.0),
      last_dog_pitch_time_(0.0),
      last_dog_roll_time_(0.0),
      yaw_rate_filter_gain_(0.3),
      dog_yaw_rate_initialized_(false),
      last_dog_vel_(Eigen::Vector3d::Zero()),
      last_final_dog_yaw_(0.0),
      last_final_dog_pitch_(0.0),
      last_final_dog_roll_(0.0),
      last_final_dog_vel_(Eigen::Vector3d::Zero()),
      last_publish_time_(ros::Time::now()),
      final_dog_acc_(Eigen::Vector3d::Zero()),
      last_dog_vel_time_(0.0),
      acc_filter_gain_(0.3),
      dog_acc_initialized_(false),
      kf_(0.1, 0.01),        // 卡尔曼滤波器(默认关闭)
      kf_enabled_(false),    // 关闭 KF(实验中发现过度平滑导致跟踪滞后)
      yaw_filter_gain_kf_(0.3),
      filtered_yaw_(0.0),
      yaw_filter_initialized_(false),
      kf_timeout_(1.0),      // KF 超时 1s
      trigger_received_(false),
      offset_history_max_size_(10) {  // 历史记录最大长度: 10 个样本

    // ===== 参数设置 (与 traj_server.cpp 一致) =====
    yaw_filter_gain_ = 0.1;                            // yaw offset 一阶滤波增益(≈低通 α)
    yaw_stable_threshold_ = M_PI * 5.0 / 180.0;        // 5 度收敛阈值
    yaw_exceed_threshold_ = M_PI * 45.0 / 180.0;       // 45 度发散阈值
    yaw_exceed_max_count_ = 5;                          // 连续发散 5 次 → 标记 not ready
    pos_stable_threshold_ = 0.05;                       // 5cm 收敛阈值
    pos_exceed_threshold_ = 0.3;                        // 30cm 发散阈值
    pos_exceed_max_count_ = 5;                          // 连续发散 5 次 → 标记 not ready
    pos_filter_gain_ = 0.2;                             // pos offset 一阶滤波增益
    aoa_pos_filter_gain_ = 0.05;                        // AOA 辅助修正增益(比视觉小,更保守)
    aoa_pos_step_limit_ = 0.02;                         // AOA 单次迭代步长上限(防止突变)

    camera_offset_ = 0.36;  // 相机安装偏移: 从相机光心到目标几何中心的水平距离(m)

    aoa_min_distance_ = 3.0;  // AOA 最小有效距离(太近时角度噪声大)

    simulate_mode_ = false;

    // ===== ROS 发布者/订阅者注册 =====
    dog_pos_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_processed", 10);
    aoa_dog_pos_debug_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_aoa_debug", 10);

    raw_dog_pos_sub_ = nh_.subscribe("/dog_pos", 10, &DogPosProcessor::rawDogPosCallback, this);
    target_sub_ = nh_.subscribe("/target_ekf_odom", 10, &DogPosProcessor::targetCallback, this);
    vins_sub_ = nh_.subscribe("/vins_fusion/imu_propagate", 10, &DogPosProcessor::vinsCallback, this);
    aoa_sub_ = nh_.subscribe("/AOA_Tag_data", 10, &DogPosProcessor::aoaCallback, this);
    flow_sub_ = nh_.subscribe("/flow_data", 10, &DogPosProcessor::flowCallback, this);
    takeoff_sub_ = nh_.subscribe("/px4ctrl/takeoff_land", 10, &DogPosProcessor::takeoffCallback, this);
    yaw_diff_preset_sub_ = nh_.subscribe("/yaw_diff_preset", 10, &DogPosProcessor::yawDiffCallback, this);
    test_reset_sub_ = nh_.subscribe("/test_reset_initialized", 10, &DogPosProcessor::testResetCallback, this);
    trigger_sub_ = nh_.subscribe("/triger", 10, &DogPosProcessor::triggerCallback, this);

    // ===== 定时器: 50Hz 主处理 + 10Hz 状态检查 =====
    timer_ = nh_.createTimer(ros::Duration(0.05), &DogPosProcessor::processCallback, this);
    status_timer_ = nh_.createTimer(ros::Duration(0.1), &DogPosProcessor::statusCheckCallback, this);

    if (simulate_mode_) {
        camera_offset_ = 0.0;  // 仿真模式下不使用前馈补偿
    }

    ROS_INFO("Dog Position Processor initialized");
    ROS_INFO("Waiting for takeoff signal and yaw diff from traj_server...");
    ROS_INFO("Initial yaw offset: 0.0 rad (0.0 deg)");
}


// ============================================================================
// 三、角度工具函数
// ============================================================================

/**
 * normalizeAngle — 角度归一化到 [-π, π]
 *
 * 使用 while 循环而非 fmod 的原因:
 *   对于大角度(如 ±10π),while 循环保证一次归一化到位,不会出现过冲
 */
double DogPosProcessor::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}


// ============================================================================
// 四、ROS 回调 — 原始狗数据订阅
// ============================================================================

/**
 * rawDogPosCallback — 原始狗位置/速度/航向数据订阅回调
 *
 * 订阅 /dog_pos 话题,数据格式为 nav_msgs::Odometry:
 *   - position: 狗报告的 3D 位置(在狗坐标系下)
 *   - linear velocity: 狗报告的 3D 速度(在狗坐标系下)
 *   - orientation.w: 狗报告的航向角 yaw(在狗坐标系下,弧度)
 *
 * 处理步骤:
 *   1. 速度限幅 [-2.0, 2.0] m/s (防止异常值)
 *   2. 一阶低通滤波 (0.7*旧 + 0.3*新) → 抑制狗通信噪声
 *   3. Yaw 一阶低通 (α=0.5)
 *   4. 立即调用 publishProcessedDogPos() 发布处理后的结果
 *
 * 滤波权重设计原理:
 *   - 狗头方向(x): 0.7 旧 + 0.3 新 (较保守,因为狗头方向加速度大,容易抖动)
 *   - 狗侧方向(y): 0.7 旧 + 0.3 新 (同上)
 *   - 高度方向(z): 直接使用新值 (狗报告的高度通常较稳定)
 *   - 航向(yaw):  α=0.5 (折中)
 */
void DogPosProcessor::rawDogPosCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    raw_dog_pos_ = msg;
    raw_dog_pos_count_++;

    // 提取原始数据
    Eigen::Vector3d current_raw_dog_vel(
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z
    );
    double current_raw_dog_yaw = msg->pose.pose.orientation.w;

    // 速度限幅: 防止异常值(狗通信模块可能出现错误数据)
    const double min_dog_velocity = -2.0;
    const double max_dog_velocity = 2.0;
    current_raw_dog_vel.x() = std::max(min_dog_velocity, std::min(max_dog_velocity, current_raw_dog_vel.x()));
    current_raw_dog_vel.y() = std::max(min_dog_velocity, std::min(max_dog_velocity, current_raw_dog_vel.y()));
    current_raw_dog_vel.z() = std::max(min_dog_velocity, std::min(max_dog_velocity, current_raw_dog_vel.z()));

    // 速度/航向一阶低通滤波
    if (!dog_vel_initialized_) {
        raw_dog_vel_ = current_raw_dog_vel;
        raw_dog_yaw_ = current_raw_dog_yaw;
        dog_vel_initialized_ = true;

        // 初始化 last_dog_vel_: 将狗系速度转到世界系后保存为上一帧值
        double target_yaw = normalizeAngle(raw_dog_yaw_ - yaw_offset_);
        double cos_yaw = std::cos(target_yaw);
        double sin_yaw = std::sin(target_yaw);
        last_dog_vel_.x() = cos_yaw * raw_dog_vel_.x() - sin_yaw * raw_dog_vel_.y();
        last_dog_vel_.y() = sin_yaw * raw_dog_vel_.x() + cos_yaw * raw_dog_vel_.y();
        last_dog_vel_.z() = raw_dog_vel_.z();
    } else {
        // 水平速度: 0.7*旧 + 0.3*新 (强低通,抑制狗通信抖动)
        raw_dog_vel_.x() = 0.7 * raw_dog_vel_.x() + 0.3 * current_raw_dog_vel.x();
        raw_dog_vel_.y() = 0.7 * raw_dog_vel_.y() + 0.3 * current_raw_dog_vel.y();
        // 垂直速度: 直接使用新值(狗报告高度变化较可靠)
        raw_dog_vel_.z() = current_raw_dog_vel.z();

        // Yaw 一阶低通: α=0.5 的指数平滑
        double delta_yaw = normalizeAngle(current_raw_dog_yaw - raw_dog_yaw_);
        raw_dog_yaw_ += 0.5 * delta_yaw;
    }

    // 发布处理后的狗位姿(在 subscribe 后立即发布,保证低延迟)
    publishProcessedDogPos();
}


// ============================================================================
// 五、狗运动状态估计 — 角速度与加速度
// ============================================================================

/**
 * updateDogYawRate — 从连续两帧的 yaw 差估计狗的旋转角速度
 *
 * 原理: ω = Δθ / dt, 然后用一阶低通滤波平滑
 *
 * @param delta_yaw 相邻帧 yaw 差(已归一化到 [-π,π])
 */
void DogPosProcessor::updateDogYawRate(double delta_yaw) {
    double current_time = ros::Time::now().toSec();

    if (!dog_yaw_rate_initialized_) {
        last_dog_yaw_time_ = current_time;
        final_dog_yaw_rate_ = 0.0;
        dog_yaw_rate_initialized_ = true;
    } else {
        double dt = current_time - last_dog_yaw_time_;
        if (dt > 0.001) {  // 至少 1ms 间隔,防止除零
            double instant_yaw_rate = delta_yaw / dt;

            // 一阶低通: α=0.3 的指数平滑
            final_dog_yaw_rate_ = (1.0 - yaw_rate_filter_gain_) * final_dog_yaw_rate_ +
                           yaw_rate_filter_gain_ * instant_yaw_rate;

            last_dog_yaw_time_ = current_time;
        }
    }
}

/**
 * updateDogAcc — 从速度差分估计狗的加速度
 *
 * 原理: a = Δv / dt, 然后用一阶低通滤波平滑
 * 加速度用于 traj_server.cpp 中的前馈控制(提前预判狗的运动趋势)
 *
 * @param delta_vel 相邻帧速度差
 * @param dt        时间间隔(<=0 时自动计算)
 */
void DogPosProcessor::updateDogAcc(const Eigen::Vector3d& delta_vel, double dt) {
    double current_time = ros::Time::now().toSec();

    if (!dog_acc_initialized_) {
        last_dog_vel_time_ = current_time;
        final_dog_acc_ = Eigen::Vector3d::Zero();
        dog_acc_initialized_ = true;
    } else {
        double actual_dt;
        if (dt > 0.001) {
            actual_dt = dt;
        } else {
            actual_dt = current_time - last_dog_vel_time_;
        }

        if (actual_dt > 0.001) {
            Eigen::Vector3d instant_acc = delta_vel / actual_dt;

            // 一阶低通: acc_filter_gain_=0.3 (每个轴分别滤波)
            final_dog_acc_.x() = (1.0 - acc_filter_gain_) * final_dog_acc_.x() +
                          acc_filter_gain_ * instant_acc.x();
            final_dog_acc_.y() = (1.0 - acc_filter_gain_) * final_dog_acc_.y() +
                          acc_filter_gain_ * instant_acc.y();
            final_dog_acc_.z() = (1.0 - acc_filter_gain_) * final_dog_acc_.z() +
                          acc_filter_gain_ * instant_acc.z();

            last_dog_vel_time_ = current_time;
        }
    }
}


// ============================================================================
// 六、ROS 回调 — 外部数据订阅
// ============================================================================

/**
 * targetCallback — 目标(狗)视觉位姿订阅
 *
 * 订阅 /target_ekf_odom (来自 read.cpp 的视觉检测结果):
 *   - position: 狗在世界系下的位置
 *   - orientation.w: 狗在世界系下的 yaw
 *   - orientation.x: 狗在世界系下的 pitch
 *   - orientation.y: 狗在世界系下的 roll
 */
void DogPosProcessor::targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    target_ekf_last_time_ = msg->header.stamp;

    target_dog_yaw_ = msg->pose.pose.orientation.w;
    target_dog_pitch_ = msg->pose.pose.orientation.x;
    target_dog_roll_ = msg->pose.pose.orientation.y;

    target_dog_pos_.x() = msg->pose.pose.position.x;
    target_dog_pos_.y() = msg->pose.pose.position.y;
    target_dog_pos_.z() = msg->pose.pose.position.z;

    target_count_++;
}

/**
 * vinsCallback — VINS 无人机位姿订阅
 *
 * 从四元数计算:
 *   - vins_yaw_: 无人机当前航向角
 *   - R_wb_: 世界系到机体系的旋转矩阵 (用于 AOA 坐标变换)
 *   - vins_pos_: 无人机在世界系下的位置
 */
void DogPosProcessor::vinsCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    double q_w = msg->pose.pose.orientation.w;
    double q_x = msg->pose.pose.orientation.x;
    double q_y = msg->pose.pose.orientation.y;
    double q_z = msg->pose.pose.orientation.z;

    // 从四元数计算偏航角: yaw = atan2(2(qw*qz+qx*qy), 1-2(qy²+qz²))
    double siny_cosp = 2.0 * (q_w * q_z + q_x * q_y);
    double cosy_cosp = 1.0 - 2.0 * (q_y * q_y + q_z * q_z);
    vins_yaw_ = std::atan2(siny_cosp, cosy_cosp);

    R_wb_ = Eigen::Quaterniond(q_w, q_x, q_y, q_z).toRotationMatrix();

    vins_pos_.x() = msg->pose.pose.position.x;
    vins_pos_.y() = msg->pose.pose.position.y;
    vins_pos_.z() = msg->pose.pose.position.z;

    vins_count_++;
}

/**
 * takeoffCallback — 起飞/降落信号订阅
 * 起飞时重置所有初始化状态,等待新的 yaw_diff_preset 来重新对齐
 */
void DogPosProcessor::takeoffCallback(const quadrotor_msgs::TakeoffLand::ConstPtr& msg) {
    if (msg->takeoff_land_cmd == 1) {  // 起飞命令
        ROS_INFO("Takeoff detected, yaw offset will be reinitialized");
        initialized_ = false;
        precise_yaw_offset_ready_ = false;
        precise_pos_offset_ready_ = false;
        trigger_received_ = false;
    }
}

/**
 * yawDiffCallback — 接收上一次降落时保存的 yaw 差值
 *
 * 原理: 降落时 traj_server.cpp 发布 /yaw_diff_preset (飞机和狗的 yaw 差),
 *       本节点保存该值,下次起飞时用它直接初始化 yaw_offset,
 *       避免每次起飞都要重新迭代收敛。
 */
void DogPosProcessor::yawDiffCallback(const std_msgs::Float64::ConstPtr& msg) {
    if (saved_yaw_diff_ == nullptr) {
        saved_yaw_diff_ = new double(msg->data);
    } else {
        *saved_yaw_diff_ = msg->data;
    }
    ROS_INFO("Received yaw diff: %.1f deg", msg->data * 180.0 / M_PI);
}

void DogPosProcessor::testResetCallback(const std_msgs::Bool::ConstPtr& msg) {
    ROS_INFO("Test reset signal received, resetting initialized to False");
    initialized_ = false;
    precise_yaw_offset_ready_ = false;
    precise_pos_offset_ready_ = false;
}

void DogPosProcessor::triggerCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    trigger_received_ = true;
    ROS_INFO("Trigger received - starting iteration");
}


// ============================================================================
// 七、状态检查 (10Hz) — 监控各数据源的健康状态
// ============================================================================

/**
 * statusCheckCallback — 检查所有传感器数据是否在持续更新
 *
 * 原理: 通过计数器比较来判断是否有新数据到达
 *   如果连续 N 次检查没有新包 → 标记该数据源为丢失 → 防止使用过期数据
 */
void DogPosProcessor::statusCheckCallback(const ros::TimerEvent& event) {
    // 检查 dog_pos_received 状态
    if (raw_dog_pos_count_ != last_raw_dog_pos_count_) {
        raw_dog_pos_received_ = true;
        last_raw_dog_pos_count_ = raw_dog_pos_count_;
        last_dog_pos_timer_ = 0;
    } else {
        last_dog_pos_timer_++;
        if (last_dog_pos_timer_ >= 1) {
            raw_dog_pos_received_ = false;
            // 狗数据丢失时重置运动估计(角速度和加速度)
            dog_yaw_rate_initialized_ = false;
            dog_vel_initialized_ = false;
            dog_acc_initialized_ = false;
        }
    }

    // 检查 target_receive 状态(视觉检测数据)
    if (target_count_ != last_target_count_) {
        last_target_timer_ ++;
        if (last_target_timer_ >= 1) {
            target_receive_ = true;
        }
        last_target_count_ = target_count_;
        last_target_loss_timer_ = 0;
    } else {
        last_target_loss_timer_++;
        if (last_target_loss_timer_ >= 1) {
            target_receive_ = false;
        }
        last_target_timer_ = 0;
    }

    // 检查 vins_received 状态
    if (vins_count_ != last_vins_count_) {
        vins_received_ = true;
        last_vins_count_ = vins_count_;
        last_vins_timer_ = 0;
    } else {
        last_vins_timer_++;
        if (last_vins_timer_ >= 5) {
            vins_received_ = false;
        }
    }

    // 检查 aoa_received 状态
    if (aoa_count_ != last_aoa_count_) {
        aoa_received_ = true;
        last_aoa_count_ = aoa_count_;
        last_aoa_timer_ = 0;
    } else {
        last_aoa_timer_++;
        if (last_aoa_timer_ >= 5) {
            aoa_received_ = false;
        }
    }
}


// ============================================================================
// 八、主处理回调 (50Hz) — 核心坐标对齐逻辑
// ============================================================================

/**
 * processCallback — 坐标对齐主循环
 *
 * 分为以下几个阶段:
 *
 * 【阶段 1: 初始化 offset】(当 initialized_=false 且有数据时)
 *   使用保存的 yaw_diff (来自上次降落) 或默认值 0,
 *   计算初始的 yaw_offset 和 pos_offset:
 *     yaw_offset = raw_dog_yaw - (vins_yaw + diff)
 *     pos_offset = raw_dog_pos - R(yaw_offset) * vins_pos
 *
 * 【阶段 2: Yaw offset 迭代维护】
 *   当视觉检测(target)和狗通信数据(raw)同时可用时:
 *     current_yaw_offset = raw_dog_yaw - target_dog_yaw
 *     yaw_offset_diff = current_yaw_offset - yaw_offset (归一化到[-π,π])
 *     yaw_offset += yaw_filter_gain * yaw_offset_diff   (一阶低通)
 *
 *   收敛判据: |yaw_offset_diff| < 5° → precise_yaw_offset_ready
 *   发散判据: |yaw_offset_diff| > 45° 连续 5 次 → not ready
 *
 * 【阶段 3: Pos offset 迭代维护】
 *   当 yaw ready 后,用旋转后的 target 位置和 raw 位置的差来更新 pos_offset:
 *     raw_dog_pos - R(yaw_offset) * (target_pos + camera_offset * vel)
 *     pos_offset += pos_filter_gain * (current_pos_offset - pos_offset)
 *
 *   收敛判据: |pos_offset_diff| < 5cm → precise_pos_offset_ready
 *   发散判据: |pos_offset_diff| > 30cm 连续 5 次 → not ready
 *
 * 【阶段 4: AOA 辅助修正】
 *   当 yaw ready 且 AOA 可用且距离 > 3m 时,
 *   利用 UWB 单锚点的距离+角度测量,通过几何约束解算目标世界系位置,
 *   然后用解算结果修正 pos_offset。
 */
void DogPosProcessor::processCallback(const ros::TimerEvent& event) {

    // ========================================================================
    // 阶段 1: 初始化 offset — 起飞时的坐标系对齐
    // ========================================================================
    if (!initialized_ && raw_dog_pos_received_ && vins_received_) {
        // 使用保存的 yaw_diff (来自上次降落的 /yaw_diff_preset)
        double diff = 0.0;
        if (saved_yaw_diff_ != nullptr) {
            diff = *saved_yaw_diff_;
            delete saved_yaw_diff_;
            saved_yaw_diff_ = nullptr;
        } else {
            diff = 0.0;  // 首次起飞无历史数据,使用默认值 0
        }

        // yaw_offset = raw_dog_yaw - (vins_yaw + diff)
        // raw_dog_yaw 是狗系下狗的航向(在dog_pos中报告)
        // vins_yaw 是飞机的航向
        // diff 是上次降落时记录的飞机和狗的 yaw 差
        yaw_offset_ = normalizeAngle(raw_dog_yaw_ - (vins_yaw_ + diff));

        // 位置偏移: raw_dog_pos - R(yaw_offset) * vins_pos
        // R(yaw_offset) 将 VINS 位置旋转到狗坐标系下
        // pos_offset = 狗报告的自身位置 - 旋转后的 VINS 位置
        double cos_yaw = std::cos(yaw_offset_);
        double sin_yaw = std::sin(yaw_offset_);
        Eigen::Vector3d rotated_vins_pos(
            cos_yaw * vins_pos_.x() - sin_yaw * vins_pos_.y(),
            sin_yaw * vins_pos_.x() + cos_yaw * vins_pos_.y(),
            vins_pos_.z()
        );
        Eigen::Vector3d tmp_p(
            raw_dog_pos_->pose.pose.position.x,
            raw_dog_pos_->pose.pose.position.y,
            raw_dog_pos_->pose.pose.position.z
        );
        pos_offset_ = tmp_p - rotated_vins_pos;

        ROS_INFO("Reinitialized offsets: yaw_offset=%.1f deg, pos_offset=[%.3f, %.3f, %.3f]",
                 yaw_offset_ * 180.0 / M_PI, pos_offset_.x(), pos_offset_.y(), pos_offset_.z());

        // 清空历史记录,重新开始
        yaw_offset_history_.clear();
        pos_offset_history_.clear();

        // 初次使用预设值时直接认为 ready (不需要迭代等待)
        initialized_ = true;
        precise_pos_offset_ready_ = true;
        precise_yaw_offset_ready_ = true;
    }

    // ========================================================================
    // 阶段 2 & 3: Yaw offset 和 Pos offset 迭代维护
    // ========================================================================
    if (target_receive_ && raw_dog_pos_received_ && vins_received_) {

        // ----- 2a. Yaw offset 迭代 -----
        // 当前估计的 yaw_offset = raw_dog_yaw - target_dog_yaw
        // (狗报告的航向 - 视觉检测的航向 = 两个坐标系的旋转差)
        double current_yaw_offset = normalizeAngle(raw_dog_yaw_ - target_dog_yaw_);
        double yaw_offset_diff = normalizeAngle(current_yaw_offset - yaw_offset_);
        yaw_offset_diff = normalizeAngle(yaw_offset_diff);  // 两次归一化确保不越界

        // 一阶低通滤波: yaw_offset 逐渐收敛到实际值
        yaw_offset_ += yaw_filter_gain_ * yaw_offset_diff;

        // 记录历史值(用于方差估计)
        yaw_offset_history_.push_back(yaw_offset_);
        if (yaw_offset_history_.size() > offset_history_max_size_) {
            yaw_offset_history_.pop_front();
        }

        // 收敛/发散判据 — 带滞后防止抖动
        if (std::abs(yaw_offset_diff) < yaw_stable_threshold_) {
            if (!precise_yaw_offset_ready_) {
                ROS_INFO("precise_yaw_offset_ready!");
                if (precise_pos_offset_ready_) {
                    initialized_ = true;
                }
            }
            precise_yaw_offset_ready_ = true;
        }

        // 发散检测: yaw offset 变化超过 45° 连续 5 次 → 标记不可靠
        if (std::abs(yaw_offset_diff) > yaw_exceed_threshold_) {
            yaw_exceed_timer_++;
            if (yaw_exceed_timer_ > yaw_exceed_max_count_) {
                ROS_WARN("yaw_offset_diff too large!");
                precise_yaw_offset_ready_ = false;
                yaw_exceed_timer_ = 0;
            }
        } else {
            yaw_exceed_timer_ = 0;
        }

        // ----- 3a. Pos offset 迭代 -----
        if (precise_yaw_offset_ready_) {
            Eigen::Vector3d raw_dog_pos(
                raw_dog_pos_->pose.pose.position.x,
                raw_dog_pos_->pose.pose.position.y,
                raw_dog_pos_->pose.pose.position.z
            );

            // 前馈补偿: target_dog_pos 加上 camera_offset * velocity 的提前量
            // 物理含义: 相机安装在飞机上,拍照点到目标几何中心有 camera_offset 的距离,
            //          目标在运动时需要考虑这段时间内的位移
            // 加上了加速度的二次项: 0.5 * camera_offset² * acc (匀加速运动补偿)
            Eigen::Vector3d target_dog_pos_with_ff(
                target_dog_pos_.x() + camera_offset_ * final_dog_vel_.x()
                    + 0.5 * camera_offset_ * camera_offset_ * final_dog_acc_.x(),
                target_dog_pos_.y() + camera_offset_ * final_dog_vel_.y()
                    + 0.5 * camera_offset_ * camera_offset_ * final_dog_acc_.y(),
                target_dog_pos_.z()
            );

            // 将加上前馈补偿后的 target 旋转到狗坐标系
            double cos_yaw = std::cos(yaw_offset_);
            double sin_yaw = std::sin(yaw_offset_);
            Eigen::Vector3d rotated_target(
                cos_yaw * target_dog_pos_with_ff.x() - sin_yaw * target_dog_pos_with_ff.y(),
                sin_yaw * target_dog_pos_with_ff.x() + cos_yaw * target_dog_pos_with_ff.y(),
                target_dog_pos_with_ff.z()
            );

            // 当前 pos offset = raw_dog_pos - rotated_target (在狗系下做差)
            Eigen::Vector3d current_pos_offset = raw_dog_pos - rotated_target;

            // 一阶低通滤波: pos_offset 逐渐收敛
            Eigen::Vector3d pos_offset_diff = current_pos_offset - pos_offset_;
            pos_offset_ += pos_filter_gain_ * pos_offset_diff;

            // 记录历史
            pos_offset_history_.push_back(pos_offset_);
            if (pos_offset_history_.size() > offset_history_max_size_) {
                pos_offset_history_.pop_front();
            }

            double pos_offset_diff_norm = (current_pos_offset - pos_offset_).norm();

            // 收敛判据
            if (pos_offset_diff_norm < pos_stable_threshold_) {
                if (!precise_pos_offset_ready_) {
                    ROS_INFO("precise_pos_offset_ready!");
                    if (precise_yaw_offset_ready_) {
                        initialized_ = true;
                    }
                    // 位置收敛时重置卡尔曼滤波器(避免历史状态污染)
                    ROS_INFO("Position converged, resetting Kalman filter");
                    kf_.reset();
                    last_kf_time_ = ros::Time();
                    yaw_filter_initialized_ = false;
                    filtered_yaw_ = 0.0;
                }
                precise_pos_offset_ready_ = true;
            }

            // 发散检测
            if (pos_offset_diff_norm > pos_exceed_threshold_) {
                pos_exceed_timer_++;
                if (pos_exceed_timer_ > pos_exceed_max_count_) {
                    ROS_WARN("pos_offset_diff too large!");
                    precise_pos_offset_ready_ = false;
                    pos_exceed_timer_ = 0;
                }
            } else {
                pos_exceed_timer_ = 0;
            }
        }
    }

    // ========================================================================
    // 阶段 4: AOA 辅助位置修正
    // ========================================================================
    // 前提条件:
    //   1. Yaw offset 已 ready
    //   2. VINS 位姿可用
    //   3. AOA 传感器可用 (UWB 单锚点距离+角度)
    // AOA 会给一个bearing角度和一个斜距，分别表示狗在飞机水平面的方向和飞机到狗的直线距离
    //   4. 飞机朝向与狗的夹角在 ±45° 内 (AOA 传感器视场角限制)
    //
    // 几何问题定义:
    //   已知:
    //     - 飞机在世界系中的姿态 R_wb
    //     - 飞机为坐标系原点
    //     - 目标相对飞机的高度差 Δz
    //     - 目标相对飞机的水平距离 aoa_distance_horizontal
    //     - 目标在机体系下的 yaw 角 aoa_angle
    //   求:
    //     - 目标在世界系下的位置 aoa_pos_world
    //
    // 求解策略: yaw 平面 ∩ 高度平面 = 直线 → 沿直线走水平距离
    if (precise_yaw_offset_ready_ && vins_received_ && aoa_received_) {
        Eigen::Vector3d dog_vec = final_dog_pos_ - vins_pos_;
        //这里是使用上一帧计算出来的狗在世界系下的位置，和VIN上的朝向做合理性校验
        double heading_to_dog = std::atan2(dog_vec.y(), dog_vec.x());//反三角函数求角度
        double heading_diff = normalizeAngle(heading_to_dog - vins_yaw_);
        if (std::abs(heading_diff) > M_PI / 4.0) {  // 45 度
            return;
        }

        // 用 flow_z(光流高度)修正高度差
        double height_diff = final_dog_pos_.z() - vins_pos_.z();

        // 距离约束: AOA 斜距必须 ≥ 高度差(否则无解)
        if (aoa_distance_ < std::abs(height_diff)) {
            return;
        }

        // 勾股定理求水平距离: horizontal = sqrt(slant² - vertical²)
        double aoa_distance_horizontal = std::sqrt(aoa_distance_ * aoa_distance_ - height_diff * height_diff);

        if (aoa_distance_horizontal < aoa_min_distance_) {
            return;
        }

        // ===== 几何求解: 单距离+单角度 → 世界系 3D 位置 =====

        // Step 0: 基础向量
        Eigen::Vector3d ez(0.0, 0.0, 1.0);  // 世界系 z 轴

        // 狗在机体系 yaw 平面内的 bearing 方向(水平面内)
        Eigen::Vector3d bearing_b(
            std::cos(aoa_angle_),
            std::sin(aoa_angle_),
            0.0
        );

        // Step 1: 构造 yaw 平面的法向
        // yaw 平面 = 经过原点、包含 bearing_b + 机体系 z 轴的平面
        // 法向 = bearing × ez (在机体系下)
        Eigen::Vector3d n_b = bearing_b.cross(ez);

        //由机体系转到世界系：R_wb*n_b
        Eigen::Vector3d n_w = R_wb_ * n_b;
        n_w.normalize();

        // Step 2: yaw 平面与水平面(z=const)的交线
        // 交线方向 = 两个平面法向的叉乘: d ∝ n × ez
        Eigen::Vector3d d_w = -n_w.cross(ez);

        if (d_w.norm() < 1e-6) {
            return;  // yaw 平面平行于水平面(退化,无唯一解)
            //这种情况只有在飞机侧翻90度的情况出现
        }
        //

        Eigen::Vector3d d_hat = d_w.normalized();

        // Step 3 & 4: 构造最近点 p0
        // 目标: 在交线上找到距离原点最近的点
        //
        // 思路:
        //   - 从 p_start = Δz·ez 出发(满足高度约束)
        //   - 沿 n_h 方向修正,使 p0 落在 yaw 平面内
        //   - n_h = n_w - (n_w·ez)·ez = yaw 平面法向在水平面的投影
        //
        // 闭式解: p0 = Δz·ez - α·n_h, α = (n_z·Δz) / |n_h|²

        double n_z = n_w.dot(ez);  // yaw 平面法向的 z 分量

        Eigen::Vector3d n_h = n_w - n_z * ez;  // 法向的水平分量
        double n_h_sq = n_h.squaredNorm();

        if (n_h_sq < 1e-8) {
            return;
        }

        // 一步闭式解，即yaw平面和狗所在水平面的交线离飞机原点最近的点，这里的p0向量是从飞机原点指向交线的最近点的向量
        Eigen::Vector3d p0 = height_diff * ez - (n_z * height_diff / n_h_sq) * n_h;

        // Step 5: 在交线上施加水平距离约束
        // 交线: L(t) = p0 + t·d_hat (参数方程)
        // 约束: ||L(t)_horizontal|| = aoa_distance_horizontal
        // p0 的水平分量
        Eigen::Vector2d p0_h(p0.x(), p0.y());
        double p0_h_norm = p0_h.norm();

        double inside = aoa_distance_horizontal * aoa_distance_horizontal - p0_h_norm * p0_h_norm;

        if (inside < 0.0) {
            return;  // 水平圆与交线无交点
        }

        double t = std::sqrt(inside);  // 沿 bearing 正方向的唯一物理解

        // 目标相对飞机的位置(世界系)
        Eigen::Vector3d target_rel = p0 + t * d_hat;

        // Step 6: 转为世界系绝对坐标(目标相对于世界系原点）
        Eigen::Vector3d aoa_pos_world = vins_pos_ + target_rel;

        // 将 AOA 估计位置旋转到狗坐标系(使用 yaw_offset)
        //Vins所在的系就是世界系，不是机体系
        double cos_yaw = std::cos(yaw_offset_);
        double sin_yaw = std::sin(yaw_offset_);
        double rotated_aoa_x = cos_yaw * (aoa_pos_world.x() - vins_pos_.x()) - sin_yaw * (aoa_pos_world.y() - vins_pos_.y());
        double rotated_aoa_y = sin_yaw * (aoa_pos_world.x() - vins_pos_.x()) + cos_yaw * (aoa_pos_world.y() - vins_pos_.y());
        Eigen::Vector3d rotated_aoa_pos(rotated_aoa_x, rotated_aoa_y, final_dog_pos_.z());

        // 与 raw_dog_pos 比较,计算 pos_offset 修正量
        Eigen::Vector3d raw_dog_pos(
            raw_dog_pos_->pose.pose.position.x,
            raw_dog_pos_->pose.pose.position.y,
            raw_dog_pos_->pose.pose.position.z
        );
        Eigen::Vector3d current_pos_offset = raw_dog_pos - rotated_aoa_pos;

        // Debug 发布 AOA 估计位置
        nav_msgs::Odometry aoa_debug_msg;
        aoa_debug_msg.header.stamp = ros::Time::now();
        aoa_debug_msg.header.frame_id = "world";
        aoa_debug_msg.pose.pose.position.x = aoa_pos_world.x();
        aoa_debug_msg.pose.pose.position.y = aoa_pos_world.y();
        aoa_debug_msg.pose.pose.position.z = aoa_pos_world.z();
        aoa_dog_pos_debug_pub_.publish(aoa_debug_msg);

        // 位置偏移滤波: x 和 y 分开迭代(方向可能不同), z 不迭代(高度不变)
        // 每个轴独立判断: 只有在该轴有足够大的修正量时才更新
        Eigen::Vector3d pos_offset_diff = current_pos_offset - pos_offset_;

        // x 方向迭代
        double diff_x = pos_offset_diff.x();
        if (std::abs(diff_x) > 1e-6) {
            double step_x = aoa_pos_filter_gain_ * std::abs(diff_x);
            step_x = std::min(step_x, aoa_pos_step_limit_);  // 步长上限
            pos_offset_.x() += (diff_x > 0 ? step_x : -step_x);
        }

        // y 方向迭代
        double diff_y = pos_offset_diff.y();
        if (std::abs(diff_y) > 1e-6) {
            double step_y = aoa_pos_filter_gain_ * std::abs(diff_y);
            step_y = std::min(step_y, aoa_pos_step_limit_);
            pos_offset_.y() += (diff_y > 0 ? step_y : -step_y);
        }
        // z 方向不迭代,保持不变
    }
}


// ============================================================================
// 九、发布处理后的狗位姿 (publishProcessedDogPos)
// ============================================================================

/**
 * publishProcessedDogPos — 将狗系下的 raw 数据转换为世界系并发布
 *
 * 核心变换:
 *
 *   1. 位置:
 *      corrected_pos = raw_pos - pos_offset          (① 减去位置偏移)
 *      world_pos = R(-yaw_offset) * corrected_pos    (② 旋转到世界系)
 *
 *   2. 速度:
 *      target_yaw = normalizeAngle(raw_dog_yaw - yaw_offset)
 *      world_vel = R(target_yaw) * raw_vel           (旋转到世界系)
 *
 *   3. Yaw:
 *      world_yaw = normalizeAngle(raw_dog_yaw - yaw_offset)
 *
 *   发布的消息格式 (nav_msgs::Odometry):
 *     - pose.position:         世界系狗位置 [x, y, z]
 *     - pose.orientation.w:    precise_pos_offset_ready (1.0 = ready)
 *     - pose.orientation.x:    precise_yaw_offset_ready (1.0 = ready)
 *     - pose.orientation.y:    世界系狗加速度x
 *     - pose.orientation.z:    世界系狗加速度y
 *     - twist.linear:          世界系狗速度 [vx, vy, vz]
 *     - twist.angular.x:       世界系狗yaw
 *     - twist.angular.y:       狗旋转角速度
 *     - twist.angular.z:       保留(0)
 */
void DogPosProcessor::publishProcessedDogPos() {
    if (raw_dog_pos_ == nullptr) return;

    nav_msgs::Odometry msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "world";

    // ===== 1. 位置变换 =====
    Eigen::Vector3d raw_pos(
        raw_dog_pos_->pose.pose.position.x,
        raw_dog_pos_->pose.pose.position.y,
        raw_dog_pos_->pose.pose.position.z
    );

    // ① 减去 pos_offset (狗坐标系下偏移校正)
    Eigen::Vector3d corrected_pos = raw_pos - pos_offset_;

    // ② 旋转 yaw_offset: R(-yaw_offset) * corrected_pos → 世界系
    double yaw_diff = normalizeAngle(-yaw_offset_);
    double cos_yaw = std::cos(yaw_diff);
    double sin_yaw = std::sin(yaw_diff);
    double rotated_x = cos_yaw * corrected_pos.x() - sin_yaw * corrected_pos.y();
    double rotated_y = sin_yaw * corrected_pos.x() + cos_yaw * corrected_pos.y();

    // ===== 2. 速度变换 =====
    double target_yaw = normalizeAngle(raw_dog_yaw_ - yaw_offset_);
    cos_yaw = std::cos(target_yaw);
    sin_yaw = std::sin(target_yaw);
    double rotated_vx = cos_yaw * raw_dog_vel_.x() - sin_yaw * raw_dog_vel_.y();
    double rotated_vy = sin_yaw * raw_dog_vel_.x() + cos_yaw * raw_dog_vel_.y();

    // ===== 3. 填充内部状态 =====
    final_dog_pos_.x() = rotated_x;
    final_dog_pos_.y() = rotated_y;
    final_dog_pos_.z() = corrected_pos.z();

    final_dog_vel_.x() = rotated_vx;
    final_dog_vel_.y() = rotated_vy;
    final_dog_vel_.z() = raw_dog_vel_.z();
    final_dog_yaw_ = target_yaw;

    // ===== 4. 卡尔曼滤波 (可选,当前默认关闭) =====
    // 关闭原因: 实验中发现在狗做急转弯时 KF 预测(CV 模型)会在最高点过冲,
    //          导致位置出现偏差。改为直接使用原始值,配合下游 PID 的 D 项
    //          来抑制噪声。
    if (kf_enabled_ && precise_pos_offset_ready_) {
        double current_time = ros::Time::now().toSec();

        if (last_kf_time_.isZero()) {
            last_kf_time_ = ros::Time::now();
            double dt = 0.05;
        } else {
            double dt = (ros::Time::now() - last_kf_time_).toSec();
            if (dt <= 0) {
                dt = 0.05;
            }

            if (dt > kf_timeout_) {
                ROS_WARN("Kalman filter timeout (dt=%.2fs > %.2fs), resetting filter", dt, kf_timeout_);
                kf_.reset();
                yaw_filter_initialized_ = false;
                filtered_yaw_ = 0.0;
                dt = 0.05;
            }

            auto filtered = kf_.filter(final_dog_pos_, final_dog_vel_, dt);
            final_dog_pos_ = filtered.first;
            final_dog_vel_ = filtered.second;

            if (!yaw_filter_initialized_) {
                filtered_yaw_ = final_dog_yaw_;
                yaw_filter_initialized_ = true;
            } else {
                double yaw_diff_kf = normalizeAngle(final_dog_yaw_ - filtered_yaw_);
                filtered_yaw_ = normalizeAngle(
                    filtered_yaw_ + yaw_filter_gain_kf_ * yaw_diff_kf
                );
            }

            final_dog_pos_ = filtered.first;
            final_dog_vel_ = filtered.second;
            final_dog_yaw_ = filtered_yaw_;

            last_kf_time_ = ros::Time::now();
        }
    }

    // ===== 5. 计算角速度和加速度 =====
    // 角速度: ω = Δθ / Δt, 一阶低通
    double delta_yaw = normalizeAngle(final_dog_yaw_ - last_final_dog_yaw_);
    updateDogYawRate(delta_yaw);

    // 加速度: a = Δv / Δt, 一阶低通
    Eigen::Vector3d delta_vel = final_dog_vel_ - last_final_dog_vel_;
    updateDogAcc(delta_vel, default_dt);

    // 更新上一帧值
    last_final_dog_yaw_ = final_dog_yaw_;
    last_final_dog_pitch_ = target_dog_pitch_;
    last_final_dog_roll_ = target_dog_roll_;
    last_final_dog_vel_ = final_dog_vel_;

    // ===== 6. 填充 ROS 消息 =====
    msg.pose.pose.position.x = final_dog_pos_.x();
    msg.pose.pose.position.y = final_dog_pos_.y();
    msg.pose.pose.position.z = final_dog_pos_.z();

    // 状态编码(复用 orientation 的 4 个分量):
    //   w: precise_pos_offset_ready
    //   x: precise_yaw_offset_ready
    //   y: 加速度 x (世界系)
    //   z: 加速度 y (世界系)
    msg.pose.pose.orientation.w = precise_pos_offset_ready_ ? 1.0 : 0.0;
    msg.pose.pose.orientation.x = precise_yaw_offset_ready_ ? 1.0 : 0.0;
    msg.pose.pose.orientation.y = final_dog_acc_.x();
    msg.pose.pose.orientation.z = final_dog_acc_.y();

    // 世界系速度
    msg.twist.twist.linear.x = final_dog_vel_.x();
    msg.twist.twist.linear.y = final_dog_vel_.y();
    msg.twist.twist.linear.z = final_dog_vel_.z();

    // 世界系 yaw 和角速度
    msg.twist.twist.angular.x = final_dog_yaw_;
    msg.twist.twist.angular.y = final_dog_yaw_rate_;

    msg.twist.twist.angular.z = 0.0;

    dog_pos_pub_.publish(msg);
}


// ============================================================================
// 十、AOA 和光流回调
// ============================================================================

void DogPosProcessor::aoaCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 单距离 + 单角度: position.x = 距离(m), orientation.x = 角度(弧度)
    aoa_distance_ = msg->pose.pose.position.x;
    aoa_angle_ = msg->pose.pose.orientation.x;
    aoa_count_++;
}

void DogPosProcessor::flowCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 光流高度: position.z = 相对高度(m)
    flow_z_ = msg->pose.pose.position.z;
}


// ============================================================================
// 十一、spin 入口
// ============================================================================

void DogPosProcessor::spin() {
    ros::spin();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "dog_pos_processor");
    DogPosProcessor processor;
    processor.spin();
    return 0;
}
