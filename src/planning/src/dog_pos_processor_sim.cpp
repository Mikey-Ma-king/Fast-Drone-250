#include "dog_pos_processor.h"
#include <ros/ros.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>

namespace {

// 与 dog_pos_processor_kf 对齐的 LKF 参数（原 sim：外参 gain=0.04，输出 KF Q=0.01 R=0.1）
constexpr double kTimerDt = 0.05;
constexpr double kExtrinsicAlpha = 0.04;
constexpr double kExtrinsicMeasStd = 0.05;
constexpr double kExtrinsicMeasVar = kExtrinsicMeasStd * kExtrinsicMeasStd;
constexpr double kExtrinsicQ = kExtrinsicAlpha * kExtrinsicMeasVar / (1.0 - kExtrinsicAlpha);
constexpr double kExtrinsicR = kExtrinsicMeasVar;
constexpr double kOutputProcessNoise = 0.01;
constexpr double kOutputMeasNoise = 0.1;
constexpr double kOutputQRate = kOutputProcessNoise / kTimerDt;
constexpr double kExtrinsicQRate = kExtrinsicQ / kTimerDt;

void logIterMsThrottled(const char* tag,
                          const std::chrono::steady_clock::time_point& t0) {
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    ROS_INFO_THROTTLE(1.0, "%s: one iter %.3f ms", tag, ms);
}

class ScalarLkf {
public:
    ScalarLkf(double q_rate, double r) : q_rate_(q_rate), r_(r), x_(0.0), p_(r), initialized_(false) {}

    void reset(double x = 0.0) {
        x_ = x;
        p_ = r_;
        initialized_ = false;
    }

    double state() const { return x_; }

    void update(double z, double dt, bool angle_wrap = false) {
        if (!initialized_) {
            x_ = z;
            p_ = r_;
            initialized_ = true;
            return;
        }
        if (dt <= 0.0) dt = kTimerDt;
        p_ += q_rate_ * dt;
        const double innov = angle_wrap ? wrapAngle(z - x_) : (z - x_);
        const double k = p_ / (p_ + r_);
        x_ += k * innov;
        if (angle_wrap) {
            x_ = wrapAngle(x_);
        }
        p_ = (1.0 - k) * p_;
    }

private:
    static double wrapAngle(double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    double q_rate_, r_;
    double x_, p_;
    bool initialized_;
};

class Vec3Lkf {
public:
    Vec3Lkf(double q_rate, double r) : fx_(q_rate, r), fy_(q_rate, r), fz_(q_rate, r) {}

    void reset(const Eigen::Vector3d& x = Eigen::Vector3d::Zero()) {
        fx_.reset(x.x());
        fy_.reset(x.y());
        fz_.reset(x.z());
    }

    Eigen::Vector3d state() const {
        return Eigen::Vector3d(fx_.state(), fy_.state(), fz_.state());
    }

    void update(const Eigen::Vector3d& z, double dt) {
        fx_.update(z.x(), dt);
        fy_.update(z.y(), dt);
        fz_.update(z.z(), dt);
    }

private:
    ScalarLkf fx_, fy_, fz_;
};

ScalarLkf s_yaw_offset_lkf(kExtrinsicQRate, kExtrinsicR);
Vec3Lkf s_pos_offset_lkf(kExtrinsicQRate, kExtrinsicR);

void resetExtrinsicLkf(double yaw_off, const Eigen::Vector3d& pos_off) {
    s_yaw_offset_lkf.reset(yaw_off);
    s_pos_offset_lkf.reset(pos_off);
}

}  // namespace

/*
 * Dog Position Processor Module
 * 处理raw_dog_pos数据，维护yaw offset，发布处理后的dog_pos
 */

// KalmanFilter实现
// 简单的卡尔曼滤波器，用于位置和速度滤波
// 初始化卡尔曼滤波器
// 参数:
//     process_noise: 过程噪声协方差
//     measurement_noise: 测量噪声协方差
KalmanFilter::KalmanFilter(double process_noise, double measurement_noise) 
    : initialized_(false) {
    // 状态向量: [x, y, z, vx, vy, vz]
    state_ = Eigen::VectorXd::Zero(6);
    covariance_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;  // 初始协方差矩阵
    
    // 过程噪声协方差矩阵 Q
    Q_ = Eigen::MatrixXd::Identity(6, 6) * process_noise;
    
    // 测量噪声协方差矩阵 R
    R_ = Eigen::MatrixXd::Identity(6, 6) * measurement_noise;
    
    // 状态转移矩阵 F (恒定速度模型，将在predict中根据dt更新)
    F_base_ = Eigen::MatrixXd::Identity(6, 6);  // 基础矩阵
    
    // 观测矩阵 H (直接观测位置和速度)
    H_ = Eigen::MatrixXd::Identity(6, 6);
}

// 预测步骤（Q 随 dt 缩放：Q = kOutputQRate * dt * I，名义 dt=kTimerDt 时等于 kOutputProcessNoise）
void KalmanFilter::predict(double dt) {
    if (!initialized_) return;
    if (dt <= 0.0) dt = kTimerDt;
    
    // 构建状态转移矩阵（恒定速度模型）
    Eigen::MatrixXd F = F_base_;
    F(0, 3) = dt;  // x = x + vx * dt
    F(1, 4) = dt;  // y = y + vy * dt
    F(2, 5) = dt;  // z = z + vz * dt
    
    // 预测状态
    state_ = F * state_;
    
    // 预测协方差
    covariance_ = F * covariance_ * F.transpose() + Q_ * (dt / kTimerDt);
}

// 更新步骤
// 参数:
//     measurement: 测量值 [x, y, z, vx, vy, vz]
void KalmanFilter::update(const Eigen::VectorXd& measurement) {
    if (!initialized_) {
        // 首次初始化
        state_ = measurement;
        initialized_ = true;
        return;
    }
    
    // 计算残差
    Eigen::VectorXd residual = measurement - H_ * state_;
    
    // 计算残差协方差
    Eigen::MatrixXd S = H_ * covariance_ * H_.transpose() + R_;
    
    // 计算卡尔曼增益
    Eigen::MatrixXd K = covariance_ * H_.transpose() * S.inverse();
    
    // 更新状态
    state_ = state_ + K * residual;
    
    // 更新协方差
    covariance_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * covariance_;
}

// 滤波主函数
// 参数:
//     position: 位置 [x, y, z]
//     velocity: 速度 [vx, vy, vz]
//     dt: 时间间隔
// 返回:
//     滤波后的位置和速度
std::pair<Eigen::Vector3d, Eigen::Vector3d> KalmanFilter::filter(
    const Eigen::Vector3d& position, 
    const Eigen::Vector3d& velocity, 
    double dt) {
    // 构建测量向量
    Eigen::VectorXd measurement(6);
    measurement << position, velocity;
    
    // 预测
    predict(dt);
    
    // 更新
    update(measurement);
    
    // 返回滤波后的位置和速度
    return std::make_pair(state_.head<3>(), state_.tail<3>());
}

// 重置滤波器
void KalmanFilter::reset() {
    // state_ = Eigen::VectorXd::Zero(6);
    // covariance_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;
    initialized_ = false;
}

// 外参 LKF（替代原 yaw/pos 线性滤波 gain=0.04）

// 默认时间间隔参数
static const double default_dt = -1.0;  // 默认时间间隔（秒）

// DogPosProcessor实现
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
      kf_(kOutputProcessNoise, kOutputMeasNoise),
      kf_enabled_(true),
      yaw_filter_gain_kf_(0.3),
      filtered_yaw_(0.0),
      yaw_filter_initialized_(false),
      kf_timeout_(1.0),
      trigger_received_(false),
      offset_history_max_size_(10) {  // 历史记录最大长度：100个样本（约5秒，50ms周期）
    
    // 外参原线性增益 0.04 → 已由 ScalarLkf/Vec3Lkf 替代（见 dog_pos_processor_tuning.h）
    yaw_filter_gain_ = kExtrinsicAlpha;
    yaw_stable_threshold_ = M_PI * 5.0 / 180.0;  // 5度，yaw稳定阈值
    yaw_exceed_threshold_ = M_PI * 45.0 / 180.0;  // 45度，yaw超限阈值
    yaw_exceed_max_count_ = 5;  // yaw超限最大计数
    pos_stable_threshold_ = 0.05;  // 5cm，位置稳定阈值
    pos_exceed_threshold_ = 0.3;  // 30cm，位置超限阈值
    pos_exceed_max_count_ = 5;  // 位置超限最大计数
    pos_filter_gain_ = kExtrinsicAlpha;
    aoa_pos_filter_gain_ = 0.05;  // AOA位置滤波增益（比pos_filter_gain小）
    aoa_pos_step_limit_ = 0.02;   // AOA单次迭代上限（米）
    
    // 前馈系数（参考traj_server.cpp）
    camera_offset_ = 0.36;  // 关键参数，与traj_server一致

    // AOA相关参数
    aoa_min_distance_ = 3.0;  // m

    // 仿真模式：camera_offset为0
    simulate_mode_ = true;
    
    // 发布者 - 使用Odometry格式
    dog_pos_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_processed", 10);
    
    // Debug发布者 - 发布AOA计算得到的dog位置
    aoa_dog_pos_debug_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_aoa_debug", 10);
    
    // 订阅者
    raw_dog_pos_sub_ = nh_.subscribe("/dog_pos", 10, &DogPosProcessor::rawDogPosCallback, this);
    target_sub_ = nh_.subscribe("/target_ekf_odom", 10, &DogPosProcessor::targetCallback, this);
    vins_sub_ = nh_.subscribe("/vins_fusion/imu_propagate", 10, &DogPosProcessor::vinsCallback, this);
    // 订阅AOA数据
    aoa_sub_ = nh_.subscribe("/AOA_Tag_data", 10, &DogPosProcessor::aoaCallback, this);

    // 订阅光流高度（用于AOA距离勾股修正）
    flow_sub_ = nh_.subscribe("/flow_data", 10, &DogPosProcessor::flowCallback, this);
    takeoff_sub_ = nh_.subscribe("/px4ctrl/takeoff_land", 10, &DogPosProcessor::takeoffCallback, this);
    yaw_diff_preset_sub_ = nh_.subscribe("/yaw_diff_preset", 10, &DogPosProcessor::yawDiffCallback, this);
    // 测试话题订阅者 - 用于重置initialized状态
    test_reset_sub_ = nh_.subscribe("/test_reset_initialized", 10, &DogPosProcessor::testResetCallback, this);
    
    // Trigger订阅者
    trigger_sub_ = nh_.subscribe("/triger", 10, &DogPosProcessor::triggerCallback, this);
    
    // 定时器
    timer_ = nh_.createTimer(ros::Duration(0.05), &DogPosProcessor::processCallback, this);
    status_timer_ = nh_.createTimer(ros::Duration(0.1), &DogPosProcessor::statusCheckCallback, this);  // 100ms检查状态
    
    if (simulate_mode_) {
        camera_offset_ = 0.0;
    }

    ROS_INFO("dog_pos_processor_sim (Ours): cascade LKF + R/t -> /dog_pos_processed");
    ROS_INFO("  extrinsic Q=%.2e R=%.2e ~alpha=%.3f, output Q=%.3f R=%.3f",
             kExtrinsicQ, kExtrinsicR, kExtrinsicAlpha, kOutputProcessNoise, kOutputMeasNoise);
    ROS_INFO("Waiting for takeoff signal and yaw diff from traj_server...");
    ROS_INFO("Initial yaw offset: 0.0 rad (0.0 deg)");
}

// 角度归一化到[-π, π]
double DogPosProcessor::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}


// 处理原始dog位置数据（完全按照callback.cpp的逻辑）
void DogPosProcessor::rawDogPosCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    raw_dog_pos_ = msg;
    raw_dog_pos_count_++;

    raw_dog_vel_.x() = msg->twist.twist.linear.x;
    raw_dog_vel_.y() = msg->twist.twist.linear.y;
    raw_dog_vel_.z() = msg->twist.twist.linear.z;
    raw_dog_yaw_ = msg->pose.pose.orientation.w;
    dog_vel_initialized_ = true;

    publishProcessedDogPos();
}

// 更新狗通信角速度
// 参数:
//     delta_yaw: 角度差（已经过normalize_angle处理）
void DogPosProcessor::updateDogYawRate(double delta_yaw) {
    double current_time = ros::Time::now().toSec();
    
    if (!dog_yaw_rate_initialized_) {
        // 首次初始化
        last_dog_yaw_time_ = current_time;
        final_dog_yaw_rate_ = 0.0;
        dog_yaw_rate_initialized_ = true;
    } else {
        // 计算时间差
        double dt = current_time - last_dog_yaw_time_;
        
        if (dt > 0.001) {  // 避免除零，至少1ms间隔
            // 计算瞬时角速度（delta_yaw已经处理了角度循环问题）
            double instant_yaw_rate = delta_yaw / dt;
            
            // 简单滤波
            final_dog_yaw_rate_ = (1.0 - yaw_rate_filter_gain_) * final_dog_yaw_rate_ + 
                           yaw_rate_filter_gain_ * instant_yaw_rate;
            
            // 更新上一帧时间
            last_dog_yaw_time_ = current_time;
        }
    }
}

// 更新狗通信加速度
// 参数:
//     delta_vel: 速度变化量 [vx, vy, vz]
//     dt: 时间间隔（秒），如果<=0则自动计算时间差
void DogPosProcessor::updateDogAcc(const Eigen::Vector3d& delta_vel, double dt) {
    double current_time = ros::Time::now().toSec();
    
    if (!dog_acc_initialized_) {
        // 首次初始化
        last_dog_vel_time_ = current_time;
        final_dog_acc_ = Eigen::Vector3d::Zero();
        dog_acc_initialized_ = true;
    } else {
        // 计算时间差：如果提供了有效的dt则使用，否则计算时间差
        double actual_dt;
        if (dt > 0.001) {
            actual_dt = dt;  // 使用提供的dt
        } else {
            actual_dt = current_time - last_dog_vel_time_;  // 计算时间差
        }
        
        if (actual_dt > 0.001) {  // 避免除零，至少1ms间隔
            // 计算瞬时加速度
            Eigen::Vector3d instant_acc = delta_vel / actual_dt;
            
            // 简单滤波（每个分量分别滤波）
            final_dog_acc_.x() = (1.0 - acc_filter_gain_) * final_dog_acc_.x() + 
                          acc_filter_gain_ * instant_acc.x();
            final_dog_acc_.y() = (1.0 - acc_filter_gain_) * final_dog_acc_.y() + 
                          acc_filter_gain_ * instant_acc.y();
            final_dog_acc_.z() = (1.0 - acc_filter_gain_) * final_dog_acc_.z() + 
                          acc_filter_gain_ * instant_acc.z();
            
            // 更新上一帧时间
            last_dog_vel_time_ = current_time;
        }
    }
}

// 处理目标数据（来自read模块的target_ekf_odom）
void DogPosProcessor::targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 记录最新包的时间
    target_ekf_last_time_ = msg->header.stamp;
    
    // 从target_ekf_odom中提取角度（w=yaw, x=pitch, y=roll，单位：弧度）
    target_dog_yaw_ = msg->pose.pose.orientation.w;
    target_dog_pitch_ = msg->pose.pose.orientation.x;
    target_dog_roll_ = msg->pose.pose.orientation.y;
    
    // 从target_ekf_odom中提取位置
    target_dog_pos_.x() = msg->pose.pose.position.x;
    target_dog_pos_.y() = msg->pose.pose.position.y;
    target_dog_pos_.z() = msg->pose.pose.position.z;
    
    target_count_++;

}

// 处理VINS数据
void DogPosProcessor::vinsCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 从VINS消息中提取yaw
    double q_w = msg->pose.pose.orientation.w;
    double q_x = msg->pose.pose.orientation.x;
    double q_y = msg->pose.pose.orientation.y;
    double q_z = msg->pose.pose.orientation.z;
    // 计算偏航角
    double siny_cosp = 2.0 * (q_w * q_z + q_x * q_y);
    double cosy_cosp = 1.0 - 2.0 * (q_y * q_y + q_z * q_z);
    vins_yaw_ = std::atan2(siny_cosp, cosy_cosp);
    
    // 提取VINS姿态
    R_wb_ = Eigen::Quaterniond(q_w, q_x, q_y, q_z).toRotationMatrix();

    // 提取VINS位置
    vins_pos_.x() = msg->pose.pose.position.x;
    vins_pos_.y() = msg->pose.pose.position.y;
    vins_pos_.z() = msg->pose.pose.position.z;
    
    // 增加计数器
    vins_count_++;
}

// 处理起飞信号
void DogPosProcessor::takeoffCallback(const quadrotor_msgs::TakeoffLand::ConstPtr& msg) {
    if (msg->takeoff_land_cmd == 1) {  // 起飞命令
        ROS_INFO("Takeoff detected, yaw offset will be reinitialized");
        initialized_ = false;
        precise_yaw_offset_ready_ = false;
        precise_pos_offset_ready_ = false;
        trigger_received_ = false;
        resetExtrinsicLkf(0.0, Eigen::Vector3d::Zero());
        kf_.reset();
    }
}

// 处理yaw差值信息
void DogPosProcessor::yawDiffCallback(const std_msgs::Float64::ConstPtr& msg) {
    // 每次降落的时候，发布并保存yaw差值，用于下次起飞时初始化
    if (saved_yaw_diff_ == nullptr) {
        saved_yaw_diff_ = new double(msg->data);
    } else {
        *saved_yaw_diff_ = msg->data;
    }
    ROS_INFO("Received yaw diff: %.1f deg", msg->data * 180.0 / M_PI);
}

// 测试重置回调 - 接收到任意信息时重置initialized状态
void DogPosProcessor::testResetCallback(const std_msgs::Bool::ConstPtr& msg) {
    ROS_INFO("Test reset signal received, resetting initialized to False");
    initialized_ = false;
    precise_yaw_offset_ready_ = false;
    precise_pos_offset_ready_ = false;
    resetExtrinsicLkf(0.0, Eigen::Vector3d::Zero());
    kf_.reset();
}

// Trigger回调（使用PoseStamped，占位内容无需读取）
void DogPosProcessor::triggerCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    trigger_received_ = true;
    ROS_INFO("Trigger received - starting iteration");
}

// 状态检查回调（100ms频率）
void DogPosProcessor::statusCheckCallback(const ros::TimerEvent& event) {
    // 检查dog_pos_received状态（完全按照traj_server.cpp）
    if (raw_dog_pos_count_ != last_raw_dog_pos_count_) {
        raw_dog_pos_received_ = true;
        last_raw_dog_pos_count_ = raw_dog_pos_count_;
        last_dog_pos_timer_ = 0;
    } else {
        last_dog_pos_timer_++;
        if (last_dog_pos_timer_ >= 1) {  // 连续5次没有新包才重置
            raw_dog_pos_received_ = false;
            // 重置速度相关变量
            dog_yaw_rate_initialized_ = false;
            dog_vel_initialized_ = false;
            dog_acc_initialized_ = false;
        }
    }
    
    // 检查target_receive状态（模仿dog_pos_received的逻辑）
    if (target_count_ != last_target_count_) {
        last_target_timer_ ++;
        if (last_target_timer_ >= 1) {
            target_receive_ = true;
        }
        last_target_count_ = target_count_;
        last_target_loss_timer_ = 0;
    } else {
        last_target_loss_timer_++;
        if (last_target_loss_timer_ >= 1) {  // 连续5次没有新包才重置
            target_receive_ = false;
        }
        last_target_timer_ = 0;
    }
    
    // 检查vins_received状态（模仿dog_pos_received的逻辑）
    if (vins_count_ != last_vins_count_) {
        vins_received_ = true;
        last_vins_count_ = vins_count_;
        last_vins_timer_ = 0;
    } else {
        last_vins_timer_++;
        if (last_vins_timer_ >= 5) {  // 连续5次没有新包才重置
            vins_received_ = false;
        }
    }

    // 检查aoa_received状态（模仿dog_pos_received的逻辑）
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

// 主处理回调（完全按照traj_server.cpp的逻辑）
void DogPosProcessor::processCallback(const ros::TimerEvent& event) {
    const auto iter_t0 = std::chrono::steady_clock::now();
    double timer_dt = (event.current_real - event.last_real).toSec();
    if (timer_dt <= 0.0) timer_dt = kTimerDt;

    if (!initialized_ && raw_dog_pos_received_ && vins_received_) {
        double diff = 0.0;
        if (saved_yaw_diff_ != nullptr) {
            diff = *saved_yaw_diff_;
            delete saved_yaw_diff_;
            saved_yaw_diff_ = nullptr;
        }

        yaw_offset_ = normalizeAngle(raw_dog_yaw_ - (vins_yaw_ + diff));

        const double cos_yaw = std::cos(yaw_offset_);
        const double sin_yaw = std::sin(yaw_offset_);
        const Eigen::Vector3d rotated_vins_pos(
            cos_yaw * vins_pos_.x() - sin_yaw * vins_pos_.y(),
            sin_yaw * vins_pos_.x() + cos_yaw * vins_pos_.y(),
            vins_pos_.z());
        const Eigen::Vector3d tmp_p(
            raw_dog_pos_->pose.pose.position.x,
            raw_dog_pos_->pose.pose.position.y,
            raw_dog_pos_->pose.pose.position.z);
        pos_offset_ = tmp_p - rotated_vins_pos;

        resetExtrinsicLkf(yaw_offset_, pos_offset_);

        ROS_INFO("Reinitialized offsets: yaw_offset=%.1f deg, pos_offset=[%.3f, %.3f, %.3f]",
                 yaw_offset_ * 180.0 / M_PI, pos_offset_.x(), pos_offset_.y(), pos_offset_.z());

        yaw_offset_history_.clear();
        pos_offset_history_.clear();

        initialized_ = true;
        precise_pos_offset_ready_ = true;
        precise_yaw_offset_ready_ = true;
    }

    if (target_receive_ && raw_dog_pos_received_ && vins_received_) {
        const double current_yaw_offset = normalizeAngle(raw_dog_yaw_ - target_dog_yaw_);
        const double yaw_before = yaw_offset_;
        s_yaw_offset_lkf.update(current_yaw_offset, timer_dt, true);
        yaw_offset_ = s_yaw_offset_lkf.state();
        const double yaw_offset_diff = normalizeAngle(current_yaw_offset - yaw_before);

        yaw_offset_history_.push_back(yaw_offset_);
        if (yaw_offset_history_.size() > offset_history_max_size_) {
            yaw_offset_history_.pop_front();
        }

        if (std::abs(yaw_offset_diff) < yaw_stable_threshold_) {
            if (!precise_yaw_offset_ready_) {
                ROS_INFO("precise_yaw_offset_ready!");
                if (precise_pos_offset_ready_) {
                    initialized_ = true;
                }
            }
            precise_yaw_offset_ready_ = true;
        }

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

        if (precise_yaw_offset_ready_) {
            const Eigen::Vector3d raw_dog_pos(
                raw_dog_pos_->pose.pose.position.x,
                raw_dog_pos_->pose.pose.position.y,
                raw_dog_pos_->pose.pose.position.z);

            const Eigen::Vector3d target_dog_pos_with_ff(
                target_dog_pos_.x() + camera_offset_ * final_dog_vel_.x()
                    + 0.5 * camera_offset_ * camera_offset_ * final_dog_acc_.x(),
                target_dog_pos_.y() + camera_offset_ * final_dog_vel_.y()
                    + 0.5 * camera_offset_ * camera_offset_ * final_dog_acc_.y(),
                target_dog_pos_.z());

            const double cos_yaw = std::cos(yaw_offset_);
            const double sin_yaw = std::sin(yaw_offset_);
            const Eigen::Vector3d rotated_target(
                cos_yaw * target_dog_pos_with_ff.x() - sin_yaw * target_dog_pos_with_ff.y(),
                sin_yaw * target_dog_pos_with_ff.x() + cos_yaw * target_dog_pos_with_ff.y(),
                target_dog_pos_with_ff.z());

            const Eigen::Vector3d current_pos_offset = raw_dog_pos - rotated_target;
            const Eigen::Vector3d pos_before = pos_offset_;
            s_pos_offset_lkf.update(current_pos_offset, timer_dt);
            pos_offset_ = s_pos_offset_lkf.state();

            pos_offset_history_.push_back(pos_offset_);
            if (pos_offset_history_.size() > offset_history_max_size_) {
                pos_offset_history_.pop_front();
            }

            const double pos_offset_diff_norm = (current_pos_offset - pos_before).norm();

            if (pos_offset_diff_norm < pos_stable_threshold_) {
                if (!precise_pos_offset_ready_) {
                    ROS_INFO("precise_pos_offset_ready!");
                    if (precise_yaw_offset_ready_) {
                        initialized_ = true;
                    }
                    ROS_INFO("Position converged, resetting output Kalman filter");
                    kf_.reset();
                    last_kf_time_ = ros::Time();
                    yaw_filter_initialized_ = false;
                    filtered_yaw_ = 0.0;
                }
                precise_pos_offset_ready_ = true;
            }

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

    // 维护AOA位置偏移（单距离单角度方案）
    // 前提：yaw ready、vins可用、aoa可用，且飞机朝向与当前处理后狗位姿夹角在±30度内
    if (precise_yaw_offset_ready_ && vins_received_ && aoa_received_) {
        // 角度限制：飞机朝向需面向当前processed dog pos的±30度
        Eigen::Vector3d dog_vec = final_dog_pos_ - vins_pos_;
        double heading_to_dog = std::atan2(dog_vec.y(), dog_vec.x());
        double heading_diff = normalizeAngle(heading_to_dog - vins_yaw_);
        if (std::abs(heading_diff) > M_PI / 4.0) {  // 45度
            return;
        }
    
        double height_diff = final_dog_pos_.z() - vins_pos_.z();
        if (aoa_distance_ < std::abs(height_diff)) {
            return;
        }
        double aoa_distance_horizontal = std::sqrt(aoa_distance_ * aoa_distance_ - height_diff * height_diff);

        if (aoa_distance_horizontal < aoa_min_distance_) {
            return;
        }

        // ========================== 坐标与几何问题定义 ==========================
        //
        // 目标：
        //   已知：
        //     - 飞机在世界系中的姿态 R_wb_
        //     - 飞机为原点
        //     - 目标相对飞机的高度差 height_diff = Δz
        //     - 目标相对飞机的水平距离 aoa_distance_horizontal
        //     - 目标在机体系下的 yaw 角 aoa_angle_
        //
        //   求：
        //     - 目标在世界系下的位置 aoa_pos_world
        //
        // 约束条件：
        //   1) 目标位于“yaw 平面”内（由 yaw 角确定）
        //   2) 目标的 z 坐标等于 Δz
        //   3) 目标的水平距离等于 aoa_distance_horizontal
        //
        // 几何思路：
        //   - yaw 平面 ∩ 水平面 (z = Δz) 是一条直线
        //   - 先在这条直线上找到“距离原点最近的点 p0”
        //   - 再沿该直线方向走到满足水平距离约束的位置
        // ======================================================================


        // ========================== Step 0：基础向量 ==========================

        // 世界系 z 轴单位向量
        Eigen::Vector3d ez(0.0, 0.0, 1.0);

        // 目标在机体系 yaw 平面中的 bearing 方向（x-y 平面内）
        Eigen::Vector3d bearing_b(
            std::cos(aoa_angle_),
            std::sin(aoa_angle_),
            0.0
        );


        // ========================== Step 1：构造 yaw 平面的法向 ==========================
        //
        // yaw 平面定义：
        //   - 经过飞机原点
        //   - 包含 bearing_b 方向
        //   - 包含机体系 z 轴
        //
        // 因此其法向（机体系）为：
        Eigen::Vector3d n_b = bearing_b.cross(ez);

        // 转到世界系
        Eigen::Vector3d n_w = R_wb_ * n_b;

        // 归一化，便于后续解释和数值稳定
        n_w.normalize();


        // ========================== Step 2：yaw 平面与水平面的交线 ==========================
        //
        // yaw 平面与水平面 (z = const) 的交集是一条直线
        // 其方向等于两个平面法向的叉乘：
        //   d ∥ n × ez
        //
        Eigen::Vector3d d_w = - n_w.cross(ez);

        // 退化情况：yaw 平面与水平面平行（无唯一解）
        if (d_w.norm() < 1e-6)
        {
            // 此时仅凭水平距离无法唯一确定目标
            return;
        }

        // 该交线的单位方向（用于后续参数化）
        Eigen::Vector3d d_hat = d_w.normalized();


        // ========================== Step 3：构造“最近点” p0 ==========================
        //
        // 目标：
        //   在 yaw 平面 ∩ z = Δz 的直线上，
        //   找到距离原点最近的点 p0。
        //
        // 思路：
        //   - 从 p = Δz * ez 出发（只满足高度约束）
        //   - 只能在“水平面内”修正，否则会破坏 z = Δz
        //   - 修正方向必须取 yaw 平面法向 n 在水平面的投影
        //

        // yaw 平面法向在竖直方向的分量
        double n_z = n_w.dot(ez);

        // yaw 平面法向在水平面的分量（正确的“拉回方向”）
        Eigen::Vector3d n_h = n_w - n_z * ez;

        // |n_h|^2
        double n_h_sq = n_h.squaredNorm();

        // 理论上此处不应退化（与 d_w 检查等价），保险起见
        if (n_h_sq < 1e-8)
        {
            return;
        }


        // ========================== Step 4：一步写出 p0（闭式解） ==========================
        //
        // 构造未知点：
        //   p0 = Δz * ez - α * n_h
        //
        // 其中 α 表示沿 n_h 方向修正的量。
        // 将 p0 代入 yaw 平面方程：
        //   n · p0 = 0
        //
        // 得到一元一次方程：
        //   Δz * n_z - α * |n_h|^2 = 0
        //
        // 解得：
        //   α = (n_z * Δz) / |n_h|^2
        //
        Eigen::Vector3d p0 =
            height_diff * ez
            - (n_z * height_diff / n_h_sq) * n_h;


        // ========================== Step 5：在 yaw 平面内施加水平距离约束 ==========================
        //
        // yaw 平面 ∩ z = Δz 是一条直线：
        //   L(t) = p0 + t * d_hat
        //
        // 其中：
        //   - p0     是该直线中距离原点最近的点
        //   - d_hat  是直线的单位方向（完全在水平面内）
        //
        // 目标点必须满足：
        //   || (p0 + t * d_hat)_h || = aoa_distance_horizontal
        //

        // p0 的水平分量
        Eigen::Vector2d p0_h(p0.x(), p0.y());

        // 最近点到原点的水平距离
        double p0_h_norm = p0_h.norm();

        // 解二次方程前的判定
        double inside =
            aoa_distance_horizontal * aoa_distance_horizontal
            - p0_h_norm * p0_h_norm;

        if (inside < 0.0)
        {
            // 水平圆与 yaw 平面直线无交点
            return;
        }

        // 沿 bearing 正方向的唯一物理解
        double t = std::sqrt(inside);

        // 最终目标相对飞机的位置（世界系）
        Eigen::Vector3d target_rel = p0 + t * d_hat;


        // ========================== Step 6：转为世界系绝对坐标 ==========================
        //
        // 前面所有计算均以飞机为原点
        // 最终只在这里加上飞机世界系位置
        //
        Eigen::Vector3d aoa_pos_world = vins_pos_ + target_rel;


        // 将AOA估计位置旋转到狗坐标系（使用 yaw_offset_）
        double cos_yaw = std::cos(yaw_offset_);
        double sin_yaw = std::sin(yaw_offset_);
        double rotated_aoa_x = cos_yaw * (aoa_pos_world.x() - vins_pos_.x()) - sin_yaw * (aoa_pos_world.y() - vins_pos_.y());
        double rotated_aoa_y = sin_yaw * (aoa_pos_world.x() - vins_pos_.x()) + cos_yaw * (aoa_pos_world.y() - vins_pos_.y());
        Eigen::Vector3d rotated_aoa_pos(rotated_aoa_x, rotated_aoa_y, final_dog_pos_.z());

        // 与raw_dog_pos比较，更新pos_offset_
        Eigen::Vector3d raw_dog_pos(
            raw_dog_pos_->pose.pose.position.x,
            raw_dog_pos_->pose.pose.position.y,
            raw_dog_pos_->pose.pose.position.z
        );
        Eigen::Vector3d current_pos_offset = raw_dog_pos - rotated_aoa_pos;

        // Debug发布AOA估计位置
        nav_msgs::Odometry aoa_debug_msg;
        aoa_debug_msg.header.stamp = ros::Time::now();
        aoa_debug_msg.header.frame_id = "world";
        aoa_debug_msg.pose.pose.position.x = aoa_pos_world.x();
        aoa_debug_msg.pose.pose.position.y = aoa_pos_world.y();
        aoa_debug_msg.pose.pose.position.z = aoa_pos_world.z();
        aoa_dog_pos_debug_pub_.publish(aoa_debug_msg);

        // 位置偏移滤波：x和y分开迭代，z不迭代
        Eigen::Vector3d pos_offset_diff = current_pos_offset - pos_offset_;
        
        // x方向迭代
        double diff_x = pos_offset_diff.x();
        if (std::abs(diff_x) > 1e-6) {
            double step_x = aoa_pos_filter_gain_ * std::abs(diff_x);
            step_x = std::min(step_x, aoa_pos_step_limit_);
            pos_offset_.x() += (diff_x > 0 ? step_x : -step_x);
        }
        
        // y方向迭代
        double diff_y = pos_offset_diff.y();
        if (std::abs(diff_y) > 1e-6) {
            double step_y = aoa_pos_filter_gain_ * std::abs(diff_y);
            step_y = std::min(step_y, aoa_pos_step_limit_);
            pos_offset_.y() += (diff_y > 0 ? step_y : -step_y);
        }
        
        // z方向不迭代，保持不变
    }

    logIterMsThrottled("dog_pos_processor [timer]", iter_t0);
}

// 发布处理后的dog_pos（使用Odometry格式）
void DogPosProcessor::publishProcessedDogPos() {
    if (raw_dog_pos_ == nullptr) return;

    const auto iter_t0 = std::chrono::steady_clock::now();
    
    nav_msgs::Odometry msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "world";
    
    // 位置处理：先减掉pos_offset，再旋转yaw_offset
    Eigen::Vector3d raw_pos(
        raw_dog_pos_->pose.pose.position.x,
        raw_dog_pos_->pose.pose.position.y,
        raw_dog_pos_->pose.pose.position.z
    );
    
    // 先减掉pos_offset（狗坐标系下的偏移）
    Eigen::Vector3d corrected_pos = raw_pos - pos_offset_;
    
    // 再旋转yaw_offset（将狗坐标系转换到世界坐标系）
    double yaw_diff = normalizeAngle(-yaw_offset_);
    double cos_yaw = std::cos(yaw_diff);
    double sin_yaw = std::sin(yaw_diff);
    double rotated_x = cos_yaw * corrected_pos.x() - sin_yaw * corrected_pos.y();
    double rotated_y = sin_yaw * corrected_pos.x() + cos_yaw * corrected_pos.y();
    
    // 速度旋转
    double target_yaw = normalizeAngle(raw_dog_yaw_ - yaw_offset_);
    cos_yaw = std::cos(target_yaw);
    sin_yaw = std::sin(target_yaw);
    double rotated_vx = cos_yaw * raw_dog_vel_.x() - sin_yaw * raw_dog_vel_.y();
    double rotated_vy = sin_yaw * raw_dog_vel_.x() + cos_yaw * raw_dog_vel_.y();

    // 先计算并存储所有最终输出的dog相关量到self中
    final_dog_pos_.x() = rotated_x;
    final_dog_pos_.y() = rotated_y;
    final_dog_pos_.z() = corrected_pos.z();

    final_dog_vel_.x() = rotated_vx;
    final_dog_vel_.y() = rotated_vy;
    final_dog_vel_.z() = raw_dog_vel_.z();
    final_dog_yaw_ = target_yaw;

    // 如果target_receive为true，直接用target_dog_pos替换位置（在KF之前）
    // 加上camera_offset前馈偏移量（基于final_dog_vel，在世界坐标系下）
    // if (target_receive_) {
    //     final_dog_pos_.x() = target_dog_pos_.x() + camera_offset_ * final_dog_vel_.x() + 0.5 * camera_offset_ * camera_offset_ * final_dog_acc_.x();
    //     final_dog_pos_.y() = target_dog_pos_.y() + camera_offset_ * final_dog_vel_.y() + 0.5 * camera_offset_ * camera_offset_ * final_dog_acc_.y();
    //     final_dog_pos_.z() = target_dog_pos_.z();
    //     final_dog_yaw_ = target_dog_yaw_;
    // }

    // 卡尔曼滤波：只有pos收敛了才使用滤波器
    if (kf_enabled_ && precise_pos_offset_ready_) {
        double current_time = ros::Time::now().toSec();
        
        if (last_kf_time_.isZero()) {
            last_kf_time_ = ros::Time::now();
            double dt = 0.05;  // 默认50ms
        } else {
            double dt = (ros::Time::now() - last_kf_time_).toSec();
            if (dt <= 0) {
                dt = 0.05;  // 防止非正时间间隔
            }
            
            // 如果dt超过超时阈值，重置滤波器
            if (dt > kf_timeout_) {
                ROS_WARN("Kalman filter timeout (dt=%.2fs > %.2fs), resetting filter", dt, kf_timeout_);
                kf_.reset();
                yaw_filter_initialized_ = false;
                filtered_yaw_ = 0.0;
                dt = 0.05;  // 重置后使用默认时间间隔
            }
            
            // 对位置和速度进行卡尔曼滤波
            auto filtered = kf_.filter(final_dog_pos_, final_dog_vel_, dt);
            final_dog_pos_ = filtered.first;
            final_dog_vel_ = filtered.second;
            
            // 对 yaw 进行简单一阶滤波（处理角度环绕）
            if (!yaw_filter_initialized_) {
                filtered_yaw_ = final_dog_yaw_;
                yaw_filter_initialized_ = true;
            } else {
                double yaw_diff = normalizeAngle(final_dog_yaw_ - filtered_yaw_);
                filtered_yaw_ = normalizeAngle(
                    filtered_yaw_ + yaw_filter_gain_kf_ * yaw_diff
                );
            }
            
            // 使用滤波后的值
            final_dog_pos_ = filtered.first;
            final_dog_vel_ = filtered.second;
            final_dog_yaw_ = filtered_yaw_;
            
            last_kf_time_ = ros::Time::now();
        }
    }
    
    // 计算三个轴的角速度和加速度（在publishProcessedDogPos中统一计算，使用update函数）
    
    // 1. 计算yaw角速度
    double delta_yaw = normalizeAngle(final_dog_yaw_ - last_final_dog_yaw_);
    updateDogYawRate(delta_yaw);
    
    // 2. 计算加速度（使用final_dog_vel_）
    Eigen::Vector3d delta_vel = final_dog_vel_ - last_final_dog_vel_;
    updateDogAcc(delta_vel, default_dt);
    
    // 更新上一次的值
    last_final_dog_yaw_ = final_dog_yaw_;
    last_final_dog_pitch_ = target_dog_pitch_;
    last_final_dog_roll_ = target_dog_roll_;
    last_final_dog_vel_ = final_dog_vel_;
    
    // 使用self的变量填充msg
    msg.pose.pose.position.x = final_dog_pos_.x();
    msg.pose.pose.position.y = final_dog_pos_.y();
    msg.pose.pose.position.z = final_dog_pos_.z();

    // 标准差和加速度：
    // w = pos_offset标准差（协方差矩阵迹的平方根）
    // x = yaw_offset标准差
    // y = 加速度x（世界坐标系）
    // z = 加速度y（世界坐标系）

    // msg.pose.pose.orientation.w = precise_pos_offset_ready_ ? 1.0 : 0.0;
    // msg.pose.pose.orientation.x = precise_yaw_offset_ready_ ? 1.0 : 0.0;

    msg.pose.pose.orientation.w = 1.0;  // pos_ready（降落实验恒 true）
    msg.pose.pose.orientation.x = 1.0;  // yaw_ready（降落实验恒 true）
    msg.pose.pose.orientation.y = final_dog_acc_.x();  // 发布x方向加速度（世界坐标系）
    msg.pose.pose.orientation.z = final_dog_acc_.y();  // 发布y方向加速度（世界坐标系）

    // 速度
    msg.twist.twist.linear.x = final_dog_vel_.x();
    msg.twist.twist.linear.y = final_dog_vel_.y();
    msg.twist.twist.linear.z = final_dog_vel_.z();
    
    // 最终yaw放在twist.angular.x,另外，y放上狗通信角速度
    msg.twist.twist.angular.x = final_dog_yaw_;
    msg.twist.twist.angular.y = final_dog_yaw_rate_;  // 发布狗通信角速度
    
    // 保留twist.angular.z字段，目前不再用于感知置信度，固定为0
    msg.twist.twist.angular.z = 0.0;

    dog_pos_pub_.publish(msg);
    logIterMsThrottled("dog_pos_processor [raw]", iter_t0);
}

void DogPosProcessor::aoaCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 单距离 + 单角度：position.x = 距离，orientation.x = 相对于无人机朝向的角度
    double distance_raw = msg->pose.pose.position.x;
    double angle_raw = msg->pose.pose.orientation.x;

    aoa_distance_ = distance_raw;
    aoa_angle_ = angle_raw;

    aoa_count_++;
}

void DogPosProcessor::flowCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 光流高度在 position.z（参考 withdraw 的 /flow_data 使用 Odometry）
    flow_z_ = msg->pose.pose.position.z;
}

void DogPosProcessor::spin() {
    ros::spin();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "dog_pos_processor_sim");
    DogPosProcessor processor;
    processor.spin();
    return 0;
}

