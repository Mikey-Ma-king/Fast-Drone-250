#include "dog_pos_processor.h"
#include <ros/ros.h>
#include <Eigen/Dense>
#include <cmath>

/*
 * Dog Position Processor (LKF Base)
 * 使用 target_ekf_odom（直接）与 raw_dog_pos（与上一次的增量 + 当前最终状态）
 * 共同输入一个 LKF，持续维护并输出 dog_pos_processed。
 */
// target 速度、raw 上一帧（用于增量），仅本文件使用
static Eigen::Vector3d s_target_dog_vel = Eigen::Vector3d::Zero();
static Eigen::Vector3d s_last_raw_dog_pos = Eigen::Vector3d::Zero();
static Eigen::Vector3d s_last_raw_dog_vel = Eigen::Vector3d::Zero();
static bool s_last_raw_initialized = false;
// 首次收到 target_ekf_odom 的时间，5s 后视为收敛（仅本 cpp 使用，不改 h）
static ros::Time s_first_target_ekf_time;

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

// 预测步骤
void KalmanFilter::predict(double dt) {
    if (!initialized_) return;
    
    // 构建状态转移矩阵（恒定速度模型）
    Eigen::MatrixXd F = F_base_;
    F(0, 3) = dt;  // x = x + vx * dt
    F(1, 4) = dt;  // y = y + vy * dt
    F(2, 5) = dt;  // z = z + vz * dt
    
    // 预测状态
    state_ = F * state_;
    
    // 预测协方差
    covariance_ = F * covariance_ * F.transpose() + Q_;
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
      kf_(0.01, 0.1),
      kf_enabled_(true),
      yaw_filter_gain_kf_(0.3),
      filtered_yaw_(0.0),
      yaw_filter_initialized_(false),
      kf_timeout_(1.0),
      trigger_received_(false),
      offset_history_max_size_(10) {  // 历史记录最大长度：100个样本（约5秒，50ms周期）
    
    // 参数（完全按照traj_server.cpp的设置）
    yaw_filter_gain_ = 0.1;  // yaw offset滤波增益
    yaw_stable_threshold_ = M_PI * 5.0 / 180.0;  // 5度，yaw稳定阈值
    yaw_exceed_threshold_ = M_PI * 45.0 / 180.0;  // 45度，yaw超限阈值
    yaw_exceed_max_count_ = 5;  // yaw超限最大计数
    pos_stable_threshold_ = 0.05;  // 5cm，位置稳定阈值
    pos_exceed_threshold_ = 0.3;  // 30cm，位置超限阈值
    pos_exceed_max_count_ = 5;  // 位置超限最大计数
    pos_filter_gain_ = 0.2;  // 位置滤波增益
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

    // 一开始显式 reset KF，保证从未初始化状态开始
    kf_.reset();

    ROS_INFO("Dog Position Processor (LKF base): target_ekf_odom + raw_dog_pos(increment+state) -> one LKF -> dog_pos_processed");
}

// 角度归一化到[-π, π]
double DogPosProcessor::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}


// 处理原始 dog 位置：不坐标转换直接用；用与上一次的增量 + 当前最终状态 作为观测，经 LKF 更新
void DogPosProcessor::rawDogPosCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    raw_dog_pos_ = msg;
    raw_dog_pos_count_++;

    Eigen::Vector3d current_raw_pos(
        msg->pose.pose.position.x,
        msg->pose.pose.position.y,
        msg->pose.pose.position.z
    );
    Eigen::Vector3d current_raw_vel(
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z
    );
    raw_dog_yaw_ = msg->pose.pose.orientation.w;

    Eigen::Vector3d meas_pos, meas_vel;
    if (s_last_raw_initialized) {
        Eigen::Vector3d delta_pos = current_raw_pos - s_last_raw_dog_pos;
        Eigen::Vector3d delta_vel = current_raw_vel - s_last_raw_dog_vel;
        meas_pos = final_dog_pos_ + delta_pos;
        meas_vel = final_dog_vel_ + delta_vel;
    } else {
        meas_pos = current_raw_pos;
        meas_vel = current_raw_vel;
        s_last_raw_initialized = true;
    }
    s_last_raw_dog_pos = current_raw_pos;
    s_last_raw_dog_vel = current_raw_vel;

    double dt = 0.05;
    if (!last_kf_time_.isZero()) {
        dt = (ros::Time::now() - last_kf_time_).toSec();
        if (dt <= 0) dt = 0.05;
        if (dt > kf_timeout_) {
            kf_.reset();
            dt = 0.05;
        }
    }
    last_kf_time_ = ros::Time::now();

    if (kf_enabled_) {
        auto res = kf_.filter(meas_pos, meas_vel, dt);
        final_dog_pos_ = res.first;
        final_dog_vel_ = res.second;
    } else {
        final_dog_pos_ = meas_pos;
        final_dog_vel_ = meas_vel;
    }
    final_dog_yaw_ = raw_dog_yaw_;

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

// 处理 target_ekf_odom：不坐标转换直接用，作为观测经 LKF 更新最终状态
void DogPosProcessor::targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    target_ekf_last_time_ = msg->header.stamp;
    // 首次收到 target_ekf_odom 时记录时间，用于 5s 后判定收敛
    if (s_first_target_ekf_time.isZero()) {
        s_first_target_ekf_time = ros::Time::now();
    }

    target_dog_yaw_ = msg->pose.pose.orientation.w;
    target_dog_pitch_ = msg->pose.pose.orientation.x;
    target_dog_roll_ = msg->pose.pose.orientation.y;

    target_dog_pos_.x() = msg->pose.pose.position.x;
    target_dog_pos_.y() = msg->pose.pose.position.y;
    target_dog_pos_.z() = msg->pose.pose.position.z;

    s_target_dog_vel.x() = msg->twist.twist.linear.x;
    s_target_dog_vel.y() = msg->twist.twist.linear.y;
    s_target_dog_vel.z() = msg->twist.twist.linear.z;

    target_count_++;

    double dt = 0.05;
    if (!last_kf_time_.isZero()) {
        dt = (ros::Time::now() - last_kf_time_).toSec();
        if (dt <= 0) dt = 0.05;
        if (dt > kf_timeout_) {
            kf_.reset();
            dt = 0.05;
        }
    }
    last_kf_time_ = ros::Time::now();

    if (kf_enabled_) {
        auto res = kf_.filter(target_dog_pos_, s_target_dog_vel, dt);
        final_dog_pos_ = res.first;
        final_dog_vel_ = res.second;
    } else {
        final_dog_pos_ = target_dog_pos_;
        final_dog_vel_ = s_target_dog_vel;
    }
    final_dog_yaw_ = target_dog_yaw_;

    publishProcessedDogPos();
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
        initialized_ = false;  // 重置初始化状态
        precise_yaw_offset_ready_ = false;
        precise_pos_offset_ready_ = false;
        trigger_received_ = false;
        kf_.reset();
        last_kf_time_ = ros::Time();
        s_first_target_ekf_time = ros::Time(0, 0);  // 起飞后重新计时 5s 收敛
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
    kf_.reset();
    last_kf_time_ = ros::Time();
    s_first_target_ekf_time = ros::Time(0, 0);  // 重置后重新计时 5s 收敛
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

// 主处理回调：50Hz 定时发布当前 LKF 维护的 dog_pos_processed（无新测量时也持续输出）
void DogPosProcessor::processCallback(const ros::TimerEvent& event) {
    (void)event;
    publishProcessedDogPos();
}

// 发布 LKF 维护的 dog_pos_processed（仅用 final 状态，无坐标转换）
void DogPosProcessor::publishProcessedDogPos() {
    nav_msgs::Odometry msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "world";

    msg.pose.pose.position.x = final_dog_pos_.x();
    msg.pose.pose.position.y = final_dog_pos_.y();
    msg.pose.pose.position.z = final_dog_pos_.z();

    // 有 target_ekf_odom 输入且已过 5s 则视为收敛，用 orientation.w/x 标记 pos_ready / yaw_ready
    double elapsed = s_first_target_ekf_time.isZero() ? 0.0 : (ros::Time::now() - s_first_target_ekf_time).toSec();
    bool converged = (elapsed >= 1.0);
    msg.pose.pose.orientation.w = converged ? 1.0 : 0.0;   // pos_ready
    msg.pose.pose.orientation.x = converged ? 1.0 : 0.0;   // yaw_ready
    msg.pose.pose.orientation.y = 0.0;
    msg.pose.pose.orientation.z = 0.0;

    msg.twist.twist.linear.x = final_dog_vel_.x();
    msg.twist.twist.linear.y = final_dog_vel_.y();
    msg.twist.twist.linear.z = final_dog_vel_.z();

    msg.twist.twist.angular.x = final_dog_yaw_;
    msg.twist.twist.angular.y = 0.0;
    msg.twist.twist.angular.z = 0.0;

    dog_pos_pub_.publish(msg);
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
    ros::init(argc, argv, "dog_pos_processor");
    DogPosProcessor processor;
    processor.spin();
    return 0;
}

