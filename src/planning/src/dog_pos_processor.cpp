#include "dog_pos_processor.h"
#include <ros/ros.h>
#include <Eigen/Dense>
#include <cmath>

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
    state_ = Eigen::VectorXd::Zero(6);
    covariance_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;
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
      raw_dog_pos_received_(false),
      raw_dog_pos_count_(0),
      last_raw_dog_pos_count_(0),
      last_dog_pos_timer_(0),
      vins_yaw_(0.0),
      vins_pos_(Eigen::Vector3d::Zero()),
      vins_received_(false),
      vins_count_(0),
      last_vins_count_(0),
      last_vins_timer_(0),
      target_dog_yaw_(0.0),
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
      dog_yaw_rate_(0.0),
      last_dog_yaw_time_(0.0),
      yaw_rate_filter_gain_(0.3),
      dog_yaw_rate_initialized_(false),
      kf_(0.01, 0.1),
      kf_enabled_(true),
      yaw_filter_gain_kf_(0.3),
      filtered_yaw_(0.0),
      yaw_filter_initialized_(false),
      kf_timeout_(1.0),
      trigger_received_(false) {
    
    // 参数（完全按照traj_server.cpp的设置）
    yaw_filter_gain_ = 0.1;  // yaw offset滤波增益
    yaw_stable_threshold_ = M_PI * 5.0 / 180.0;  // 5度，yaw稳定阈值
    yaw_exceed_threshold_ = M_PI * 45.0 / 180.0;  // 45度，yaw超限阈值
    yaw_exceed_max_count_ = 5;  // yaw超限最大计数
    pos_stable_threshold_ = 0.05;  // 5cm，位置稳定阈值
    pos_exceed_threshold_ = 0.3;  // 30cm，位置超限阈值
    pos_exceed_max_count_ = 5;  // 位置超限最大计数
    pos_filter_gain_ = 0.05;  // 位置滤波增益
    aoa_pos_filter_gain_ = 0.05;  // AOA位置滤波增益（比pos_filter_gain小）
    aoa_pos_step_limit_ = 0.01;   // AOA单次迭代上限（米）
    
    // 前馈系数（参考traj_server.cpp）
    camera_offset_ = 0.37;  // 关键参数，与traj_server一致

    // AOA相关参数
    aoa_min_distance_ = 2.0;  // m
    aoa_min_distance_diff_ = 0.1;  // 两个anchor距离差的最小值，小于此值则认为距离差太小，不更新
    aoa_anchor_separation_ = 0.5;  // 两个anchor之间的距离（50cm）
    flow_height_bias_ = 0.47;  // 与withdraw一致的零偏

    // 仿真模式：直接输出target_ekf为dog_pos_processed
    simulate_mode_ = false;
    
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
    
    ROS_INFO("Dog Position Processor initialized");
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
    
    // 提取位置、速度、yaw
    Eigen::Vector3d current_raw_dog_vel(
        msg->twist.twist.linear.x,
        msg->twist.twist.linear.y,
        msg->twist.twist.linear.z
    );
    double current_raw_dog_yaw = msg->pose.pose.orientation.w;
    
    // 速度限制（完全按照callback.cpp）
    const double min_dog_velocity = -2.0;
    const double max_dog_velocity = 2.0;
    current_raw_dog_vel.x() = std::max(min_dog_velocity, std::min(max_dog_velocity, current_raw_dog_vel.x()));
    current_raw_dog_vel.y() = std::max(min_dog_velocity, std::min(max_dog_velocity, current_raw_dog_vel.y()));
    current_raw_dog_vel.z() = std::max(min_dog_velocity, std::min(max_dog_velocity, current_raw_dog_vel.z()));
    
    // 速度滤波（完全按照callback.cpp的逻辑）
    if (!dog_vel_initialized_) {
        raw_dog_vel_ = current_raw_dog_vel;
        raw_dog_yaw_ = current_raw_dog_yaw;
        dog_vel_initialized_ = true;
    } else {
        // 狗头
        raw_dog_vel_.x() = 0.7 * raw_dog_vel_.x() + 0.3 * current_raw_dog_vel.x();
        // 狗侧
        raw_dog_vel_.y() = 0.7 * raw_dog_vel_.y() + 0.3 * current_raw_dog_vel.y();
        // 狗上
        raw_dog_vel_.z() = current_raw_dog_vel.z();
        
        // yaw滤波（完全按照callback.cpp）
        double delta_yaw = normalizeAngle(current_raw_dog_yaw - raw_dog_yaw_);
        raw_dog_yaw_ += 0.2 * delta_yaw;
        updateDogYawRate(delta_yaw);
    }
    
    // 收到raw dog pos后立即发布处理后的dog_pos
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
        dog_yaw_rate_ = 0.0;
        dog_yaw_rate_initialized_ = true;
    } else {
        // 计算时间差
        double dt = current_time - last_dog_yaw_time_;
        
        if (dt > 0.001) {  // 避免除零，至少1ms间隔
            // 计算瞬时角速度（delta_yaw已经处理了角度循环问题）
            double instant_yaw_rate = delta_yaw / dt;
            
            // 简单滤波
            dog_yaw_rate_ = (1.0 - yaw_rate_filter_gain_) * dog_yaw_rate_ + 
                           yaw_rate_filter_gain_ * instant_yaw_rate;
            
            // 更新上一帧时间
            last_dog_yaw_time_ = current_time;
        }
    }
}

// 处理目标数据（来自read模块的target_ekf_odom）
void DogPosProcessor::targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 从target_ekf_odom中解算yaw（仅使用该消息的四元数）
    const auto& q = msg->pose.pose.orientation;
    double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    target_dog_yaw_ = std::atan2(siny_cosp, cosy_cosp);
    
    // 从target_ekf_odom中提取位置
    target_dog_pos_.x() = msg->pose.pose.position.x;
    target_dog_pos_.y() = msg->pose.pose.position.y;
    target_dog_pos_.z() = msg->pose.pose.position.z;
    
    target_count_++;

    // 仿真模式：直接在目标回调中发布 dog_pos_processed
    if (simulate_mode_) {
        nav_msgs::Odometry out;
        out.header.stamp = ros::Time::now();
        out.header.frame_id = "world";

        // 位置
        out.pose.pose.position.x = msg->pose.pose.position.x;
        out.pose.pose.position.y = msg->pose.pose.position.y;
        out.pose.pose.position.z = msg->pose.pose.position.z;

        // ready flags
        out.pose.pose.orientation.w = 1.0;  // initialized_
        out.pose.pose.orientation.x = 0.0;
        out.pose.pose.orientation.y = 1.0;  // precise_pos_offset_ready_
        out.pose.pose.orientation.z = 1.0;  // precise_yaw_offset_ready_

        // 速度
        out.twist.twist.linear.x = msg->twist.twist.linear.x;
        out.twist.twist.linear.y = msg->twist.twist.linear.y;
        out.twist.twist.linear.z = msg->twist.twist.linear.z;

        // yaw 和 yaw_rate
        out.twist.twist.angular.x = target_dog_yaw_;
        out.twist.twist.angular.y = 0.0;
        out.twist.twist.angular.z = 0.0;

        dog_pos_pub_.publish(out);
    }
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
        // 重置速度相关变量
        dog_yaw_rate_initialized_ = false;
        dog_vel_initialized_ = false;
        dog_yaw_rate_ = 0.0;
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
        if (last_dog_pos_timer_ >= 5) {  // 连续5次没有新包才重置
            raw_dog_pos_received_ = false;
        }
    }
    
    // 检查target_receive状态（模仿dog_pos_received的逻辑）
    if (target_count_ != last_target_count_) {
        last_target_timer_ ++;
        if (last_target_timer_ >= 5) {
            target_receive_ = true;
        }
        last_target_count_ = target_count_;
        last_target_loss_timer_ = 0;
    } else {
        last_target_loss_timer_++;
        if (last_target_loss_timer_ >= 5) {  // 连续5次没有新包才重置
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
    // 处理起飞重新初始化
    if (!initialized_ && raw_dog_pos_received_ && vins_received_) {
        // 使用新的初始化方式：raw_dog_yaw - (vins_yaw + diff)
        double diff = 0.0;
        if (saved_yaw_diff_ != nullptr) {
            diff = *saved_yaw_diff_;
            delete saved_yaw_diff_;
            saved_yaw_diff_ = nullptr;
        } else {
            diff = 0.0;  // 默认diff=0
        }
        
        yaw_offset_ = normalizeAngle(raw_dog_yaw_ - (vins_yaw_ + diff));
        
        // 位置偏移：raw_dog_pos
        // 将vins位置按照yaw_offset旋转后，加到pos_offset后面
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
        
        // 自动进入预设模式
        initialized_ = true;  // 初始化完成
        precise_pos_offset_ready_ = true;
        precise_yaw_offset_ready_ = true;
    }
    
    // 处理hc14_dog数据：当同时收到target和hc14数据时，维护yaw和pos差值补偿；如果使用预设offset，则不进行迭代
    // 只有在初始化后且收到trigger后才进行offset迭代
    if (target_receive_ && raw_dog_pos_received_ && vins_received_) {
        // 正常的yaw offset迭代逻辑
        double current_yaw_offset = normalizeAngle(raw_dog_yaw_ - target_dog_yaw_);
        double yaw_offset_diff = normalizeAngle(current_yaw_offset - yaw_offset_);
        // 处理角度环绕问题
        yaw_offset_diff = normalizeAngle(yaw_offset_diff);
        
        // 对yaw offset进行线性滤波，防止突变
        yaw_offset_ += yaw_filter_gain_ * yaw_offset_diff;
        
        // 标记hc14_dog信息可用，采用计数器方式防止抖动
        if (std::abs(yaw_offset_diff) < yaw_stable_threshold_) {
            if (!precise_yaw_offset_ready_) {
                ROS_INFO("precise_yaw_offset_ready!");
                if (precise_pos_offset_ready_) {
                    initialized_ = true;  // 通过迭代达到precise_yaw_offset_ready和precise_pos_offset_ready，设置initialized
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
            // 更新位置偏移：raw_dog_pos - vins_pos
            Eigen::Vector3d raw_dog_pos(
                raw_dog_pos_->pose.pose.position.x,
                raw_dog_pos_->pose.pose.position.y,
                raw_dog_pos_->pose.pose.position.z
            );

            // 计算前馈偏移量（基于final_dog_vel，在世界坐标系下）
            // 参考traj_server: offset = camera_offset * vel
            // target_dog_pos先加上前馈偏移量（在世界坐标系下）
            Eigen::Vector3d target_dog_pos_with_ff(
                target_dog_pos_.x() + camera_offset_ * final_dog_vel_.x(),
                target_dog_pos_.y() + camera_offset_ * final_dog_vel_.y(),
                target_dog_pos_.z()
            );
            
            // 然后将加上前馈后的target_dog_pos旋转到狗坐标系
            double cos_yaw = std::cos(yaw_offset_);
            double sin_yaw = std::sin(yaw_offset_);
            Eigen::Vector3d rotated_target(
                cos_yaw * target_dog_pos_with_ff.x() - sin_yaw * target_dog_pos_with_ff.y(),
                sin_yaw * target_dog_pos_with_ff.x() + cos_yaw * target_dog_pos_with_ff.y(),
                target_dog_pos_with_ff.z()
            );
            
            Eigen::Vector3d current_pos_offset = raw_dog_pos - rotated_target;
            
            // 对位置偏移进行线性滤波，防止突变
            Eigen::Vector3d pos_offset_diff = current_pos_offset - pos_offset_;
            pos_offset_ += pos_filter_gain_ * pos_offset_diff;
            
            // 计算位置偏移差值
            double pos_offset_diff_norm = (current_pos_offset - pos_offset_).norm();
            
            // 标记位置偏移是否精确ready
            if (pos_offset_diff_norm < pos_stable_threshold_) {
                if (!precise_pos_offset_ready_) {
                    ROS_INFO("precise_pos_offset_ready!");
                    if (precise_yaw_offset_ready_) {
                        initialized_ = true;  // 通过迭代达到precise_yaw_offset_ready和precise_pos_offset_ready，设置initialized
                    }
                    // 位置收敛时重置卡尔曼滤波器
                    ROS_INFO("Position converged, resetting Kalman filter");
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
        if (std::abs(heading_diff) > M_PI / 6.0) {  // 30度
            return;
        }

        // 利用单个距离和相对角度估计狗的世界坐标
        double global_yaw_to_dog = normalizeAngle(vins_yaw_ + aoa_angle_);
        Eigen::Vector3d aoa_pos_world(
            vins_pos_.x() + aoa_distance_ * std::cos(global_yaw_to_dog),
            vins_pos_.y() + aoa_distance_ * std::sin(global_yaw_to_dog),
            vins_pos_.z()
        );

        // 将AOA估计位置旋转到狗坐标系（使用 yaw_offset_）
        double cos_yaw = std::cos(yaw_offset_);
        double sin_yaw = std::sin(yaw_offset_);
        double rotated_aoa_x = cos_yaw * (aoa_pos_world.x() - vins_pos_.x()) - sin_yaw * (aoa_pos_world.y() - vins_pos_.y());
        double rotated_aoa_y = sin_yaw * (aoa_pos_world.x() - vins_pos_.x()) + cos_yaw * (aoa_pos_world.y() - vins_pos_.y());
        Eigen::Vector3d rotated_aoa_pos(rotated_aoa_x, rotated_aoa_y, aoa_pos_world.z());

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

        // 位置偏移滤波
        Eigen::Vector3d pos_offset_diff = current_pos_offset - pos_offset_;
        double diff_norm = pos_offset_diff.norm();
        if (diff_norm > 1e-6) {
            // 按比例逼近，单次步长上限 aoa_pos_step_limit_
            double step = aoa_pos_filter_gain_ * diff_norm;
            step = std::min(step, aoa_pos_step_limit_);
            pos_offset_ += pos_offset_diff * (step / diff_norm);
        }
    }
}

// 发布处理后的dog_pos（使用Odometry格式）
void DogPosProcessor::publishProcessedDogPos() {
    if (raw_dog_pos_ == nullptr) return;
    
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
    if (target_receive_) {
        final_dog_pos_.x() = target_dog_pos_.x() + camera_offset_ * final_dog_vel_.x();
        final_dog_pos_.y() = target_dog_pos_.y() + camera_offset_ * final_dog_vel_.y();
        final_dog_pos_.z() = target_dog_pos_.z();
        final_dog_yaw_ = target_dog_yaw_;
    }

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
    
    // 使用self的变量填充msg
    msg.pose.pose.position.x = final_dog_pos_.x();
    msg.pose.pose.position.y = final_dog_pos_.y();
    msg.pose.pose.position.z = final_dog_pos_.z();

    // 状态位：w=initialized, x=aoa_converged, y=precise_pos_ready, z=precise_yaw_ready
    msg.pose.pose.orientation.w = initialized_ ? 1.0 : 0.0;
    msg.pose.pose.orientation.x = 0.0;
    msg.pose.pose.orientation.y = precise_pos_offset_ready_ ? 1.0 : 0.0;
    msg.pose.pose.orientation.z = precise_yaw_offset_ready_ ? 1.0 : 0.0;

    // 速度
    msg.twist.twist.linear.x = final_dog_vel_.x();
    msg.twist.twist.linear.y = final_dog_vel_.y();
    msg.twist.twist.linear.z = final_dog_vel_.z();
    
    // 最终yaw放在twist.angular.x,另外，y和z放上转换后的狗的速度
    msg.twist.twist.angular.x = final_dog_yaw_;
    msg.twist.twist.angular.y = dog_yaw_rate_;  // 发布狗通信角速度
    msg.twist.twist.angular.z = 0.0;

    dog_pos_pub_.publish(msg);
}

void DogPosProcessor::aoaCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 单距离 + 单角度：position.x = 距离，orientation.x = 相对于无人机朝向的角度
    double distance_raw = msg->pose.pose.position.x;
    double angle_raw = msg->pose.pose.orientation.x;
    
    // 检查距离有效性
    if (distance_raw < aoa_min_distance_) {
        return;
    }

    // 用光流高度修正：水平距离 = sqrt(max(0, d^2 - h^2))
    double height = (flow_z_ - flow_height_bias_);
    double d_sq = std::max(0.0, distance_raw * distance_raw - height * height);
    aoa_distance_ = std::sqrt(d_sq);

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

