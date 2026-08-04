#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/TakeoffLand.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>

/*
 * Dog Position Processor — Joint EKF Baseline
 * ===========================================
 * 与 dog_pos_processor.cpp（级联解耦：外参低通 + 输出 KF）对标的 baseline。
 *
 * 主方案把「外参估计」和「目标位姿融合」拆成两级；本文件用单一 EKF 同时估：
 *
 *   x = [px, py, pz, vx, vy, vz, yaw_offset, off_x, off_y, off_z]  (10 维)
 *
 *   - (p, v)           : 世界系下目标（狗）位姿
 *   - yaw_offset       : raw /dog_pos 航向 与 目标航向 的固定偏差（外参 yaw）
 *   - off              : raw 坐标系到世界系的平移外参
 *
 * 运动模型：恒定速度；外参 yaw/平移视为随机游走（Q 很小）。
 *
 * 量测更新（均为 EKF 线性/线性化观测）：
 *   /target_ekf_odom  → 直接观测 (p, v)，线性 H
 *   /dog_pos          → raw_pos ≈ R(yaw_off)*p + off，raw_vel 经 yaw 差旋转后观测 v
 *   /AOA_Tag_data     → 几何解算世界系位置，线性观测 p（20Hz 定时器里更新）
 *
 * 数据流：各 topic 回调里 predict→update；50ms 定时器做 AOA 更新并发布。
 * 发布：/dog_pos_processed_kf（与主方案 /dog_pos_processed 区分，可同时运行）
 */

namespace {

constexpr int kNState = 10;
constexpr int kIdxYawOff = 6;  // 航向外参：raw_yaw - target_yaw
constexpr int kIdxOff = 7;     // 平移外参起点索引 (off_x, off_y, off_z)

// 与 dog_pos_processor.cpp 对齐的非架构参数（Q/R、阈值、预处理增益等）
constexpr double kTimerDt = 0.05;
constexpr double kExtrinsicAlpha = 0.04;
constexpr double kExtrinsicMeasStd = 0.05;
constexpr double kExtrinsicMeasVar = kExtrinsicMeasStd * kExtrinsicMeasStd;
constexpr double kExtrinsicQ = kExtrinsicAlpha * kExtrinsicMeasVar / (1.0 - kExtrinsicAlpha);
constexpr double kExtrinsicR = kExtrinsicMeasVar;
constexpr double kOutputProcessNoise = 0.01;
constexpr double kOutputMeasNoise = 0.1;
constexpr double kAoaPosFilterGain = 0.05;
constexpr double kAoaPosStepLimit = 0.02;
constexpr double kYawRateFilterGain = 0.3;
constexpr double kCameraOffset = 0.36;
constexpr double kAoaMinDistance = 3.0;
constexpr double kAoaHeadingLimit = M_PI / 4.0;
constexpr double kJointQPos = kOutputProcessNoise / kTimerDt;
constexpr double kJointQVel = kOutputProcessNoise / kTimerDt;
constexpr double kJointQYawOff = kExtrinsicQ / kTimerDt;
constexpr double kJointQPosOff = kExtrinsicQ / kTimerDt;
// 主方案 AOA 以 gain=0.05 修正外参；联合 EKF 等效为更大 R（更低信任）
constexpr double kAoaMeasNoise = kExtrinsicR / kAoaPosFilterGain;

void logIterMsThrottled(const char* tag,
                          const std::chrono::steady_clock::time_point& t0) {
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    ROS_INFO_THROTTLE(1.0, "%s: one iter %.3f ms", tag, ms);
}

double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

Eigen::Matrix3d rotZ(double yaw) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    Eigen::Matrix3d R;
    R << c, -s, 0.0,
         s,  c, 0.0,
         0.0, 0.0, 1.0;
    return R;
}

Eigen::Matrix3d dRotZdYaw(double yaw) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    Eigen::Matrix3d dR;
    dR << -s, -c, 0.0,
           c, -s, 0.0,
         0.0, 0.0, 0.0;
    return dR;
}

class JointDogPosEKF {
public:
    JointDogPosEKF() {
        x_ = Eigen::VectorXd::Zero(kNState);
        P_ = Eigen::MatrixXd::Identity(kNState, kNState) * 10.0;
    }

    void reset() {
        x_.setZero();
        P_ = Eigen::MatrixXd::Identity(kNState, kNState) * 10.0;
        initialized_ = false;
        last_predict_time_ = ros::Time(0);
    }

    bool initialized() const { return initialized_; }

    // 首次初始化：需 raw_dog + vins；若有 target 则以其为 (p,v) 初值，否则从 raw 反推
    void initializeFromMeasurements(
        const Eigen::Vector3d* target_pos,
        const Eigen::Vector3d* target_vel,
        const double* target_yaw,
        const Eigen::Vector3d& raw_pos,
        const Eigen::Vector3d& raw_vel,
        double raw_yaw,
        double vins_yaw,
        const double* saved_yaw_diff) {
        if (target_pos) {
            // 有视觉 target：直接信任其世界系位姿
            x_.segment<3>(0) = *target_pos;
            if (target_vel) x_.segment<3>(3) = *target_vel;
            if (target_yaw) {
                x_[kIdxYawOff] = normalizeAngle(raw_yaw - *target_yaw);
            } else {
                // 无 target yaw 时用 vins_yaw + 预设 yaw_diff 近似
                const double diff = saved_yaw_diff ? *saved_yaw_diff : 0.0;
                x_[kIdxYawOff] = normalizeAngle(raw_yaw - (vins_yaw + diff));
            }
        } else {
            // 仅 raw：用 vins 航向先估 yaw_off，再旋转 raw 得到世界系 (p,v)
            const double diff = saved_yaw_diff ? *saved_yaw_diff : 0.0;
            x_[kIdxYawOff] = normalizeAngle(raw_yaw - (vins_yaw + diff));
            const double yaw_inv = -x_[kIdxYawOff];
            x_.segment<3>(0) = rotZ(yaw_inv) * raw_pos;
            const double yaw_diff = normalizeAngle(raw_yaw - x_[kIdxYawOff]);
            x_.segment<3>(3) = rotZ(yaw_diff) * raw_vel;
        }

        // 平移外参：使 raw_pos = R(yaw_off)*p + off 在当前帧成立
        const Eigen::Matrix3d R = rotZ(x_[kIdxYawOff]);
        x_.segment<3>(kIdxOff) = raw_pos - R * x_.segment<3>(0);

        P_.setZero();
        P_(0, 0) = P_(1, 1) = P_(2, 2) = 1.0;
        P_(3, 3) = P_(4, 4) = P_(5, 5) = 0.5;
        P_(6, 6) = 0.2;
        P_(7, 7) = P_(8, 8) = P_(9, 9) = 0.2;

        initialized_ = true;
        last_predict_time_ = ros::Time::now();
    }

    // 恒定速度模型；Q 随 dt 缩放（q_rate * dt，名义 dt=kTimerDt 时与 kOutputProcessNoise/kExtrinsicQ 一致）
    void predict(double dt) {
        if (!initialized_ || dt <= 0.0) return;

        Eigen::MatrixXd F = Eigen::MatrixXd::Identity(kNState, kNState);
        F(0, 3) = F(1, 4) = F(2, 5) = dt;  // p += v*dt
        x_ = F * x_;

        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kNState, kNState);
        Q(0, 0) = Q(1, 1) = Q(2, 2) = q_pos_ * dt;
        Q(3, 3) = Q(4, 4) = Q(5, 5) = q_vel_ * dt;
        Q(6, 6) = q_yaw_off_ * dt;
        Q(7, 7) = Q(8, 8) = Q(9, 9) = q_pos_off_ * dt;

        P_ = F * P_ * F.transpose() + Q;
    }

    // target_ekf_odom：线性观测 z = [p; v]
    void updateTarget(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel) {
        Eigen::VectorXd z(6);
        z << pos, vel;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6, kNState);
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        H.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6);
        R.block<3, 3>(0, 0) *= r_target_pos_;
        R.block<3, 3>(3, 3) *= r_target_vel_;
        ekfUpdate(z, H * x_, H, R);
    }

    // /dog_pos 非线性外参约束，拆成位置、速度两次序贯 EKF 更新
    void updateRawDog(
        const Eigen::Vector3d& raw_pos,
        const Eigen::Vector3d& raw_vel,
        double raw_yaw) {
        const Eigen::Vector3d p_w = x_.segment<3>(0);
        const double yaw_off = x_[kIdxYawOff];
        const Eigen::Vector3d off = x_.segment<3>(kIdxOff);

        // h_pos = R(yaw_off) * p + off，对 yaw_off 求雅可比 dR/dyaw * p
        const Eigen::Matrix3d R = rotZ(yaw_off);
        const Eigen::Vector3d h_pos = R * p_w + off;
        Eigen::MatrixXd H_pos = Eigen::MatrixXd::Zero(3, kNState);
        H_pos.block<3, 3>(0, 0) = R;
        H_pos.block<3, 1>(0, kIdxYawOff) = dRotZdYaw(yaw_off) * p_w;
        H_pos.block<3, 3>(0, kIdxOff) = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R_pos = Eigen::Matrix3d::Identity() * r_raw_pos_;
        ekfUpdate(raw_pos, h_pos, H_pos, R_pos);

        // raw_vel 在 raw 系，先旋到世界系：z = R(raw_yaw - yaw_off) * raw_vel ≈ v
        const double yaw_diff = normalizeAngle(raw_yaw - x_[kIdxYawOff]);
        const Eigen::Matrix3d Rvd = rotZ(yaw_diff);
        const Eigen::Vector3d z_vel = Rvd * raw_vel;
        const Eigen::Vector3d h_vel = x_.segment<3>(3);
        Eigen::MatrixXd H_vel = Eigen::MatrixXd::Zero(3, kNState);
        H_vel.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        H_vel.block<3, 1>(0, kIdxYawOff) = -dRotZdYaw(yaw_diff) * raw_vel;
        Eigen::Matrix3d R_vel = Eigen::Matrix3d::Identity() * r_raw_vel_;
        ekfUpdate(z_vel, h_vel, H_vel, R_vel);
    }

    // AOA 几何解算得到的世界系位置，直接观测 p
    void updateAoaPosition(const Eigen::Vector3d& pos_world) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, kNState);
        H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * r_aoa_pos_;
        ekfUpdate(pos_world, x_.segment<3>(0), H, R);
    }

    Eigen::Vector3d pos() const { return x_.segment<3>(0); }
    Eigen::Vector3d vel() const { return x_.segment<3>(3); }
    double yawOffset() const { return x_[kIdxYawOff]; }
    Eigen::Vector3d posOffset() const { return x_.segment<3>(kIdxOff); }
    const Eigen::MatrixXd& covariance() const { return P_; }

    ros::Time lastPredictTime() const { return last_predict_time_; }
    void setLastPredictTime(const ros::Time& t) { last_predict_time_ = t; }

private:
    // 标准 EKF 更新：h 为在当前 x_ 处线性化的预测量测
    void ekfUpdate(
        const Eigen::VectorXd& z,
        const Eigen::VectorXd& h,
        const Eigen::MatrixXd& H,
        const Eigen::MatrixXd& R) {
        if (!initialized_) return;
        const Eigen::VectorXd y = z - h;
        const Eigen::MatrixXd S = H * P_ * H.transpose() + R;
        const Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
        x_ = x_ + K * y;
        P_ = (Eigen::MatrixXd::Identity(kNState, kNState) - K * H) * P_;
    }

    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;
    bool initialized_ = false;
    ros::Time last_predict_time_;

    // 过程噪声 Q（与 dog_pos_processor 对齐）
    double q_pos_ = kJointQPos;
    double q_vel_ = kJointQVel;
    double q_yaw_off_ = kJointQYawOff;
    double q_pos_off_ = kJointQPosOff;

    // 量测噪声 R：外参约束用 kExtrinsicR，输出层用 kOutputMeasNoise
    double r_target_pos_ = kOutputMeasNoise;
    double r_target_vel_ = kOutputMeasNoise;
    double r_raw_pos_ = kExtrinsicR;
    double r_raw_vel_ = kOutputMeasNoise;
    double r_aoa_pos_ = kAoaMeasNoise;
};

// ROS 封装：订阅多源量测，驱动 EKF，发布与主方案相同格式的 /dog_pos_processed
class DogPosProcessorKF {
public:
    DogPosProcessorKF()
        : nh_("~"),
          saved_yaw_diff_(nullptr),
          raw_dog_yaw_(0.0),
          raw_dog_vel_(Eigen::Vector3d::Zero()),
          dog_yaw_rate_(0.0),
          dog_yaw_rate_initialized_(false),
          last_dog_yaw_time_(0.0),
          target_yaw_(0.0),
          vins_yaw_(0.0),
          R_wb_(Eigen::Matrix3d::Identity()),
          aoa_distance_(0.0),
          aoa_angle_(0.0),
          raw_dog_pos_received_(false),
          target_receive_(false),
          vins_received_(false),
          aoa_received_(false),
          raw_dog_pos_count_(0),
          last_raw_dog_pos_count_(0),
          last_dog_pos_timer_(0),
          target_count_(0),
          last_target_count_(0),
          last_target_timer_(0),
          last_target_loss_timer_(0),
          vins_count_(0),
          last_vins_count_(0),
          last_vins_timer_(0),
          aoa_count_(0),
          last_aoa_count_(0),
          last_aoa_timer_(0),
          simulate_mode_(false) {
        camera_offset_ = kCameraOffset;
        aoa_min_distance_ = kAoaMinDistance;
        simulate_mode_ = true;
        if (simulate_mode_) {
            camera_offset_ = 0.0;
        }

        dog_pos_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_processed_kf", 10);
        aoa_debug_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_aoa_debug_kf", 10);

        raw_dog_pos_sub_ = nh_.subscribe("/dog_pos", 10, &DogPosProcessorKF::rawDogPosCallback, this);
        target_sub_ = nh_.subscribe("/target_ekf_odom", 10, &DogPosProcessorKF::targetCallback, this);
        vins_sub_ = nh_.subscribe("/vins_fusion/imu_propagate", 10, &DogPosProcessorKF::vinsCallback, this);
        aoa_sub_ = nh_.subscribe("/AOA_Tag_data", 10, &DogPosProcessorKF::aoaCallback, this);
        flow_sub_ = nh_.subscribe("/flow_data", 10, &DogPosProcessorKF::flowCallback, this);
        takeoff_sub_ = nh_.subscribe("/px4ctrl/takeoff_land", 10, &DogPosProcessorKF::takeoffCallback, this);
        yaw_diff_sub_ = nh_.subscribe("/yaw_diff_preset", 10, &DogPosProcessorKF::yawDiffCallback, this);
        test_reset_sub_ = nh_.subscribe("/test_reset_initialized", 10, &DogPosProcessorKF::testResetCallback, this);
        trigger_sub_ = nh_.subscribe("/triger", 10, &DogPosProcessorKF::triggerCallback, this);

        timer_ = nh_.createTimer(ros::Duration(0.05), &DogPosProcessorKF::processCallback, this);
        status_timer_ = nh_.createTimer(ros::Duration(0.1), &DogPosProcessorKF::statusCheckCallback, this);

        ROS_INFO(
            "dog_pos_processor_kf: joint EKF (extrinsic Q=%.2e R=%.2e, output Q=%.3f R=%.3f, aoa R=%.3f)",
            kExtrinsicQ, kExtrinsicR, kOutputProcessNoise, kOutputMeasNoise, kAoaMeasNoise);
    }

    void spin() { ros::spin(); }

private:
    // 每次量测更新前先 predict 到当前时刻，保证时间对齐
    void predictToNow() {
        if (!ekf_.initialized()) return;
        const ros::Time now = ros::Time::now();
        if (ekf_.lastPredictTime().isZero()) {
            ekf_.setLastPredictTime(now);
            return;
        }
        double dt = (now - ekf_.lastPredictTime()).toSec();
        if (dt > 1.0) {
            ROS_WARN_THROTTLE(2.0, "Joint EKF gap %.2fs", dt);
        }
        ekf_.predict(dt);
        ekf_.setLastPredictTime(now);
    }

    // 等待 raw_dog + vins 到位后一次性初始化 EKF 状态
    void maybeInit() {
        if (ekf_.initialized() || !raw_dog_pos_received_ || !vins_received_ || !raw_dog_pos_) {
            return;
        }
        const Eigen::Vector3d raw_pos(
            raw_dog_pos_->pose.pose.position.x,
            raw_dog_pos_->pose.pose.position.y,
            raw_dog_pos_->pose.pose.position.z);
        const Eigen::Vector3d raw_vel(
            raw_dog_pos_->twist.twist.linear.x,
            raw_dog_pos_->twist.twist.linear.y,
            raw_dog_pos_->twist.twist.linear.z);

        const Eigen::Vector3d* tgt_pos = target_receive_ ? &target_pos_ : nullptr;
        const Eigen::Vector3d* tgt_vel = target_receive_ ? &target_vel_ : nullptr;
        const double* tgt_yaw = target_receive_ ? &target_yaw_ : nullptr;

        ekf_.initializeFromMeasurements(
            tgt_pos, tgt_vel, tgt_yaw,
            raw_pos, raw_vel, raw_dog_yaw_, vins_yaw_,
            saved_yaw_diff_);

        if (saved_yaw_diff_) {
            delete saved_yaw_diff_;
            saved_yaw_diff_ = nullptr;
        }

        ROS_INFO("Joint EKF initialized: yaw_off=%.1f deg pos_off=[%.3f, %.3f, %.3f]",
                 ekf_.yawOffset() * 180.0 / M_PI,
                 ekf_.posOffset().x(), ekf_.posOffset().y(), ekf_.posOffset().z());
    }

    void updateDogYawRate(double delta_yaw) {
        const double now = ros::Time::now().toSec();
        if (!dog_yaw_rate_initialized_) {
            last_dog_yaw_time_ = now;
            dog_yaw_rate_ = 0.0;
            dog_yaw_rate_initialized_ = true;
            return;
        }
        const double dt = now - last_dog_yaw_time_;
        if (dt > 0.001) {
            const double instant = delta_yaw / dt;
            dog_yaw_rate_ = (1.0 - kYawRateFilterGain) * dog_yaw_rate_
                          + kYawRateFilterGain * instant;
            last_dog_yaw_time_ = now;
        }
    }

    void rawDogPosCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        raw_dog_pos_ = msg;
        raw_dog_pos_count_++;

        const double prev_yaw = raw_dog_yaw_;
        raw_dog_yaw_ = msg->pose.pose.orientation.w;
        raw_dog_vel_.x() = msg->twist.twist.linear.x;
        raw_dog_vel_.y() = msg->twist.twist.linear.y;
        raw_dog_vel_.z() = msg->twist.twist.linear.z;
        updateDogYawRate(normalizeAngle(raw_dog_yaw_ - prev_yaw));

        maybeInit();
        if (!ekf_.initialized()) return;

        const auto iter_t0 = std::chrono::steady_clock::now();
        predictToNow();
        const Eigen::Vector3d raw_pos(
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z);
        ekf_.updateRawDog(raw_pos, raw_dog_vel_, raw_dog_yaw_);
        logIterMsThrottled("dog_pos_processor_kf [raw]", iter_t0);
    }

    void targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        target_yaw_ = msg->pose.pose.orientation.w;
        target_pos_ << msg->pose.pose.position.x,
                       msg->pose.pose.position.y,
                       msg->pose.pose.position.z;
        target_vel_ << msg->twist.twist.linear.x,
                       msg->twist.twist.linear.y,
                       msg->twist.twist.linear.z;
        target_count_++;

        maybeInit();
        if (!ekf_.initialized()) return;

        predictToNow();
        Eigen::Vector3d pos_meas = target_pos_;
        if (camera_offset_ > 0.0) {
            pos_meas.x() += camera_offset_ * target_vel_.x();
            pos_meas.y() += camera_offset_ * target_vel_.y();
        }
        ekf_.updateTarget(pos_meas, target_vel_);
    }

    void vinsCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        const double qw = msg->pose.pose.orientation.w;
        const double qx = msg->pose.pose.orientation.x;
        const double qy = msg->pose.pose.orientation.y;
        const double qz = msg->pose.pose.orientation.z;
        R_wb_ = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
        const double siny = 2.0 * (qw * qz + qx * qy);
        const double cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
        vins_yaw_ = std::atan2(siny, cosy);
        vins_pos_ << msg->pose.pose.position.x,
                     msg->pose.pose.position.y,
                     msg->pose.pose.position.z;
        vins_count_++;
    }

    void aoaCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        aoa_distance_ = msg->pose.pose.position.x;
        aoa_angle_ = msg->pose.pose.orientation.x;
        aoa_count_++;
    }

    void flowCallback(const nav_msgs::Odometry::ConstPtr& /*msg*/) {}

    void takeoffCallback(const quadrotor_msgs::TakeoffLand::ConstPtr& msg) {
        if (msg->takeoff_land_cmd == 1) {
            ROS_INFO("Takeoff: reset joint EKF");
            ekf_.reset();
            dog_yaw_rate_initialized_ = false;
            dog_yaw_rate_ = 0.0;
        }
    }

    void yawDiffCallback(const std_msgs::Float64::ConstPtr& msg) {
        if (saved_yaw_diff_) delete saved_yaw_diff_;
        saved_yaw_diff_ = new double(msg->data);
        ROS_INFO("Received yaw diff preset: %.1f deg", msg->data * 180.0 / M_PI);
    }

    void testResetCallback(const std_msgs::Bool::ConstPtr& /*msg*/) {
        ROS_INFO("Test reset: clear joint EKF");
        ekf_.reset();
        dog_yaw_rate_initialized_ = false;
    }

    void triggerCallback(const geometry_msgs::PoseStamped::ConstPtr& /*msg*/) {
        ROS_INFO("Trigger received");
    }

    // 100ms：根据 topic 计数判断各源是否仍在线（与主方案相同逻辑）
    void statusCheckCallback(const ros::TimerEvent& /*event*/) {
        if (raw_dog_pos_count_ != last_raw_dog_pos_count_) {
            raw_dog_pos_received_ = true;
            last_raw_dog_pos_count_ = raw_dog_pos_count_;
            last_dog_pos_timer_ = 0;
        } else {
            last_dog_pos_timer_++;
            if (last_dog_pos_timer_ >= 1) raw_dog_pos_received_ = false;
        }

        if (target_count_ != last_target_count_) {
            last_target_timer_++;
            if (last_target_timer_ >= 1) {
                target_receive_ = true;
            }
            last_target_count_ = target_count_;
            last_target_loss_timer_ = 0;
        } else {
            last_target_loss_timer_++;
            if (last_target_loss_timer_ >= 1) target_receive_ = false;
            last_target_timer_ = 0;
        }

        if (vins_count_ != last_vins_count_) {
            vins_received_ = true;
            last_vins_count_ = vins_count_;
            last_vins_timer_ = 0;
        } else {
            last_vins_timer_++;
            if (last_vins_timer_ >= 5) vins_received_ = false;
        }

        if (aoa_count_ != last_aoa_count_) {
            aoa_received_ = true;
            last_aoa_count_ = aoa_count_;
            last_aoa_timer_ = 0;
        } else {
            last_aoa_timer_++;
            if (last_aoa_timer_ >= 5) aoa_received_ = false;
        }
    }

    // 由 UWB 距离 + 方位角 + VINS 位姿，几何解算狗在世界系的位置（与主方案一致）
    bool computeAoaWorldPos(Eigen::Vector3d& out) const {
        if (!ekf_.initialized() || !vins_received_ || !aoa_received_) return false;
        if (aoa_distance_ <= aoa_min_distance_) return false;

        const Eigen::Vector3d p_w = ekf_.pos();
        const double height_diff = p_w.z() - vins_pos_.z();
        if (aoa_distance_ < std::abs(height_diff)) return false;

        const Eigen::Vector3d dog_vec = p_w - vins_pos_;
        const double heading_to_dog = std::atan2(dog_vec.y(), dog_vec.x());
        if (std::abs(normalizeAngle(heading_to_dog - vins_yaw_)) > kAoaHeadingLimit) return false;

        const double horiz = std::sqrt(aoa_distance_ * aoa_distance_ - height_diff * height_diff);
        const Eigen::Vector3d ez(0.0, 0.0, 1.0);
        const Eigen::Vector3d bearing_b(
            std::cos(aoa_angle_), std::sin(aoa_angle_), 0.0);
        const Eigen::Vector3d n_b = bearing_b.cross(ez);
        Eigen::Vector3d n_w = R_wb_ * n_b;
        const double n_norm = n_w.norm();
        if (n_norm < 1e-8) return false;
        n_w /= n_norm;

        Eigen::Vector3d d_w = -n_w.cross(ez);
        if (d_w.norm() < 1e-6) return false;
        const Eigen::Vector3d d_hat = d_w.normalized();

        const double n_z = n_w.dot(ez);
        const Eigen::Vector3d n_h = n_w - n_z * ez;
        const double n_h_sq = n_h.squaredNorm();
        if (n_h_sq < 1e-8) return false;

        const Eigen::Vector3d p0 = height_diff * ez - (n_z * height_diff / n_h_sq) * n_h;
        const Eigen::Vector2d p0_h(p0.x(), p0.y());
        const double inside = horiz * horiz - p0_h.squaredNorm();
        if (inside < 0.0) return false;

        out = vins_pos_ + p0 + std::sqrt(inside) * d_hat;
        return true;
    }

    // 50ms 主循环：AOA 量测更新 + 发布（raw/target 在各自回调里已更新）
    void processCallback(const ros::TimerEvent& /*event*/) {
        if (!ekf_.initialized()) return;

        const auto iter_t0 = std::chrono::steady_clock::now();
        predictToNow();
        Eigen::Vector3d aoa_pos;
        if (computeAoaWorldPos(aoa_pos)) {
            ekf_.updateAoaPosition(aoa_pos);
            nav_msgs::Odometry dbg;
            dbg.header.stamp = ros::Time::now();
            dbg.header.frame_id = "world";
            dbg.pose.pose.position.x = aoa_pos.x();
            dbg.pose.pose.position.y = aoa_pos.y();
            dbg.pose.pose.position.z = aoa_pos.z();
            aoa_debug_pub_.publish(dbg);
        }
        publishProcessed();
        logIterMsThrottled("dog_pos_processor_kf [timer]", iter_t0);
    }

    void publishProcessed() {
        const Eigen::Vector3d pos = ekf_.pos();
        const Eigen::Vector3d vel = ekf_.vel();
        const double out_yaw = normalizeAngle(raw_dog_yaw_ - ekf_.yawOffset());

        nav_msgs::Odometry msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "world";
        msg.pose.pose.position.x = pos.x();
        msg.pose.pose.position.y = pos.y();
        msg.pose.pose.position.z = pos.z();
        msg.pose.pose.orientation.w = 1.0;  // pos_ready（降落实验恒 true）
        msg.pose.pose.orientation.x = 1.0;  // yaw_ready（降落实验恒 true）
        msg.pose.pose.orientation.y = 0.0;
        msg.pose.pose.orientation.z = 0.0;
        msg.twist.twist.linear.x = vel.x();
        msg.twist.twist.linear.y = vel.y();
        msg.twist.twist.linear.z = vel.z();
        msg.twist.twist.angular.x = out_yaw;
        msg.twist.twist.angular.y = dog_yaw_rate_;
        msg.twist.twist.angular.z = 0.0;
        dog_pos_pub_.publish(msg);
    }

    ros::NodeHandle nh_;
    JointDogPosEKF ekf_;

    ros::Publisher dog_pos_pub_;
    ros::Publisher aoa_debug_pub_;
    ros::Subscriber raw_dog_pos_sub_;
    ros::Subscriber target_sub_;
    ros::Subscriber vins_sub_;
    ros::Subscriber aoa_sub_;
    ros::Subscriber flow_sub_;
    ros::Subscriber takeoff_sub_;
    ros::Subscriber yaw_diff_sub_;
    ros::Subscriber test_reset_sub_;
    ros::Subscriber trigger_sub_;
    ros::Timer timer_;
    ros::Timer status_timer_;

    double camera_offset_;
    double aoa_min_distance_;
    bool simulate_mode_;

    double* saved_yaw_diff_;
    double raw_dog_yaw_;
    Eigen::Vector3d raw_dog_vel_;
    double dog_yaw_rate_;
    bool dog_yaw_rate_initialized_;
    double last_dog_yaw_time_;

    nav_msgs::Odometry::ConstPtr raw_dog_pos_;
    Eigen::Vector3d target_pos_;
    Eigen::Vector3d target_vel_;
    double target_yaw_;

    double vins_yaw_;
    Eigen::Vector3d vins_pos_;
    Eigen::Matrix3d R_wb_;

    double aoa_distance_;
    double aoa_angle_;

    bool raw_dog_pos_received_;
    bool target_receive_;
    bool vins_received_;
    bool aoa_received_;

    unsigned int raw_dog_pos_count_;
    unsigned int last_raw_dog_pos_count_;
    int last_dog_pos_timer_;
    unsigned int target_count_;
    unsigned int last_target_count_;
    int last_target_timer_;
    int last_target_loss_timer_;
    unsigned int vins_count_;
    unsigned int last_vins_count_;
    int last_vins_timer_;
    unsigned int aoa_count_;
    unsigned int last_aoa_count_;
    int last_aoa_timer_;
};

}  // namespace

int main(int argc, char** argv) {
    ros::init(argc, argv, "dog_pos_processor_kf");
    DogPosProcessorKF node;
    node.spin();
    return 0;
}
