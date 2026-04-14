#include "dog_pos_processor.h"
#include <ros/ros.h>
#include <Eigen/Dense>
#include <cmath>

/*
 * Dog Position Processor (Vision-only Base)
 * 仅接收 target_ekf_odom 视觉信息，经 LKF 滤波后发布 dog_pos_processed
 */

// ===================== KalmanFilter 实现 =====================
KalmanFilter::KalmanFilter(double process_noise, double measurement_noise)
    : initialized_(false) {
    state_ = Eigen::VectorXd::Zero(6);
    covariance_ = Eigen::MatrixXd::Identity(6, 6) * 100.0;
    Q_ = Eigen::MatrixXd::Identity(6, 6) * process_noise;
    R_ = Eigen::MatrixXd::Identity(6, 6) * measurement_noise;
    F_base_ = Eigen::MatrixXd::Identity(6, 6);
    H_ = Eigen::MatrixXd::Identity(6, 6);
}

void KalmanFilter::predict(double dt) {
    if (!initialized_) return;
    Eigen::MatrixXd F = F_base_;
    F(0, 3) = dt;
    F(1, 4) = dt;
    F(2, 5) = dt;
    state_ = F * state_;
    covariance_ = F * covariance_ * F.transpose() + Q_;
}

void KalmanFilter::update(const Eigen::VectorXd& measurement) {
    if (!initialized_) {
        state_ = measurement;
        initialized_ = true;
        return;
    }
    Eigen::VectorXd residual = measurement - H_ * state_;
    Eigen::MatrixXd S = H_ * covariance_ * H_.transpose() + R_;
    Eigen::MatrixXd K = covariance_ * H_.transpose() * S.inverse();
    state_ = state_ + K * residual;
    covariance_ = (Eigen::MatrixXd::Identity(6, 6) - K * H_) * covariance_;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> KalmanFilter::filter(
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& velocity,
    double dt) {
    Eigen::VectorXd measurement(6);
    measurement << position, velocity;
    predict(dt);
    update(measurement);
    return std::make_pair(state_.head<3>(), state_.tail<3>());
}

void KalmanFilter::reset() {
    initialized_ = false;
}

// ===================== DogPosProcessor 实现 =====================
double DogPosProcessor::normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

double DogPosProcessor::calculateYawOffsetVariance() { return 0.0; }
double DogPosProcessor::calculatePosOffsetVariance() { return 0.0; }

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
      kf_(0.1, 0.01),
      kf_enabled_(true),
      yaw_filter_gain_kf_(0.3),
      filtered_yaw_(0.0),
      yaw_filter_initialized_(false),
      kf_timeout_(1.0),
      trigger_received_(false),
      offset_history_max_size_(10) {

    dog_pos_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_processed", 10);
    aoa_dog_pos_debug_pub_ = nh_.advertise<nav_msgs::Odometry>("/dog_pos_aoa_debug", 10);

    target_sub_ = nh_.subscribe("/target_ekf_odom", 10, &DogPosProcessor::targetCallback, this);

    timer_ = nh_.createTimer(ros::Duration(0.05), &DogPosProcessor::processCallback, this);
    status_timer_ = nh_.createTimer(ros::Duration(0.1), &DogPosProcessor::statusCheckCallback, this);

    ROS_INFO("Dog Position Processor (vis_base): subscribe /target_ekf_odom -> LKF -> /dog_pos_processed");
}

void DogPosProcessor::targetCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    target_ekf_last_time_ = msg->header.stamp;

    target_dog_yaw_ = msg->pose.pose.orientation.w;
    target_dog_pitch_ = msg->pose.pose.orientation.x;
    target_dog_roll_ = msg->pose.pose.orientation.y;

    target_dog_pos_.x() = msg->pose.pose.position.x;
    target_dog_pos_.y() = msg->pose.pose.position.y;
    target_dog_pos_.z() = msg->pose.pose.position.z;

    raw_dog_vel_.x() = msg->twist.twist.linear.x;
    raw_dog_vel_.y() = msg->twist.twist.linear.y;
    raw_dog_vel_.z() = msg->twist.twist.linear.z;

    target_count_++;
}

void DogPosProcessor::rawDogPosCallback(const nav_msgs::Odometry::ConstPtr& msg) { (void)msg; }
void DogPosProcessor::vinsCallback(const nav_msgs::Odometry::ConstPtr& msg) { (void)msg; }
void DogPosProcessor::aoaCallback(const nav_msgs::Odometry::ConstPtr& msg) { (void)msg; }
void DogPosProcessor::flowCallback(const nav_msgs::Odometry::ConstPtr& msg) { (void)msg; }
void DogPosProcessor::takeoffCallback(const quadrotor_msgs::TakeoffLand::ConstPtr& msg) { (void)msg; }
void DogPosProcessor::yawDiffCallback(const std_msgs::Float64::ConstPtr& msg) { (void)msg; }
void DogPosProcessor::testResetCallback(const std_msgs::Bool::ConstPtr& msg) { (void)msg; }
void DogPosProcessor::triggerCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) { (void)msg; }

void DogPosProcessor::updateDogYawRate(double delta_yaw) { (void)delta_yaw; }
void DogPosProcessor::updateDogAcc(const Eigen::Vector3d& delta_vel, double dt) { (void)delta_vel; (void)dt; }

void DogPosProcessor::statusCheckCallback(const ros::TimerEvent& event) {
    (void)event;
    if (target_count_ != last_target_count_) {
        last_target_timer_++;
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
}

void DogPosProcessor::processCallback(const ros::TimerEvent& event) {
    (void)event;
    if (target_count_ == 0) return;

    double dt = 0.05;
    if (last_kf_time_.isZero()) {
        last_kf_time_ = ros::Time::now();
    } else {
        dt = (ros::Time::now() - last_kf_time_).toSec();
        if (dt <= 0) dt = 0.05;
        if (dt > kf_timeout_) {
            kf_.reset();
            yaw_filter_initialized_ = false;
            filtered_yaw_ = 0.0;
            dt = 0.05;
        }
    }

    if (kf_enabled_) {
        auto filtered = kf_.filter(target_dog_pos_, raw_dog_vel_, dt);
        final_dog_pos_ = filtered.first;
        final_dog_vel_ = filtered.second;

        if (!yaw_filter_initialized_) {
            filtered_yaw_ = target_dog_yaw_;
            yaw_filter_initialized_ = true;
        } else {
            double yaw_diff = normalizeAngle(target_dog_yaw_ - filtered_yaw_);
            filtered_yaw_ = normalizeAngle(filtered_yaw_ + yaw_filter_gain_kf_ * yaw_diff);
        }
        final_dog_yaw_ = filtered_yaw_;
    } else {
        final_dog_pos_ = target_dog_pos_;
        final_dog_vel_ = raw_dog_vel_;
        final_dog_yaw_ = target_dog_yaw_;
    }

    last_kf_time_ = ros::Time::now();
    publishProcessedDogPos();
}

void DogPosProcessor::publishProcessedDogPos() {
    nav_msgs::Odometry msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "world";

    msg.pose.pose.position.x = final_dog_pos_.x();
    msg.pose.pose.position.y = final_dog_pos_.y();
    msg.pose.pose.position.z = final_dog_pos_.z();

    // vis_base: 始终认为 pos/yaw 都已就绪
    msg.pose.pose.orientation.w = 1.0;  // pos_ready = true
    msg.pose.pose.orientation.x = 1.0;  // yaw_ready = true
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

void DogPosProcessor::spin() {
    ros::spin();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "dog_pos_processor");
    DogPosProcessor processor;
    processor.spin();
    return 0;
}
