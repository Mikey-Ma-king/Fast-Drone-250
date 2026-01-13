#pragma once

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/TakeoffLand.h>
#include <Eigen/Dense>
#include <cmath>

class KalmanFilter {
public:
    KalmanFilter(double process_noise = 0.01, double measurement_noise = 0.02);
    void predict(double dt);
    void update(const Eigen::VectorXd& measurement);
    std::pair<Eigen::Vector3d, Eigen::Vector3d> filter(const Eigen::Vector3d& position, 
                                                        const Eigen::Vector3d& velocity, 
                                                        double dt);
    void reset();
    
private:
    Eigen::VectorXd state_;  // [x, y, z, vx, vy, vz]
    Eigen::MatrixXd covariance_;
    Eigen::MatrixXd Q_;  // 过程噪声
    Eigen::MatrixXd R_;  // 测量噪声
    Eigen::MatrixXd F_base_;
    Eigen::MatrixXd H_;  // 观测矩阵
    bool initialized_;
    ros::Time last_time_;
};

class DogPosProcessor {
public:
    DogPosProcessor();
    void spin();
    
private:
    // 工具函数
    double normalizeAngle(double angle);
    
    // 回调函数
    void rawDogPosCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void targetCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void vinsCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void aoaCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void flowCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void takeoffCallback(const quadrotor_msgs::TakeoffLand::ConstPtr& msg);
    void yawDiffCallback(const std_msgs::Float64::ConstPtr& msg);
    void testResetCallback(const std_msgs::Bool::ConstPtr& msg);
    void triggerCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void statusCheckCallback(const ros::TimerEvent& event);
    void processCallback(const ros::TimerEvent& event);
    
    // 内部函数
    void updateDogYawRate(double delta_yaw);
    void updateDogAcc(const Eigen::Vector3d& delta_vel);
    void publishProcessedDogPos();
    
    // ROS节点
    ros::NodeHandle nh_;
    
    // 发布者
    ros::Publisher dog_pos_pub_;
    ros::Publisher aoa_dog_pos_debug_pub_;  // Debug发布者 - 发布AOA计算得到的dog位置
    
    // 订阅者
    ros::Subscriber raw_dog_pos_sub_;
    ros::Subscriber target_sub_;
    ros::Subscriber vins_sub_;
    ros::Subscriber aoa_sub_;
    ros::Subscriber flow_sub_;
    ros::Subscriber takeoff_sub_;
    ros::Subscriber yaw_diff_preset_sub_;
    ros::Subscriber test_reset_sub_;
    ros::Subscriber trigger_sub_;
    
    // 定时器
    ros::Timer timer_;
    ros::Timer status_timer_;
    
    // 参数
    double yaw_filter_gain_;
    double yaw_stable_threshold_;
    double yaw_exceed_threshold_;
    int yaw_exceed_max_count_;
    double pos_stable_threshold_;
    double pos_exceed_threshold_;
    int pos_exceed_max_count_;
    double pos_filter_gain_;
    double aoa_pos_filter_gain_;  // AOA位置滤波增益（比pos_filter_gain小）
    double aoa_pos_step_limit_;   // AOA位置偏移单次迭代上限（米）
    double camera_offset_;
    double aoa_min_distance_;
    double flow_height_bias_;
    
    // 状态变量
    double yaw_offset_;
    Eigen::Vector3d pos_offset_;
    bool precise_yaw_offset_ready_;
    bool precise_pos_offset_ready_;
    int yaw_exceed_timer_;
    int pos_exceed_timer_;
    double* saved_yaw_diff_;  // 使用指针，nullptr表示未设置
    bool initialized_;
    bool simulate_mode_;  // 仿真模式：直接输出target_ekf作为dog_pos_processed
    
    // 原始数据
    nav_msgs::Odometry::ConstPtr raw_dog_pos_;
    Eigen::Vector3d raw_dog_vel_;
    double raw_dog_yaw_;
    bool raw_dog_pos_received_;
    unsigned int raw_dog_pos_count_;
    unsigned int last_raw_dog_pos_count_;
    int last_dog_pos_timer_;
    
    // VINS数据
    double vins_yaw_;
    Eigen::Matrix3d R_wb_;
    Eigen::Vector3d vins_pos_;
    bool vins_received_;
    unsigned int vins_count_;
    unsigned int last_vins_count_;
    int last_vins_timer_;
    
    // 目标数据
    double target_dog_yaw_;
    Eigen::Vector3d target_dog_pos_;
    bool target_receive_;
    unsigned int target_count_;
    unsigned int last_target_count_;
    int last_target_timer_;
    int last_target_loss_timer_;
    unsigned int last_target_loss_count_;
    // AOA数据
    bool aoa_received_;
    unsigned int aoa_count_;
    unsigned int last_aoa_count_;
    int last_aoa_timer_;
    // double aoa_anchor1_distance_;
    // double aoa_anchor1_angle_;
    // double aoa_anchor2_distance_;
    // double aoa_anchor2_angle_;
    double aoa_distance_;   // 单anchor距离（已做高度修正）
    double aoa_angle_;      // 相对无人机朝向的角度
    
    // 光流数据
    double flow_z_;
    
    // 最终输出
    Eigen::Vector3d final_dog_pos_;
    Eigen::Vector3d final_dog_vel_;
    double final_dog_yaw_;
    
    // 速度相关
    bool dog_vel_initialized_;
    double final_dog_yaw_rate_;
    double last_dog_yaw_time_;
    double yaw_rate_filter_gain_;
    bool dog_yaw_rate_initialized_;
    Eigen::Vector3d last_dog_vel_;  // 上一次滤波后的速度，用于计算加速度
    
    // 加速度相关
    Eigen::Vector3d final_dog_acc_;
    double last_dog_vel_time_;
    double acc_filter_gain_;
    bool dog_acc_initialized_;
    
    // 卡尔曼滤波
    KalmanFilter kf_;
    bool kf_enabled_;
    ros::Time last_kf_time_;
    double yaw_filter_gain_kf_;
    double filtered_yaw_;
    bool yaw_filter_initialized_;
    double kf_timeout_;
    
    // Trigger状态
    bool trigger_received_;
};

