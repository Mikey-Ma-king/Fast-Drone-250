#pragma once

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/TakeoffLand.h>
#include <Eigen/Dense>
#include <cmath>
#include <deque>

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
    double calculateYawOffsetVariance();  // 计算yaw_offset历史记录的方差
    double calculatePosOffsetVariance();   // 计算pos_offset历史记录的方差（协方差矩阵的迹）
    
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
    void updateDogPitchRate(double delta_pitch, double dt = -1.0);  // 更新pitch角速度，dt<=0时自动计算
    void updateDogRollRate(double delta_roll);    // 更新roll角速度
    void updateDogAcc(const Eigen::Vector3d& delta_vel, double dt = -1.0);  // 更新加速度，dt<=0时自动计算
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
    
    // 感知置信度参数
    double perception_confidence_time_threshold_;  // 时间阈值（秒），超过此时间置信度衰减
    double perception_confidence_yaw_rate_threshold_;   // yaw角速度阈值（弧度/秒）
    double perception_confidence_pitch_rate_threshold_; // pitch角速度阈值（弧度/秒）
    double perception_confidence_roll_rate_threshold_; // roll角速度阈值（弧度/秒）
    double perception_confidence_yaw_angle_threshold_;   // yaw角度阈值（弧度）
    double perception_confidence_pitch_angle_threshold_; // pitch角度阈值（弧度）
    double perception_confidence_roll_angle_threshold_; // roll角度阈值（弧度）
    double perception_confidence_acc_threshold_;   // 加速度阈值（m/s^2）
    double perception_confidence_std_threshold_;   // 标准差阈值
    double perception_confidence_filter_gain_;     // 感知置信度滤波增益
    double filtered_perception_confidence_;        // 滤波后的感知置信度
    double flow_height_bias_;
    
    // 状态变量
    double yaw_offset_;
    Eigen::Vector3d pos_offset_;
    bool precise_yaw_offset_ready_;
    bool precise_pos_offset_ready_;
    int yaw_exceed_timer_;
    int pos_exceed_timer_;
    
    // 历史记录用于计算方差
    std::deque<double> yaw_offset_history_;
    std::deque<Eigen::Vector3d> pos_offset_history_;
    int offset_history_max_size_;  // 历史记录最大长度
    double* saved_yaw_diff_;  // 使用指针，nullptr表示未设置
    bool initialized_;
    bool simulate_mode_;  // 仿真模式：直接输出target_ekf作为dog_pos_processed
    
    // 原始数据
    nav_msgs::Odometry::ConstPtr raw_dog_pos_;
    Eigen::Vector3d raw_dog_vel_;
    double raw_dog_yaw_;
    double raw_dog_pitch_;  // pitch角度（弧度）
    double raw_dog_roll_;   // roll角度（弧度）
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
    double target_dog_pitch_;  // pitch角度（弧度）
    double target_dog_roll_;   // roll角度（弧度）
    Eigen::Vector3d target_dog_pos_;
    bool target_receive_;
    unsigned int target_count_;
    unsigned int last_target_count_;
    int last_target_timer_;
    int last_target_loss_timer_;
    unsigned int last_target_loss_count_;
    ros::Time target_ekf_last_time_;  // target_ekf最新包的时间
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
    double final_dog_pitch_rate_;   // pitch角速度（弧度/秒）
    double final_dog_roll_rate_;     // roll角速度（弧度/秒）
    double last_dog_yaw_time_;       // yaw角速度上一次的时间
    double last_dog_pitch_time_;    // pitch角速度上一次的时间
    double last_dog_roll_time_;     // roll角速度上一次的时间
    double yaw_rate_filter_gain_;
    bool dog_yaw_rate_initialized_;
    bool dog_pitch_rate_initialized_; // pitch角速度初始化标志
    bool dog_roll_rate_initialized_;  // roll角速度初始化标志
    Eigen::Vector3d last_dog_vel_;  // 上一次滤波后的速度，用于计算加速度
    // 用于在publishProcessedDogPos中计算角速度和加速度的上一次值
    double last_final_dog_yaw_;      // 上一次的final_dog_yaw_，用于计算yaw角速度
    double last_final_dog_pitch_;   // 上一次的final_dog_pitch_，用于计算pitch角速度
    double last_final_dog_roll_;    // 上一次的final_dog_roll_，用于计算roll角速度
    Eigen::Vector3d last_final_dog_vel_;  // 上一次的final_dog_vel_，用于计算加速度
    ros::Time last_publish_time_;   // 上一次发布的时间，用于计算时间差
    
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

