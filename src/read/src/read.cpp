/**
 * ===========================================================================
 * read.cpp — 视觉目标检测与状态估计模块
 * ===========================================================================
 *
 * 【在整个系统中的角色】
 *   本模块是感知管道的前端，负责通过单目 USB 摄像头实时检测贴在目标(狗/移动平台)
 *   上的 ArUco 二维码标记，解算目标在世界坐标系下的 3D 位姿(位置 + yaw/pitch/roll)，
 *   并通过 Bezier 曲线拟合实现轨迹预测以补偿视觉延迟，最终发布 /target_ekf_odom
 *   供下游节点(dog_pos_processor、MPC、traj_server)消费。
 *
 * 【核心管道】
 *   USB 摄像头图像 → ArUco 标记检测 → PnP 位姿估计(标记坐标系)
 *     → M_tag2camera 变换(标记→相机坐标系)
 *     → M_camera2drone 变换(相机→无人机坐标系)
 *     → R1/T1 变换(无人机坐标系→世界坐标系,来自 VINS 延迟补偿)
 *     → 多标记加权融合(位置+yaw/pitch/roll)
 *     → Bezier 曲线预测(补偿延迟+外推速度)
 *     → 发布 /target_ekf_odom
 *
 * 【关键坐标系定义】
 *   相机坐标系:  x=右,  y=下,  z=前 (OpenCV 标准)
 *   无人机坐标系: x=前,  y=左,  z=上 (ROS 机体坐标系)
 *   标记坐标系:  x=右,  y=前, z=上 (ArUco 标记,红色=x,绿色=y,蓝色=z)
 *   世界坐标系:  VINS 里程计的全局参考系
 *
 * 【依赖项】
 *   - OpenCV + cv::aruco (二维码检测)
 *   - Eigen (线性代数)
 *   - bezier_predict.h (Bezier 轨迹预测)
 *   - VINS-Fusion (/vins_fusion/imu_propagate 提供无人机位姿)
 */

#define _USE_MATH_DEFINES
#include <read.h>

#include <geometry_msgs/PoseStamped.h>
#include "quadrotor_msgs/PositionCommand.h"
#include "quadrotor_msgs/TakeoffLand.h"
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>

#include <Eigen/Core>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <Eigen/Dense>

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <tf/tf.h>
#include <vector>
#include <deque>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <boost/thread.hpp>
#include <stdio.h>
#include <fstream>
#include <signal.h>
#include <chrono>
#include <mutex>
#include <tf/transform_datatypes.h>
#include <dlfcn.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <bezier_predict.h>
#include <map>
using namespace cv;

namespace image {

// ============================================================================
// 一、全局状态变量 — 线程间共享的感知和定位数据
// ============================================================================
// 注意: 这些全局变量在 nodelet 的多线程环境下被读写,部分使用 mutex 保护

int flag = 1;
Eigen::Vector3d glo_pos{0.0, 0.0, 0.0};     // 目标在世界系中的当前位置(视觉检测结果)
Eigen::Vector3d glo_vel{0.0, 0.0, 0.0};     // 目标在世界系中的速度(速度拟合结果)
Eigen::Vector3d averge_v{0.0, 0.0, 0.0};    // 速度的平滑平均值
std::vector<double> pos_copy;                 // 位置历史副本,用于降落触发判断
std::mutex global_mutex;                      // 保护全局变量的互斥锁
int flag1 = 0;                                // 有效检测帧计数器
Eigen::Vector3d averge_vel{0.0, 0.0, 0.0};   // 位置的平滑平均值
cv::Mat frame;                                // 当前相机图像帧(由 imageCallback 更新)
double last_yaw = 0;                          // 上一帧的 yaw 角度(弧度)
double fin_yaw = 0;                           // 滤波后的最终 yaw 角度(弧度)
double fin_pitch = 0;                         // 滤波后的最终 pitch 角度(弧度)
double fin_roll = 0;                          // 滤波后的最终 roll 角度(弧度)
int triger2 = 0;                              // 二次触发标志
int vins_triger = 0;                          // VINS 有效标志,用于控制 odom 发布
int gos_triger = 0;                           // 狗定位触发标志
int map_triger = 16;                          // 标定计数器,用于控制狗-视觉对齐的采样时机
Eigen::Vector3d avg{0.0, 0.0, 0.0};          // 位置滑动窗口均值

// VINS/SVO 融合位姿
Eigen::Vector3d svo_p{0.0, 0.0, 0.0};        // 无人机在世界系的位置(来自 VINS/SVO)
Eigen::Vector3d svo_v{0.0, 0.0, 0.0};        // 无人机在世界系的速度(来自 VINS/SVO)
Eigen::Vector3d vins_p{0.0, 0.0, 0.0};       // VINS 输出的无人机位置
Eigen::Vector3d last_svo_p{0.0, 0.0, 0.0};   // 上一帧 SVO 位置(用于差分计算速度)
Eigen::Vector3d last_svo_v{0.0, 0.0, 0.0};   // 上一帧 SVO 速度
Eigen::Quaterniond svo_q;                      // 无人机姿态四元数

// Bezier 预测结果缓冲区
// 每条轨迹包含 _PREDICT_SEG 个 6 维向量 [px, py, pz, vx, vy, vz]
std::vector<double> result_x(6,0.0);           // 发布给 target_ekf_odom 的 x 序列
std::vector<double> result_y(6,0.0);           // 发布给 target_ekf_odom 的 y 序列
std::vector<double> result_z(6,0.0);           // 发布给 target_ekf_odom 的 z 序列
std::vector<double> result_vx(6,0.0);          // 发布给 target_ekf_odom 的 vx 序列
std::vector<double> result_vy(6,0.0);          // 发布给 target_ekf_odom 的 vy 序列
std::vector<double> result_vz(6,0.0);          // 发布给 target_ekf_odom 的 vz 序列
std::vector<Eigen::Matrix<double, 6, 1>> g_predictedTrajectory(
    _PREDICT_SEG, Eigen::Matrix<double, 6, 1>::Zero()
);
int predict_index = 0;                        // 当前使用的预测点索引
static bool svo_initialize = false;           // SVO 预测器是否已初始化

// 狗(Gos)定位相关 — 机器人狗自身报告的位置,用于与视觉检测做坐标系对齐
Eigen::Vector3d gos_v{0.0, 0.0, 0.0};        // 狗报告的速度
Eigen::Vector3d gos_vins_v{0.0, 0.0, 0.0};   // 狗速度转到 VINS 坐标系
double gos_yaw;                                // 狗报告的 yaw
double gos_yaw0 = 0;                           // 标定时刻狗报告的 yaw
double gos_vins_yaw0 = 0;                      // 标定时刻视觉检测的 yaw
double gos_vins_yaw = 0;                       // 狗 yaw 转到 VINS 坐标系
double d_yaw = 0;                              // 标定 yaw 差 = gos_vins_yaw0 - gos_yaw0
Eigen::Vector3d gos_pos{0.0, 0.0, 0.0};       // 狗报告的自身位置
Eigen::Vector3d vins_p0{0.0, 0.0, 0.0};       // 标定时刻 VINS 位置
Eigen::Vector3d gos_p0{0.0, 0.0, 0.0};        // 标定时刻狗报告位置
Eigen::Vector3d gos_vins_p0{0.0, 0.0, 0.0};   // 标定时刻狗位置转到 VINS 坐标系
Eigen::Vector3d gos_vins_pos{0.0, 0.0, 0.0};  // 当前狗位置转到 VINS 坐标系
std::vector<double> yaw_list;                  // yaw 历史记录
std::vector<std::vector<Eigen::Vector3d>> reflect_list;  // 标定反射列表,每项 {dog_p0, dog_vins_p0, r_yaw}
int land_triger = 0;                           // 降落触发标志

ros::Time start_gos_time;                      // 狗数据开始接收的时间
ros::Time target_start_time = ros::Time::now(); // 目标检测的计时起点

std::vector<Eigen::Vector4d> target_detect_lis; // 目标检测历史 [x,y,z,t],用于 Bezier 拟合
Bezierpredict predict;                          // 狗位置预测器(将狗通信数据转到 VINS 系)

ros::Time predict_time;

double last_gos_vins_vx = -2.0;                // 上一帧狗速度 x(VINS 系), -2.0 表示未初始化
double last_gos_vins_vy = -2.0;                // 上一帧狗速度 y(VINS 系)

double AOA_distance;                            // UWB AOA 传感器测得的距离(m)
double AOA_angle;                               // UWB AOA 传感器测得的角度(弧度)

double flow_z = 0.0;                           // 光流高度(m),用于辅助高度估计


// ============================================================================
// 二、多测量卡尔曼滤波器 (MultiKalmanFilter)
// ============================================================================
// 同时融合 3D 位置测量和 1D 距离测量(AOA 距离),
// 通过两次 Kalman 更新步骤(先位置后距离)来改善状态估计

/**
 * 构造函数
 * @param pos_process_var  位置过程噪声方差
 * @param pos_meas_var     位置测量噪声方差
 * @param dist_meas_var    距离测量噪声方差(AOA)
 */
MultiKalmanFilter::MultiKalmanFilter(double pos_process_var, double pos_meas_var, double dist_meas_var)
        : process_noise_(Eigen::Matrix3d::Identity() * pos_process_var),
          pos_meas_noise_(Eigen::Matrix3d::Identity() * pos_meas_var),
          dist_meas_noise_(dist_meas_var),
          is_initialized_(false) {}

/**
 * 滤波主函数 — 两步更新策略
 * @param pos_meas       3D 位置测量值(来自视觉检测)
 * @param distance_meas  1D 距离测量值(来自 AOA UWB 传感器)
 * @return               滤波后的 3D 位置估计
 *
 * 步骤:
 *   1. 预测: x̂ = F·x, P = F·P·Fᵀ + Q  (恒定位置模型,F=I)
 *   2. 位置更新: 用 3D 位置测量做标准 Kalman 更新
 *   3. 距离更新: 将距离测量作为非线性约束,用 EKF 雅可比做更新
 *      - 观测函数 h(x) = ||x|| (状态的欧氏距离)
 *      - 雅可比 H = xᵀ/||x|| (方向导数)
 */
Eigen::Vector3d MultiKalmanFilter::filter(const Eigen::Vector3d& pos_meas, double distance_meas) {
    if (!is_initialized_) {
        state_ = pos_meas;
        covariance_ = Eigen::Matrix3d::Identity();
        is_initialized_ = true;
        return state_;
    }

    // 预测步骤 — 恒定位置模型(状态不变,只增加过程噪声)
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();  // 状态转移矩阵 = 单位阵(恒定位置)
    state_ = F * state_;
    covariance_ = F * covariance_ * F.transpose() + process_noise_;

    // 更新步骤 1: 位置测量更新(线性观测)
    Eigen::Matrix3d H_pos = Eigen::Matrix3d::Identity(); // 位置观测矩阵,直接观测全部三维
    Eigen::Vector3d y_pos = pos_meas - H_pos * state_;   // 位置残差
    Eigen::Matrix3d S_pos = H_pos * covariance_ * H_pos.transpose() + pos_meas_noise_;
    Eigen::Matrix3d K_pos = covariance_ * H_pos.transpose() * S_pos.inverse();
    state_ = state_ + K_pos * y_pos;
    covariance_ = (Eigen::Matrix3d::Identity() - K_pos * H_pos) * covariance_;

    // 更新步骤 2: 距离测量更新(EKF 雅可比,因为距离 = sqrt(x²+y²+z²) 是非线性函数)
    double pred_distance = state_.norm();
    double y_dist = distance_meas - pred_distance;  // 距离残差

    // 计算雅可比矩阵: ∂(||x||)/∂x = xᵀ/||x||
    Eigen::Matrix<double, 1, 3> H_dist;
    if (pred_distance > 1e-6) {
        H_dist << state_.x()/pred_distance,
                  state_.y()/pred_distance,
                  state_.z()/pred_distance;
    } else {
        H_dist.setZero();
    }

    // 标准 EKF 距离更新
    double S_dist = (H_dist * covariance_ * H_dist.transpose())(0) + dist_meas_noise_;
    Eigen::Matrix<double, 3, 1> K_dist = covariance_ * H_dist.transpose() / S_dist;

    state_ = state_ + K_dist * y_dist;
    covariance_ = (Eigen::Matrix3d::Identity() - K_dist * H_dist) * covariance_;

    return state_;
}


// ============================================================================
// 三、一维标量卡尔曼滤波器 (KalmanFilter)
// ============================================================================
// 用于对单个维度的位置坐标(x/y/z 分别滤波)做一阶滤波

KalmanFilter::KalmanFilter(double process_variance, double measurement_variance)
        : process_variance_(process_variance),
          measurement_variance_(measurement_variance),
          is_initialized_(false),
          state_(0.0),
          covariance_(1.0) {}

double KalmanFilter::filter(double measurement) {
    if (!is_initialized_) {
        state_ = measurement;
        covariance_ = 1.0;
        is_initialized_ = true;
        return state_;
    }

    // 预测 — 恒定值模型
    double predicted_state = state_;
    double predicted_covariance = covariance_ + process_variance_;

    // 更新 — 标量 Kalman
    double kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance_);
    state_ = predicted_state + kalman_gain * (measurement - predicted_state);
    covariance_ = (1 - kalman_gain) * predicted_covariance;

    return state_;
}


// ============================================================================
// 四、中值滤波器 (MedianFilter1)
// ============================================================================
// 滑动窗口中值滤波,用于滤除角度/位置的偶发离群值

MedianFilter1::MedianFilter1(size_t size) : size(size) {}

double MedianFilter1::update(double new_value){
    // 维护固定大小的滑动窗口
    if (window.size() >= size) {
        window.pop_front();
    }
    window.push_back(new_value);

    // 排序后取中值(O(n log n))
    std::vector<double> sorted_window(window.begin(), window.end());
    std::sort(sorted_window.begin(), sorted_window.end());
    size_t mid = sorted_window.size() / 2;

    if (sorted_window.size() % 2 == 0) {
        return (sorted_window[mid - 1] + sorted_window[mid]) / 2.0;  // 偶数: 中间两个均值
    } else {
        return sorted_window[mid];  // 奇数: 中间值
    }
}


// ============================================================================
// 五、里程计发布函数
// ============================================================================

/**
 * publishOdometry — 发布 VINS/SVO 融合后的无人机里程计
 *
 * 发布到 /vins_fusion/imu_propagate 的 Odometry 消息,
 * 供整个系统中的其他节点(VINS、traj_server、dog_pos_processor)使用。
 * 位置和速度从 Bezier 预测轨迹中插值得到,姿态来自 SVO 原始四元数。
 */
void publishOdometry(const ros::TimerEvent&, ros::Publisher &odom_pub) {
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.header.frame_id = "world";
    if(svo_initialize){
        // 速度取预测轨迹和原始 SVO 速度的加权平均(低通滤波效果)
        if(predict_index>=1){
            svo_v.x() = 0.5*g_predictedTrajectory[predict_index](3) + 0.5*g_predictedTrajectory[0](3);
            svo_v.y() = 0.5*g_predictedTrajectory[predict_index](4) + 0.5*g_predictedTrajectory[0](4);
            svo_v.z() = 0.5*g_predictedTrajectory[predict_index](5) + 0.5*g_predictedTrajectory[0](5);
        }
        else{
            svo_v.x() = 0.3*g_predictedTrajectory[predict_index](3) + 0.7*g_predictedTrajectory[predict_index](3);
            svo_v.y() = 0.3*g_predictedTrajectory[predict_index](4) + 0.7*g_predictedTrajectory[predict_index](4);
            svo_v.z() = 0.3*g_predictedTrajectory[predict_index](5) + 0.7*g_predictedTrajectory[predict_index](5);
        }

        // 位置 — 使用 result_* 序列(由 predictPositions 做匀速外推)
        odom_msg.pose.pose.position.x = result_x[predict_index];
        odom_msg.pose.pose.position.y = result_y[predict_index];
        odom_msg.pose.pose.position.z = result_z[predict_index];

        // 姿态 — 直接使用 SVO 原始四元数
        odom_msg.pose.pose.orientation.x = svo_q.x();
        odom_msg.pose.pose.orientation.y = svo_q.y();
        odom_msg.pose.pose.orientation.z = svo_q.z();
        odom_msg.pose.pose.orientation.w = svo_q.w();

        // 速度安全门: 速度 > 4m/s 时停止发布(防止 VINS 漂移导致控制发散)
        if (svo_v.norm() < 4 || vins_triger == 0){
            odom_msg.twist.twist.linear.x = svo_v.x();
            odom_msg.twist.twist.linear.y = svo_v.y();
            odom_msg.twist.twist.linear.z = svo_v.z();
            odom_pub.publish(odom_msg);
        }
        else{
            if (vins_triger == 0)
                std::cout<<"停止发布vins"<<std::endl;
            vins_triger = 0;
        }

        // 逐步前进预测索引(前几帧使用最近的预测点,避免初始化瞬变)
        if(predict_index<=3)
            predict_index+=1;
    }
}


// ============================================================================
// 六、辅助工具函数
// ============================================================================

/**
 * correctYaw — yaw 角修正
 * 将姿态矩阵的 yaw 角翻转 180 度(±π),用于处理 ArUco 标记方向歧义
 */
Eigen::Matrix3d correctYaw(const Eigen::Matrix3d& pose) {
    Eigen::Vector3d eulerAngles = pose.eulerAngles(2, 1, 0); // Z-Y-X 顺序

    double yaw = eulerAngles[0];    // 偏航角
    double pitch = eulerAngles[1];  // 俯仰角
    double roll = eulerAngles[2];   // 滚转角

    // yaw 归一化到 [-π, π]
    if (yaw > M_PI) {
        yaw -= 2 * M_PI;
    } else if (yaw < -M_PI) {
        yaw += 2 * M_PI;
    }

    // 翻转±180度: ArUco 标记的旋转有 ±π 歧义性
    if (yaw > 0) {
        yaw -=  M_PI;
    } else if (yaw < 0) {
        yaw += M_PI;
    }

    // 用修正后的 yaw 重建旋转矩阵
    Eigen::Matrix3d correctedPose = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                                    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix() *
                                    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
    return correctedPose;
}

/**
 * gos_pos_pub — 狗位姿发布(坐标系对齐 + Bezier 预测)
 *
 * 将狗通信模块报告的位置(gos_pos)通过标定的 offset 转换到 VINS 世界坐标系,
 * 再通过 Bezier 曲线平滑预测。目前该函数的大部分发布逻辑被注释,
 * 仅在 gos_triger=1 时更新内部状态。
 *
 * 坐标系变换: gos(狗系) → VINS(世界系)
 *   - 对每个反射记录: dog_vins_pos = R(-r_yaw)·(gos_pos - dog_p0) + dog_vins_p0
 *   - 主变换:        gos_vins_pos = R(-r_yaw_main)·(gos_pos - gos_p0) + gos_vins_p0
 */
void gos_pos_pub(const ros::TimerEvent&, ros::Publisher &odom_pub) {
    if(gos_triger == 1){
        nav_msgs::Odometry gos_msg;
        gos_msg.header.stamp = ros::Time::now();
        gos_msg.header.frame_id = "world";

        // 遍历所有标定反射记录,对每个记录计算狗在 VINS 系下的位置
        std::vector<Eigen::Vector3d> gos_p_list;
        std::vector<Eigen::Vector3d> gos_v_list;
        std::vector<double> gos_yaw_list;
        for(int i = 0;i<reflect_list.size();i++){
            std::vector<Eigen::Vector3d> vec = reflect_list[i];
            Eigen::Vector3d dog_p0 = vec[0];       // 标定时狗报告位置
            Eigen::Vector3d dog_vins_p0 = vec[1];  // 标定时 VINS 系下狗位置
            double r_yaw = vec[2].x();             // 标定 yaw 差
            Eigen::Vector3d r_p = gos_pos - dog_p0;
            // 用标定 yaw 差旋转相对位置向量
            double cos_theta = cos(r_yaw);
            double sin_theta = sin(r_yaw);
            double rotated_x = cos_theta * r_p.x() - sin_theta * r_p.y();
            double rotated_y = sin_theta * r_p.x() + cos_theta * r_p.y();
            r_p.x() = rotated_x;
            r_p.y() = rotated_y;
            Eigen::Vector3d dog_vins_pos = r_p + dog_vins_p0;
            double dog_vins_yaw = gos_yaw + r_yaw;
            // 狗速度也做同样的 yaw 旋转
            double rotated_vx = cos_theta * gos_v.x() - sin_theta * gos_v.y();
            double rotated_vy = sin_theta * gos_v.x() + cos_theta * gos_v.y();
            Eigen::Vector3d dog_vins_v = gos_v;
            dog_vins_v.x() = rotated_vx;
            dog_vins_v.y() = rotated_vy;
            gos_p_list.push_back(dog_vins_pos);
            gos_v_list.push_back(dog_vins_v);
            gos_yaw_list.push_back(dog_vins_yaw);
        }

        // 主变换: 用主要的标定参数转换狗位置
        Eigen::Vector3d r_p = gos_pos - gos_p0;
        double r_yaw = gos_vins_yaw0 - gos_yaw0;
        double cos_theta = cos(r_yaw);
        double sin_theta = sin(r_yaw);

        double rotated_x = cos_theta * r_p.x() - sin_theta * r_p.y();
        double rotated_y = sin_theta * r_p.x() + cos_theta * r_p.y();
        r_p.x() = rotated_x;
        r_p.y() = rotated_y;
        gos_vins_pos = r_p + gos_vins_p0;

        gos_vins_yaw = gos_yaw - gos_yaw0 + gos_vins_yaw0;

        double rotated_vx = cos_theta * gos_v.x() - sin_theta * gos_v.y();
        double rotated_vy = sin_theta * gos_v.x() + cos_theta * gos_v.y();
        gos_vins_v = gos_v;
        gos_vins_v.x() = rotated_vx;
        gos_vins_v.y() = rotated_vy;

        // Bezier 曲线预测 — 用历史 VINS 系下的狗位置做平滑和外推
        static bool target_initialize = false;
        static std::vector<Eigen::Matrix<double,6,1>> predict_target_list;
        Eigen::Vector4d state(gos_vins_pos.x(),gos_vins_pos.y(),gos_vins_pos.z(),ros::Time::now().toSec());
        if(!target_initialize){
            target_detect_lis.push_back(state);
            if(target_detect_lis.size()>=_MAX_SEG){
                target_initialize=1;
            }
        }
        else{
            target_detect_lis.erase(target_detect_lis.begin());
            target_detect_lis.push_back(state);

            int bezier_flag = predict.TrackingGeneration(5,5,target_detect_lis);
            if(bezier_flag==0){
                predict_target_list = predict.getStateListFromBezier(_PREDICT_SEG);
                // 速度使用预测值和上一帧值的加权平均(低通滤波)
                if(last_gos_vins_vx == -2.0)
                {
                    last_gos_vins_vx  = predict_target_list[0](3);
                    last_gos_vins_vy  = predict_target_list[0](4);
                }
                gos_msg.twist.twist.linear.x = 0.5*predict_target_list[0](3) + 0.5 * last_gos_vins_vx;
                gos_msg.twist.twist.linear.y = 0.5*predict_target_list[0](4) + 0.5 * last_gos_vins_vy;
                gos_msg.twist.twist.linear.z = 0;
                last_gos_vins_vx = predict_target_list[0](3);
                last_gos_vins_vy = predict_target_list[0](4);
            }
        }

        gos_msg.pose.pose.position.x = round_to_decimal_places(gos_vins_pos.x(), 2);
        gos_msg.pose.pose.position.y = round_to_decimal_places(gos_vins_pos.y(), 2);
        // 高度安全门: 狗报告高度与 VINS 高度差超过 3m 则使用 VINS 高度
        if(fabs(gos_vins_pos.z() - svo_p.z()) < 3)
            gos_msg.pose.pose.position.z = round_to_decimal_places(gos_vins_pos.z(), 2)+0.05;
        else
            gos_msg.pose.pose.position.z = svo_p.z();
        gos_msg.pose.pose.orientation.w = gos_vins_yaw;
        gos_msg.pose.pose.orientation.x = 0;
        gos_msg.pose.pose.orientation.y = 0;
        gos_msg.pose.pose.orientation.z = 0;

        if( land_triger == 1)
            return;
        // 注意: 当前版本不发布 gos_msg (被注释),仅维护内部预测状态
    }
}

/**
 * predictPositions — 匀速运动位置预测
 *
 * 假设目标以恒定速度运动,根据最近两个位置计算速度,外推未来 5 个位置。
 * 用于将 60Hz 的视觉检测结果插值到 200Hz 的发布频率,平滑控制指令。
 *
 * @param pos1 上一时刻位置
 * @param pos2 当前时刻位置
 * @return     [pos2, pos2+v*dt, pos2+v*2dt, ..., pos2+v*5dt] (共 6 个点)
 */
std::vector<double> predictPositions(double pos1, double pos2) {
    const double dt_input = 1.0 / 60;    // 输入位置的时间间隔(视觉检测 ~60Hz)
    const double dt_predict = 1.0 / 200;  // 预测位置的时间间隔(发布频率 200Hz)

    double velocity = (pos2 - pos1) / dt_input;  // 匀速模型: v = Δx / Δt

    std::vector<double> predictedPositions;
    predictedPositions.push_back(pos2);  // 起点 = 当前位置
    for (int i = 1; i <= 5; ++i) {
        double timeElapsed = i * dt_predict;
        double newPos = pos2 + velocity * timeElapsed;  // x(t) = x0 + v·t
        predictedPositions.push_back(newPos);
    }
    return predictedPositions;
}


// ============================================================================
// 七、ROS 回调类 — 订阅各类传感器数据
// ============================================================================

/**
 * gos_listener — 狗通信模块数据订阅器
 *
 * 订阅 /dog_pos 话题,获取机器人狗自身报告的 GPS/UWB 位置和速度。
 * 狗报告的位置在"狗坐标系"下,需要通过标定 offset 转换到世界坐标系。
 * 每收到 2 个消息取均值 (line 442),防止单帧抖动。
 */
gos_listener::gos_listener():
    trigger_condition_met(false), T1(Eigen::Vector3d::Zero()), R1(Eigen::Matrix3d::Identity()){
    pose_sub = nh.subscribe<nav_msgs::Odometry>("/dog_pos", 3, &gos_listener::pose_cb, this);
    land_sub = nh.subscribe<geometry_msgs::PoseStamped>("/land_triger", 3, &gos_listener::pose_cb_land, this);
}

void gos_listener::pose_cb(const nav_msgs::Odometry::ConstPtr& msg) {
        start_gos_time = ros::Time::now();
        Eigen::Vector3d current_position(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        Eigen::Vector3d current_v(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
        list.push_back(current_position);

        // 累计 2 帧后取均值,减少测量噪声
        if (list.size() == 2) {
            T1 = std::accumulate(list.begin(), list.end(), Eigen::Vector3d(0,0,0),
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a + b;
            });

            if (list.size() > 0) {
                T1 /= static_cast<double>(list.size());
            } else {
                T1 = Eigen::Vector3d::Zero();
            }

            trigger_condition_met = true;
            if (T1[0] != 0 && T1[2] != 0){
                gos_pos = T1;                         // 狗在世界系或狗系下的位置
                gos_yaw = msg->pose.pose.orientation.w; // 狗报告的 yaw 角
                gos_v = current_v;}                     // 狗报告的速度

            list.clear();
        }
    }

void gos_listener::pose_cb_land(const geometry_msgs::PoseStamped::ConstPtr& msg){
    land_triger = 1;
}

/**
 * UAVStateListener1 — 无人机状态订阅器
 *
 * 同时订阅:
 *   - /vins_fusion/imu_propagate (VINS 里程计,提供世界系位姿)
 *   - /svo/pose_imu (SVO 视觉里程计,用于 Bezier 预测)
 *   - /AOA_Tag_data (UWB AOA 距离+角度)
 *   - /flow_data (光流高度)
 *
 * 核心功能: 对 VINS 位姿做 FIXED_DELAY 延迟补偿。
 * 视觉检测从拍照到处理完成有固定延迟,因此从 VINS 历史缓冲区中
 * 取出延迟对应时刻的位姿,使视觉检测结果和 VINS 位姿时间对齐。
 */
UAVStateListener1::UAVStateListener1()
        : trigger_condition_met(false),  T1(Eigen::Vector3d::Zero()), R1(Eigen::Matrix3d::Identity()),
          vins_input_count(0) {

#ifndef SIMULATE
    svo_sub = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/svo/pose_imu", 3, &UAVStateListener1::svo_callback, this);
#else
    svo_sub = nh.subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 3, &UAVStateListener1::svo_callback, this);
#endif
    pose_sub = nh.subscribe<nav_msgs::Odometry>("/vins_fusion/imu_propagate", 3, &UAVStateListener1::pose_cb, this);
    AOA_sub = nh.subscribe<nav_msgs::Odometry>("/AOA_Tag_data", 3, &UAVStateListener1::AOA_callback, this);
    flow_sub = nh.subscribe<nav_msgs::Odometry>("/flow_data", 3, &UAVStateListener1::flow_callback, this);
}

void UAVStateListener1::AOA_callback(const nav_msgs::Odometry::ConstPtr& msg){
    AOA_distance = msg->pose.pose.position.x;
    AOA_angle = msg->pose.pose.orientation.x;
}

void UAVStateListener1::flow_callback(const nav_msgs::Odometry::ConstPtr& msg){
    flow_z = msg->pose.pose.position.z;
}

/**
 * VINS 位姿回调 — 带固定延迟补偿
 *
 * 原理:
 *   相机捕获图像 → ArUco 检测 → 坐标变换 有约 FIXED_DELAY 个采样周期的延迟。
 *   因此不能直接用最新的 VINS 位姿来做坐标变换,需要取延迟对应时刻的位姿。
 *
 * 实现:
 *   1. 维护 vins_states 循环缓冲区,不断追加最新位姿
 *   2. 动态裁剪缓冲区: 超过 FIXED_DELAY*5 时删除前面 FIXED_DELAY*3 个(内存管理)
 *   3. 每 UPDATE_INTERVAL 次输入(当前代码 UPDATE_INTERVAL=5),
 *      从缓冲区 [size-FIXED_DELAY-UPDATE_INTERVAL, size-FIXED_DELAY) 区间取均值作为 T1,
 *      取中间时刻的旋转矩阵作为 R1
 *   4. 这样 get_T1_R1() 返回的是"延迟后"的位姿,
 *      与"延迟后"的视觉检测结果时间对齐
 */
void UAVStateListener1::pose_cb(const nav_msgs::Odometry::ConstPtr& msg) {
    // 将最新 VINS 状态追加到缓冲区
    vins_states.push_back(VINSState(msg));

    // 动态维护缓冲区长度,防止无限增长
    if (vins_states.size() > FIXED_DELAY * 5) {
        vins_states.erase(vins_states.begin(), vins_states.begin() + FIXED_DELAY * 3);
    }

    vins_input_count++;

    // 每 UPDATE_INTERVAL 次输入更新一次 T1(位置)和 R1(姿态)
    if (vins_input_count >= UPDATE_INTERVAL && vins_states.size() > FIXED_DELAY + UPDATE_INTERVAL) {
        // 计算延迟后的起始索引: 取 UPDATE_INTERVAL 个历史点做均值
        int start_idx = vins_states.size() - FIXED_DELAY - UPDATE_INTERVAL;
        int end_idx = vins_states.size() - FIXED_DELAY;

        if (start_idx < 0) {
            start_idx = 0;
            end_idx = std::min(UPDATE_INTERVAL, static_cast<int>(vins_states.size()));
        }

        // 位置: FIXED_DELAY 延迟后的 UPDATE_INTERVAL 个点的均值
        T1 = std::accumulate(
            vins_states.begin() + start_idx,
            vins_states.begin() + end_idx,
            Eigen::Vector3d(0,0,0),
            [](const Eigen::Vector3d& sum, const VINSState& state) {
                return sum + state.position;
            }
        );
        T1 /= (end_idx - start_idx);

        // 姿态: 取该时间窗口中间时刻的旋转矩阵
        int rotation_idx = start_idx + (end_idx - start_idx) / 2;
        if (rotation_idx >= vins_states.size()) {
            rotation_idx = vins_states.size() - 1;
        }
        R1 = vins_states[rotation_idx].orientation.toRotationMatrix();

        vins_input_count = 0;
    }
}

/**
 * SVO/位姿回调 — Bezier 轨迹预测
 *
 * 接收视觉里程计(SVO)或仿真位姿,通过 Bezier 曲线拟合历史轨迹,
 * 生成平滑的预测轨迹 g_predictedTrajectory。
 *
 * Bezier 预测的优势:
 *   - 平滑性: 保证 C² 连续(位置+速度+加速度连续)
 *   - 延迟补偿: 预测未来轨迹可以减少视觉延迟对控制的影响
 *   - 鲁棒性: 对单帧离群值不敏感
 */
#ifdef SIMULATE
    void UAVStateListener1::svo_callback(const boost::shared_ptr<const geometry_msgs::PoseStamped>& msg) {
#else
    void UAVStateListener1::svo_callback(const boost::shared_ptr<const geometry_msgs::PoseWithCovarianceStamped>& msg) {
#endif
        predict_index = 0;

#ifdef SIMULATE
        Eigen::Vector4d state(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z,ros::Time::now().toSec());
#else
        Eigen::Vector4d state(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,ros::Time::now().toSec());
#endif
        static std::vector<Eigen::Matrix<double,6,1>> svo_predict_list;
        // 初始化阶段: 收集 _MAX_SEG 个历史点
        if(!svo_initialize){
            svo_list.push_back(state);
            if(svo_list.size()>=_MAX_SEG){
                svo_initialize=1;
            }
        }
        else{
            // 滑动窗口: 删除最旧的点,加入最新的点
            svo_list.erase(svo_list.begin());
            svo_list.push_back(state);

            int bezier_flag = svopredict.TrackingGeneration(5,5,svo_list);
            if(bezier_flag==0){
                svo_predict_list = svopredict.getStateListFromBezier(_PREDICT_SEG);
                g_predictedTrajectory = svo_predict_list;
                // 保存上一帧状态用于差分
                last_svo_p = svo_p;
                svo_p.x() = svo_predict_list[0](0);
                svo_p.y() = svo_predict_list[0](1);
                svo_p.z() = svo_predict_list[0](2);
                svo_v.x() = svo_predict_list[0](3);
                svo_v.y() = svo_predict_list[0](4);
                svo_v.z() = svo_predict_list[0](5);
#ifndef SIMULATE
                svo_q.w() = msg->pose.pose.orientation.w;
                svo_q.x() = msg->pose.pose.orientation.x;
                svo_q.y() = msg->pose.pose.orientation.y;
                svo_q.z() = msg->pose.pose.orientation.z;
#else
                svo_q.w() = msg->pose.orientation.w;
                svo_q.x() = msg->pose.orientation.x;
                svo_q.y() = msg->pose.orientation.y;
                svo_q.z() = msg->pose.orientation.z;
#endif
                // 生成 200Hz 的匀速插值序列(将 60Hz SVO 上采样到 200Hz)
                result_x = predictPositions(last_svo_p.x(), svo_p.x());
                result_y = predictPositions(last_svo_p.y(), svo_p.y());
                result_z = predictPositions(last_svo_p.z(), svo_p.z());
                result_vx = predictPositions(last_svo_v.x(), svo_v.x());
                result_vy = predictPositions(last_svo_v.y(), svo_v.y());
                result_vz = predictPositions(last_svo_v.z(), svo_v.z());
            }
        }
    }


std::pair<Eigen::Vector3d, Eigen::Matrix3d>  UAVStateListener1::get_T1_R1() const {
    return std::make_pair(T1, R1);
}


// ============================================================================
// 八、数学工具函数
// ============================================================================

double round_to_decimal_places(double value, int decimal_places) {
    double factor = std::pow(10.0, decimal_places);
    return std::round(value * factor) / factor;
}

/**
 * correlation_coefficient — 皮尔逊相关系数
 * 用于评估速度线性拟合的质量(线性度越高,速度估计越可靠)
 */
double correlation_coefficient(const std::vector<double>& times, const std::vector<double>& data) {
    size_t n = data.size();
    if (n < 2 || times.size() != n) {
        throw std::invalid_argument("Data size too small or time size does not match data size");
    }

    double sum_x = std::accumulate(times.begin(), times.end(), 0.0);
    double sum_y = std::accumulate(data.begin(), data.end(), 0.0);
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    double sum_yy = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_xy += times[i] * data[i];
        sum_xx += times[i] * times[i];
        sum_yy += data[i] * data[i];
    }

    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y));

    if (denominator == 0) {
        throw std::runtime_error("Denominator is zero, cannot calculate correlation coefficient.");
    }

    double r = numerator / denominator;
    return r;
}

/**
 * imageCallback — 相机图像回调(ROS 订阅方式,当前未使用)
 * 保留的备用接口,实际代码使用 OpenCV 直接读取 USB 摄像头
 */
void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    try
    {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        frame = cv_ptr->image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

bool isRotationMatrix(const Eigen::Matrix3d& R) {
    Eigen::Matrix3d Rt = R.transpose();
    Eigen::Matrix3d shouldBeIdentity = Rt * R;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    double n = (I - shouldBeIdentity).norm();
    return n < 1e-6;
}

/**
 * rotationMatrixToEulerAngles — 旋转矩阵转 Z 轴欧拉角(yaw)
 * 使用 Z-Y-X 顺序分解,仅返回 yaw 分量
 * MATLAB 等效: atan2(R(2,1), R(1,1)) (Z 轴旋转)
 */
double rotationMatrixToEulerAngles(const Eigen::Matrix3d& R) {
    assert(isRotationMatrix(R));

    double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

    bool singular = sy < 1e-6;

    double x, y, z;
    if (!singular) {
        x = std::atan2(R(2, 1), R(2, 2));
        y = std::atan2(-R(2, 0), sy);
        z = std::atan2(R(1, 0), R(0, 0));
    } else {
        x = std::atan2(-R(1, 2), R(1, 1));
        y = std::atan2(-R(2, 0), sy);
        z = 0;
    }

    // 归一化到 [-π, π]
    if (z > M_PI) {
        z -= 2 * M_PI;
    }

    return z;
}

/**
 * rotationMatrixToEulerAnglesFull — 旋转矩阵转完整欧拉角(roll, pitch, yaw)
 * 用于同时获取三个姿态角,单位为弧度
 */
void rotationMatrixToEulerAnglesFull(const Eigen::Matrix3d& R, double& roll, double& pitch, double& yaw) {
    assert(isRotationMatrix(R));

    double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

    bool singular = sy < 1e-6;

    if (!singular) {
        roll = std::atan2(R(2, 1), R(2, 2));   // x 轴旋转
        pitch = std::atan2(-R(2, 0), sy);       // y 轴旋转
        yaw = std::atan2(R(1, 0), R(0, 0));    // z 轴旋转
    } else {
        roll = std::atan2(-R(1, 2), R(1, 1));
        pitch = std::atan2(-R(2, 0), sy);
        yaw = 0;
    }
}


// ============================================================================
// 九、主处理循环 — read::reda()
// ============================================================================
// 这是本模块的核心函数,运行在独立线程中,执行以下管道的每一帧:
//   1. 打开 USB 摄像头 (cv::VideoCapture, 1280x720 @30fps)
//   2. 初始化 ArUco 检测器 (DICT_7X7_250)
//   3. 每帧:
//      a) 获取延迟补偿后的 VINS 位姿 (T1, R1)
//      b) ArUco 标记检测 (detectMarkers)
//      c) PnP 位姿估计 (estimatePoseSingleMarkers) 对多种尺寸的标记
//      d) 坐标系变换: 标记→相机→无人机→世界
//      e) 多标记加权融合 (位置+yaw/pitch/roll)
//      f) 狗-视觉标定 (map_triger 条件)
//      g) Bezier 轨迹预测 + 发布 /target_ekf_odom

void read::reda(ros::NodeHandle& nh) {
    std::cout<<"here"<<std::endl;
    // 发布目标位姿到 /target_ekf_odom 话题
    target_pose_pub = nh.advertise<nav_msgs::Odometry>("/target_ekf_odom", 10);
#ifdef DELAY_TEST
    // 调试模式: 发布延迟补偿后的 VINS 位姿,用于验证延迟补偿效果
    delayed_vins_pub = nh.advertise<nav_msgs::Odometry>("/vins_delayed", 10);
#endif

    // 创建定时器: 200Hz 发布 VINS 里程计, 30Hz 发布狗位置预测
    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/vins_fusion/imu_propagate", 10);
    ros::Timer timer = nh.createTimer(ros::Duration(1.0 / 200.0),
                                     boost::bind(publishOdometry, _1, boost::ref(odom_pub)));
    ros::Timer gos_timer = nh.createTimer(ros::Duration(1.0 / 30.0),
                                     boost::bind(gos_pos_pub, _1, boost::ref(target_pose_pub)));

    // 三个独立的标量 Kalman 滤波器(x/y/z 轴分别滤波)
    static KalmanFilter kf_x(0.01, 0.0025);
    static KalmanFilter kf_y(0.01, 0.0025);
    static KalmanFilter kf_z(0.01, 0.0025);

    sleep(6);  // 等待各传感器节点启动完成
    double glo_yaw;  // 当前帧检测到的 yaw 角度(弧度)

    // ========================================================================
    // 坐标系变换矩阵定义
    // ========================================================================
    // 三个坐标系之间的变换关系:
    //
    // 相机坐标系 (OpenCV 标准):
    //   x: 右, y: 下, z: 前
    //
    // 无人机坐标系 (ROS/NED 机体坐标系):
    //   x: 前, y: 左, z: 上
    //
    // 标记坐标系 (ArUco 标准):
    //   x: 右(红色), y: 前(绿色), z: 上(蓝色)
    //
    // 变换链: M_相机→无人机:  (x,y,z)_camera → (x,-y,-z)_drone
    //                     即: x 不变, y 取反(下→上), z 取反(前→后)
    //         M_标记→相机:   (x,y,z)_tag → (y,-x,z)_camera
    //                     即: 标记右→相机前, 标记前→相机左, 标记上→相机上

    Eigen::Matrix3d M_camera2drone;
    M_camera2drone <<  1, 0, 0,    // 相机 x(右) → 无人机 x(前)... 实际上这里保留原始映射
                       0, -1, 0,   // 相机 y(下) → 无人机 -y(上方向)
                       0, 0, -1;   // 相机 z(前) → 无人机 -z(下方向)

    Eigen::Matrix3d M_tag2camera;
    M_tag2camera  <<  0, 1, 0,     // 标记 x(右) → 相机 y(前)?
                     -1, 0, 0,     // 标记 y(前) → 相机 -x(左)?
                      0, 0, 1;     // 标记 z(上) → 相机 z(上),不变

    nh_ = nh;
    // ArUco 检测器初始化: 7x7 字典, 250 个标记
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    Eigen::Vector3d pre_vel{0.0,0.0,0.0};  // 上一帧预测速度

    // ========================================================================
    // 相机内参 (Realsense D435 标定结果)
    // ========================================================================
    double fx = 734.804843, fy = 734.561878;     // 焦距(像素单位)
    double cx = 590.087740, cy = 422.783965;     // 光心(像素坐标)

    // 畸变系数 (径向+切向)
    double k1 = -0.037109;
    double k2 = 0.011897;
    double p1 = -0.004764;
    double p2 = 0.001270;
    double k3 = 0.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
    fx, 0, cx,
    0, fy, cy,
    0, 0, 1);

    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

    // ========================================================================
    // 打开 USB 摄像头 (V4L2 backend, 1280x720 @30fps, MJPG 编码)
    // ========================================================================
    cv::VideoCapture cap;
    for(int i;i<9;i++){
        if (cap.open(i, cv::CAP_V4L2)){
            std::cout<<"成功打开相机"<<i<<std::endl;
            break;
        }
        else
            std::cout<<"无法打开相机"<<i<<std::endl;
    }

    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cap.set(cv::CAP_PROP_FOURCC, fourcc);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "当前帧率: " << fps << " FPS" << std::endl;

#ifdef SAVE_VIDEO
    // 视频保存: 用于离线分析和调试
    auto time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm* tm_now = std::localtime(&time_now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", tm_now);
    std::string time_video(buffer);
    std::string unique_filename = "/home/pc/Fast-Perching/output_video_" + time_video + ".avi";
    int fourcc_video = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter out(unique_filename, fourcc_video, 30.0, cv::Size(1280,720));

    std::map<int, Eigen::Vector3d> markerPositions;

    if (!out.isOpened()) {
        std::cerr << "无法保存视频文件" << std::endl;
        exit(1);
    }
#endif

    // ========================================================================
    // 主循环 — 逐帧处理
    // ========================================================================
    while (true)
    {
        int triger_ = 8;
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = "world";
        cap >> frame;  // 从摄像头读取一帧

        if(triger_ != 1)
            triger2 = 0;

        if (frame.empty()) {
            continue;
        }
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ===== Step 1: ArUco 标记检测 =====
        // 输入: 灰度图像 + 预定义字典 + 检测参数
        // 输出:
        //   markerIds      — 检测到的标记 ID 列表
        //   markerCorners  — 每个标记的四个角点在图像中的像素坐标 [u, v]
        //   rejectedCandidates — 被拒绝的候选(用于调试)
        // 原理: 对图像做自适应阈值 → 轮廓提取 → 四边形检测 → 字典匹配
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        // ===== Step 2: 获取延迟补偿后的 VINS 位姿 =====
        // T1: 无人机在世界系下的位置 [x, y, z] (延迟 FIXED_DELAY 帧后的历史值)
        // R1: 无人机在世界系下的姿态旋转矩阵 (3×3, 同样取延迟后的历史值)
        // 为什么用"延迟后"的位姿？
        //   摄像头拍照 → 图像传输 → ArUco检测 → 坐标变换，整个过程约 FIXED_DELAY*dt 秒。
        //   如果此时用"最新"的 VINS 位姿，时间上不匹配，会产生系统性偏差。
        //   因此从 VINS 历史缓冲区取延迟对应时刻的位姿，实现"时间对齐"。
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> T1_R1_pair = uav_state_listener.get_T1_R1();
        T1 = T1_R1_pair.first;
        R1 = T1_R1_pair.second;

#ifdef DELAY_TEST
        // 调试: 发布延迟补偿后的 VINS 位姿
        nav_msgs::Odometry delayed_vins_msg;
        delayed_vins_msg.header.stamp = ros::Time::now();
        delayed_vins_msg.header.frame_id = "world";
        delayed_vins_msg.child_frame_id = "vins_delayed";

        delayed_vins_msg.pose.pose.position.x = T1[0];
        delayed_vins_msg.pose.pose.position.y = T1[1];
        delayed_vins_msg.pose.pose.position.z = T1[2];

        Eigen::Quaterniond delayed_q(R1);
        delayed_vins_msg.pose.pose.orientation.w = delayed_q.w();
        delayed_vins_msg.pose.pose.orientation.x = delayed_q.x();
        delayed_vins_msg.pose.pose.orientation.y = delayed_q.y();
        delayed_vins_msg.pose.pose.orientation.z = delayed_q.z();

        delayed_vins_pub.publish(delayed_vins_msg);
#endif
        cv::putText(frame, std::string("vins(delayed): ") +
                    "x=" + std::to_string(T1[0]) +
                    " y=" + std::to_string(T1[1]) +
                    " z=" + std::to_string(T1[2]),
                    cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        // ====================================================================
        // Step 3: 当检测到标记时 — PnP 位姿估计 + 坐标系变换 + 多标记融合
        // ====================================================================
        if (markerIds.size() > 0)
        {
            flag1+=1;  // 有效检测帧计数

            cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            std::vector<cv::Vec3d> rvecs1, tvecs1;  // 0.15m 标记的位姿
            std::vector<cv::Vec3d> rvecs0, tvecs0;  // 0.0165m 标记的位姿
            std::vector<cv::Vec3d> rvecs2, tvecs2;  // 0.06m 标记的位姿

            // 检查是否检测到主标记(ID=29)
            bool hasMarkerMain = false;
            for (int i = 0; i < markerIds.size(); ++i) {
                if (markerIds[i] == 29) {
                    hasMarkerMain = true;
                    break;
                }
            }

            // ===== PnP 位姿估计 (Perspective-n-Point) =====
            // solvePnP 求解的问题:
            //   已知: N 个 3D 空间点(标记角点在世界/标记系下的坐标)
            //         N 个 2D 像素点(标记角点在图像中的像素坐标)
            //   求解: 相机相对于标记的 6-DOF 位姿 (3 平移 + 3 旋转)
            //
            // 关键: PnP 需要知道标记的物理尺寸!
            //       同样大小在图像上的标记,距离远时像素尺寸小,距禈近时像素尺寸大。
            //       PnP 通过"2D像素坐标 → 物理尺寸 → 3D位置"的几何关系反推距离。
            //
            // 本系统使用三种尺寸的标记:
            //   0.15m  (主标记 ID=29, 大标记, 远距离可靠)
            //   0.06m  (辅助标记 ID=0~5, 中标记, 近距离使用)
            //   0.0165m (辅助标记 ID=33, 小标记)
            if (hasMarkerMain) {
                // 有主标记时优先处理主标记(0.15m 尺寸)
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, cameraMatrix, distCoeffs, rvecs1, tvecs1);
            } else {
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.0165, cameraMatrix, distCoeffs, rvecs0, tvecs0);
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, cameraMatrix, distCoeffs, rvecs1, tvecs1);
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.06, cameraMatrix, distCoeffs, rvecs2, tvecs2);
            }

            // 加权平均累加器(位置 + yaw/pitch/roll)
            Eigen::Vector3d position{0, 0, 0};
            Eigen::Vector3d averagePosition{0, 0, 0};
            double averageYaw = 0;
            double averagePitch = 0;
            double averageRoll = 0;
            double weight_count = 0;

            // ================================================================
            // Step 3a: 对每个检测到的标记,完成坐标变换链
            // ================================================================
            for (int i = 0; i < markerIds.size(); ++i)
            {
                int currentMarkerId = markerIds[i];

                // 有主标记时跳过其他标记
                if (hasMarkerMain && (currentMarkerId != 29)) {
                    continue;
                }

                Eigen::Vector3d T2;  // 标记在相机坐标系下的平移
                Eigen::Matrix3d R2;  // 标记在相机坐标系下的旋转矩阵
                cv::Vec3d rvec;
                cv::Vec3d tvec;

                // 按标记 ID 选择对应的 PnP 结果
                if(currentMarkerId == 29){
                    rvec = rvecs1[i];
                    tvec = tvecs1[i];
                }
                else if(currentMarkerId == 33){
                    rvec = rvecs0[i];
                    tvec = tvecs0[i];
                }
                else if(currentMarkerId == 0 || currentMarkerId == 1 || currentMarkerId == 2 ||
                        currentMarkerId == 3 || currentMarkerId == 4 || currentMarkerId == 5){
                    rvec = rvecs2[i];
                    tvec = tvecs2[i];
                }
                else
                    continue;

                // ——— 子步骤 1: PnP 输出 → 旋转矩阵 + 平移向量 ———
                // rvec 是罗德里格斯向量(Rodrigues vector),不是旋转矩阵,
                // 它是一个"旋转轴×旋转角"的紧凑表示,需要先转为 3×3 矩阵。
                // 罗德里格斯公式: R = I + sinθ·[k]× + (1-cosθ)·[k]×²
                T2 << tvec[0], tvec[1], tvec[2];      // tvec = 标记原点在相机系下的平移
                cv::Mat rotationMatrix;
                cv::Rodrigues(rvec, rotationMatrix);   // rvec → 3×3 旋转矩阵

                for (int i = 0; i < 3; ++i){
                    for (int j = 0; j < 3; ++j)
                        R2(i, j) = rotationMatrix.at<double>(i, j);}

                // ——— 子步骤 2: 坐标变换链 (标记系 → 世界系) ———
                // 这一条变换链把"标记在相机看到了什么"翻译成"目标在世界系下的绝对位置"。
                //
                // 第一步: T3 = M_camera2drone · T2
                //   T2: 标记原点在相机坐标系下的位置
                //   M_camera2drone: 相机→机体的旋转变换矩阵
                //   T3: 标记原点在机体坐标系下的位置 (变量名误导,应叫 T_body)
                //   这一步做了"右手系→前左上"的坐标系转换。
                Eigen::Vector3d T3 = M_camera2drone * T2;

                // 第二步: R2 = M_camera2drone · M_tag2camera · R2
                //   把标记的姿态从"标记→相机"链式转换到"标记→机体":
                //     R_tag_in_camera (PnP直接输出)
                //       → M_tag2camera 先变换方向定义 (标记右手系→相机右手系)
                //       → M_camera2drone 再变换到机体 (相机右手系→机体前左上)
                R2 = M_camera2drone * M_tag2camera * R2;

                // 第三步: position = R1 · T3 + T1
                //   R1: 无人机机体在世界系下的姿态 (来自 VINS)
                //   T1: 无人机质心在世界系下的位置 (来自 VINS)
                //   R1·T3: 将"标记在机体下的位置"旋转到世界系的方向
                //   + T1:  加上无人机自身在世界系的位置 → 得到目标世界系绝对坐标
                //   物理含义: "标记在机体下右前方2m处" → "标记在操场东北角3m高处"
                position = R1 * T3 + T1;
#ifdef DELAY_TEST
                position = R1 * T3;  // 调试模式: 只算相对于飞机的位置(排除T1)
#endif

                // 第四步: pose = R1 · R2
                //   将标记在世界系下的姿态矩阵通过两次旋转得到:
                //     R2 已经是"标记在机体下的旋转矩阵"(第二步已转换)
                //     R1·R2 = 标记在"世界系"下的完整旋转矩阵
                Eigen::Matrix3d pose = R1 * R2;

                // 第五步: 旋转矩阵 → Z-Y-X 欧拉角分解
                //   一个 3×3 旋转矩阵有 9 个元素,但只有 3 个自由度。
                //   用 Z-Y-X 顺序(先绕 Z 转 yaw, 再绕新 Y 转 pitch, 再绕新 X 转 roll)
                //   分解出三个直观的角度。
                double roll, pitch, yaw;
                rotationMatrixToEulerAnglesFull(pose, roll, pitch, yaw);

                glo_yaw = yaw;  // 保存原始 yaw 值用于屏显调试

                // ——— 子步骤 3: 多标记加权融合 ———
                // 狗身上贴了多个 ArUco 标记,每个标记在狗身体的不同位置。
                // 单个标记可能被遮挡/光照不好/角度不好,融合多标记提高鲁棒性。
                //
                // 权重设计原则:
                //   - 主标记 (ID=29): 0.8 — 贴在最显眼的位置,检测最稳定
                //   - 侧面标记 (ID=0~5): 0.65 — 分布在四角和两侧,提供多视角
                //   - 辅助标记 (ID=33): 0.5 — 尺寸小,检测精度有限
                double weight = 0.0;
                if (currentMarkerId == 29)
                    weight = 0.8;     // 主标记: 最高权重
                else if (currentMarkerId == 33)
                    weight = 0.5;
                else if (currentMarkerId == 0 || currentMarkerId == 1 || currentMarkerId == 2 ||
                         currentMarkerId == 3 || currentMarkerId == 4 || currentMarkerId == 5)
                    weight = 0.65;

                // 角度: 直接用加权值累加,最后除以 weight_count 归一化
                if (weight > 0) {
                    averageYaw += (yaw * weight);
                    averagePitch += (pitch * weight);
                    averageRoll += (roll * weight);
                }

                // ——— 子步骤 4: 标记位置补偿 (从标记几何中心 → 狗身体几何中心) ———
                // PnP 给出的是"标记中心"在世界系下的位置,但控制需要的是"狗的几何中心"。
                // 每个标记贴在狗身体的不同位置,需要根据当前狗朝向(fin_yaw)把位置归算到中心。
                //
                // 标记在狗身体上的物理布局:
                //   ID=0: 左前角  (-19.5cm前, +14.2cm左)
                //   ID=1: 右前角  (+19.5cm前, +14.2cm右)
                //   ID=2: 右后角  (+19.5cm前, -14.2cm右)
                //   ID=3: 左后角  (-19.5cm前, -14.2cm左)
                //   ID=4: 左侧中  (0cm前,    +14.2cm左)
                //   ID=5: 右侧中  (0cm前,    -14.2cm右)
                //
                // 补偿公式: center = tag_pos + R(fin_yaw) · offset
                //   例如 ID=0:
                //     center.x = tag.x + (-0.195*cos(yaw) + 0.142*sin(yaw))
                //     center.y = tag.y + (-0.195*sin(yaw) - 0.142*cos(yaw))
                //   含义: 把"左前角坐标"加上"从中心到左前角的向量(在狗朝向下的世界系表达)"
                if (currentMarkerId == 29){
                    // 主标记: 正面偏移(-sin(yaw)方向,右方)
                    position.x() = position.x() + 0.0*std::sin(fin_yaw);
                    position.y() = position.y() - 0.0*std::cos(fin_yaw);
                    averagePosition += (position * 0.8);
                }
                else if (currentMarkerId == 33){
                    // 辅助标记 33
                    position.x() = position.x() + 0.0*std::sin(fin_yaw);
                    position.y() = position.y() - 0.0*std::cos(fin_yaw);
                    averagePosition += (position * 0.5);
                }
                else if (currentMarkerId == 0){
                    // 标记 0: 位于狗身体左前角 (前方 19.5cm, 左侧 14.2cm)
                    position.x() = position.x() - 0.195*std::cos(fin_yaw) + 0.142*std::sin(fin_yaw);
                    position.y() = position.y() - 0.195*std::sin(fin_yaw) - 0.142*std::cos(fin_yaw);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 1){
                    // 标记 1: 右前角
                    position.x() = position.x() + 0.195*std::cos(fin_yaw) + 0.142*std::sin(fin_yaw);
                    position.y() = position.y() + 0.195*std::sin(fin_yaw) - 0.142*std::cos(fin_yaw);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 2){
                    // 标记 2: 右后角
                    position.x() = position.x() + 0.195*std::cos(fin_yaw) - 0.142*std::sin(fin_yaw);
                    position.y() = position.y() + 0.195*std::sin(fin_yaw) + 0.142*std::cos(fin_yaw);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 3){
                    // 标记 3: 左后角
                    position.x() = position.x() - 0.195*std::cos(fin_yaw) - 0.142*std::sin(fin_yaw);
                    position.y() = position.y() - 0.195*std::sin(fin_yaw) + 0.142*std::cos(fin_yaw);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 4){
                    // 标记 4: 左侧中点
                    position.x() = position.x() + 0.142*std::sin(fin_yaw);
                    position.y() = position.y() - 0.142*std::cos(fin_yaw);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 5){
                    // 标记 5: 右侧中点
                    position.x() = position.x() - 0.142*std::sin(fin_yaw);
                    position.y() = position.y() + 0.142*std::cos(fin_yaw);
                    averagePosition += (position * 0.65);
                }

                // 累计权重之和,用于后续归一化
                if(currentMarkerId == 29)
                    weight_count += 0.8;
                else if(currentMarkerId == 33)
                    weight_count += 0.5;
                else if(currentMarkerId == 0 || currentMarkerId == 1 || currentMarkerId == 2 ||
                        currentMarkerId == 3 || currentMarkerId == 4 || currentMarkerId == 5)
                    weight_count += 0.65;
            }

            // ================================================================
            // Step 3b: yaw/pitch/roll 角度滤波 (死区 + 一阶低通)
            // ================================================================
            // 角度滤波的两层机制:
            //   1. 死区 (Dead Zone): 变化小于 3° 时不更新,抑制微小抖动
            //   2. 一阶低通: 变化超过死区后,不是跳到新值,而是以 α=0.5 的指数平滑靠近
            //
            // 三层条件检查 (为什么检查三个值):
            //   |this - fin| > 3°  → 直接差值超过死区
            //   |this - fin + 2π| > 3° → 考虑"新值比旧值小一圈(360°)跳变"的情况
            //   |this - fin - 2π| > 3° → 考虑"新值比旧值大一圈"的情况
            //   三个条件都满足 → 角度确实有实质变化,更新
            //   任一条件 < 3° → 在死区内,保持旧值
            //
            // 举例: fin_yaw = 0°, this_yaw = 359° (绕过了 0°)
            //   |359-0| = 359° > 3° → 第一条件满足 ✓
            //   |359-0+360| = |719-1*360| = 359° > 3° → 第二条件满足 ✓
            //   |359-0-360| = |-1| = 1° < 3° → 第三条件不满足 ✗
            //   → 判定为"这不是真实角度变化,只是碰巧在 -π 附近",不更新
            double this_yaw = averageYaw / weight_count;
            const double angle_threshold_rad = 3.0 * M_PI / 180.0;  // 3 度死区

            // yaw 滤波
            if (std::fabs(this_yaw - fin_yaw) > angle_threshold_rad &&
            std::fabs(this_yaw - fin_yaw + 2 * M_PI) > angle_threshold_rad &&
            std::fabs(this_yaw - fin_yaw - 2 * M_PI) > angle_threshold_rad)
            {
                double delta_yaw = this_yaw - fin_yaw;
                // 角度环绕: 从 fin_yaw=170° 到 this_yaw=-170°,
                // 最短路径是逆时针转 20°,而不是顺时针转 340°
                if (delta_yaw > M_PI) {
                    delta_yaw -= 2 * M_PI;    // 顺时针路径太长,改走逆时针
                } else if (delta_yaw < -M_PI) {
                    delta_yaw += 2 * M_PI;    // 逆时针路径太长,改走顺时针
                }
                // 一阶低通 (指数平滑): new = old + α·(target - old)
                // α=0.5: 两步后达到目标值的 75%, 三步后 87.5%
                fin_yaw += 0.5 * delta_yaw;
            }

            if(fin_yaw > M_PI)
                fin_yaw -= 2 * M_PI;
            else if(fin_yaw < -M_PI)
                fin_yaw += 2 * M_PI;

            // pitch 滤波 (与 yaw 相同的死区+低通策略)
            double this_pitch = averagePitch / weight_count;
            if (std::fabs(this_pitch - fin_pitch) > angle_threshold_rad &&
            std::fabs(this_pitch - fin_pitch + 2 * M_PI) > angle_threshold_rad &&
            std::fabs(this_pitch - fin_pitch - 2 * M_PI) > angle_threshold_rad)
            {
                double delta_pitch = this_pitch - fin_pitch;
                if (delta_pitch > M_PI) {
                    delta_pitch -= 2 * M_PI;
                } else if (delta_pitch < -M_PI) {
                    delta_pitch += 2 * M_PI;
                }
                fin_pitch += 0.5 * delta_pitch;
            }

            if(fin_pitch > M_PI)
                fin_pitch -= 2 * M_PI;
            else if(fin_pitch < -M_PI)
                fin_pitch += 2 * M_PI;

            // roll 滤波 (与 yaw 相同的死区+低通策略)
            double this_roll = averageRoll / weight_count;
            if (std::fabs(this_roll - fin_roll) > angle_threshold_rad &&
            std::fabs(this_roll - fin_roll + 2 * M_PI) > angle_threshold_rad &&
            std::fabs(this_roll - fin_roll - 2 * M_PI) > angle_threshold_rad)
            {
                double delta_roll = this_roll - fin_roll;
                if (delta_roll > M_PI) {
                    delta_roll -= 2 * M_PI;
                } else if (delta_roll < -M_PI) {
                    delta_roll += 2 * M_PI;
                }
                fin_roll += 0.5 * delta_roll;
            }

            if(fin_roll > M_PI)
                fin_roll -= 2 * M_PI;
            else if(fin_roll < -M_PI)
                fin_roll += 2 * M_PI;

            // ================================================================
            // Step 3c: 位置加权平均 + 滑动窗口滤波
            // ================================================================
            position = averagePosition / weight_count;
            // 超时重置: 超过 0.2s 没有新检测则清空滑动窗口
            if((ros::Time::now() - target_start_time).toSec() > 0.2)
              {
                  pos.clear();
              }
            pos.push_back({position[0], position[1], position[2]});
            target_start_time = ros::Time::now();

            // 3 帧滑动窗口均值滤波
            if(pos.size() > 3){
                pos.erase(pos.begin());
                Eigen::Vector3d sum(0.0, 0.0, 0.0);
                for (const auto& p : pos) {
                    sum += p;
                }
                avg = sum / pos.size();
            }


            // ================================================================
            // Step 3d: 狗-视觉坐标系标定
            // ================================================================
            // 原理: 狗自身报告的 UWB/GPS 位置(gos_pos)在"狗坐标系"下,
            //       视觉检测的位置(position)在"世界坐标系(VINS 系)"下,
            //       需要标定两个坐标系之间的旋转和平移变换。
            //
            // 标定方法:
            //   - 记录一对 (狗报告位置 gos_p0, 视觉位置 gos_vins_p0)
            //   - 后续通过 reflect_list 维护多对标定点
            //   - 变换: gos_vins_pos = R(-r_yaw)·(gos_pos - gos_p0) + gos_vins_p0
            //
            //   map_triger 控制标定采样频率: 每 24 帧(约 0.8s)采样一次
            if(map_triger>24){
                gos_p0 =  gos_pos;                              // 标定时狗报告的位置
                gos_vins_p0 = position;                         // 标定时视觉检测的位置
                if(pos.size() >= 3)
                    gos_vins_p0 = avg;                          // 有滑动窗口则用均值
                gos_vins_yaw0 = fin_yaw;                        // 标定时视觉 yaw
                gos_yaw0 = gos_yaw;                             // 标定时狗报告 yaw
                d_yaw = gos_vins_yaw0 - gos_yaw0;               // yaw 差
                Eigen::Vector3d reflect;
                std::vector<Eigen::Vector3d> reflect_kid;
                reflect.x() = d_yaw;
                reflect.y() = d_yaw;
                reflect.z() = d_yaw;
                reflect_kid.push_back(gos_p0);
                reflect_kid.push_back(gos_vins_p0);
                reflect_kid.push_back(reflect);                 // reflect = {dog_p0, vins_p0, yaw_diff}
                reflect_list.push_back(reflect_kid);
                if (reflect_list.size() > 10) {
                    reflect_list.erase(reflect_list.begin());   // 保留最近 10 个标定点
                }
                gos_triger = 1;                                 // 触发狗位置预测
                map_triger = 0;
              }
              map_triger+=1;


            // ================================================================
            // Step 3e: 发布目标位姿 (/target_ekf_odom)
            // ================================================================
            // 这是 read.cpp 给下游的最终输出。填充一个 Odometry 消息,
            // 包含目标狗在世界系下的完整 6-DOF 状态。
            //
            // 消息字段与物理量的对应:
            //   position.xyz      → 狗世界系位置 [x, y, z]
            //   orientation.w     → 狗 yaw 角 (滤波后,精度高)
            //   orientation.x     → 狗 pitch 角 (滤波后)
            //   orientation.y     → 狗 roll 角 (滤波后)
            //   linear_vel.xyz    → 狗世界系速度 [vx, vy, vz] (Bezier预测)
            averge_vel = {0, 0, 0};
            glo_pos = position;
            if (pos.size() >= 3) {
                // 位置: 用滑动窗口均值 avg (比单帧 position 平滑)
                double filtered_x = avg.x();
                double filtered_y = avg.y();
                double filtered_z = avg.z();

                // 第一版填充 — 位置和角度先写入 (速度可能被 Bezier 覆写)
                odom_msg.pose.pose.position.x = round_to_decimal_places(filtered_x, 2);
                odom_msg.pose.pose.position.y = round_to_decimal_places(filtered_y, 2);
                odom_msg.pose.pose.position.z = round_to_decimal_places(filtered_z, 2);
                odom_msg.twist.twist.linear.x = 1.0*averge_v[0];  // 第一版速度=0 (占位)
                odom_msg.twist.twist.linear.y = 1.0*averge_v[1];
                odom_msg.twist.twist.linear.z = 0;
                odom_msg.pose.pose.orientation.w = fin_yaw;      // yaw(弧度)
                odom_msg.pose.pose.orientation.x = fin_pitch;    // pitch(弧度)
                odom_msg.pose.pose.orientation.y = fin_roll;     // roll(弧度)
                odom_msg.pose.pose.orientation.z = 0;

                // ===== Bezier 轨迹预测 =====
                // 为什么要做 Bezier 预测?
                //   1. 视觉检测是离散的点,有噪声 → 先拟合成平滑曲线
                //   2. 视觉检测有处理延迟(~2-3帧) → 曲线外推未来位置来补偿
                //   3. 位置曲线求导 → 速度 (比直接差分噪声小得多)
                //
                // Bezier 曲线原理:
                //   对历史 N 个位置点 [t_i, x_i, y_i, z_i] 做 5 次 Bezier 多项式拟合:
                //     B(t) = Σ P_i · C(n,i) · t^i · (1-t)^(n-i)
                //   P_i 是控制点,拟合过程通过最小二乘确定最优控制点位置,
                //   保证 C² 连续 (位置+速度+加速度都连续)。
                //
                // 初始化阶段:
                //   收集至少 _MAX_SEG 个历史点才开始拟合
                static bool initialize = false;
                static std::vector<Eigen::Matrix<double,6,1>> predict_state_list;
                Eigen::Vector4d state(odom_msg.pose.pose.position.x,
                                     odom_msg.pose.pose.position.y,
                                     odom_msg.pose.pose.position.z,
                                     ros::Time::now().toSec());
                if(!initialize){
                    target_detect_list.push_back(state);
                    if(target_detect_list.size()>=_MAX_SEG){
                        initialize=1;
                    }
                }
                else{
                    // 滑动窗口: 先进先出,窗口大小固定 = _MAX_SEG
                    target_detect_list.erase(target_detect_list.begin());
                    target_detect_list.push_back(state);

                    // TrackingGeneration(5, 5, data)
                    //   参数: (Bezier阶数=5, 分段数=5, 数据点列表)
                    //   返回: 0=成功, 非0=点太少或无法拟合
                    int bezier_flag = tgpredict.TrackingGeneration(5, 5, target_detect_list);
                    if(bezier_flag==0){
                        // getStateListFromBezier(_PREDICT_SEG)
                        //   返回 _PREDICT_SEG 个状态向量 [px,py,pz, vx,vy,vz]
                        //   速度是 Bezier 曲线的解析导数 (不是差分!)
                        predict_state_list = tgpredict.getStateListFromBezier(_PREDICT_SEG);

                        // 为什么用 predict_state_list[2] 而不是 [0]?
                        //   [0] 是最靠近"当前时刻"的点 → 延迟最小,但可能有拟合边界效应
                        //   [2] 稍微超前一点 → 折中延迟补偿与平滑性
                        //   [>2] 预测太远 → 匀速假设会放大误差
                        odom_msg.twist.twist.linear.x = 1.0*predict_state_list[2](3);  // vx
                        odom_msg.twist.twist.linear.y = 1.0*predict_state_list[2](4);  // vy
                        odom_msg.twist.twist.linear.z = 0;  // z方向速度不做Bezier预测
                        pre_vel[0] = predict_state_list[0](3);  // 保存第0帧速度供调试
                        pre_vel[1] = predict_state_list[0](4);
                    }
                    // 注意: 如果 Bezier 拟合失败 (bezier_flag≠0),
                    // 速度保持第一版填充的 0 (averge_v),不更新。
                    // 下游节点会检测到速度为 0,切换到位置纯反馈控制。
                }
                target_pose_pub.publish(odom_msg);  // 发布 /target_ekf_odom → 全网使用
            }

            // 绘制标记坐标系(用于可视化调试)
            for (int i = 0; i < markerIds.size(); ++i)
            {
                int currentMarkerId1 = markerIds[i];
                cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs1[i], tvecs1[i], 0.1);
            }
        }

        // ====================================================================
        // 可视化覆盖 — 在图像上叠加调试信息
        // ====================================================================
        cv::putText(frame, std::string("c_pos: ") +
                    "x=" + std::to_string(avg[0]-vins_p[0]) +
                    " y=" + std::to_string(avg[1]-vins_p[1]) +
                    " z=" + std::to_string(avg[2]-vins_p[2]),
                    cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::putText(frame, std::string("yaw (rad): ") + std::to_string(glo_yaw),
                    cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::putText(frame, std::string("last_yaw (rad): ") + std::to_string(last_yaw),
                    cv::Point(10, 230), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::putText(frame, std::string("pre_vel: ") +
                    "x=" + std::to_string(pre_vel[0]) +
                    " y=" + std::to_string(pre_vel[1]) +
                    " z=" + std::to_string(pre_vel[2]),
                    cv::Point(10, 270), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

#ifdef SAVE_VIDEO
        out.write(frame);
#endif
#ifdef SCREEN_SHOW
        cv::imshow("ArUco Detection", frame);
#endif
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

#ifdef SAVE_VIDEO
    out.release();
    cv::destroyAllWindows();
#endif
}


// ============================================================================
// 十、Nodelet 入口 — 将 read 类注册为 ROS nodelet 插件
// ============================================================================
// nodelet 允许在同一个进程中运行多个节点,避免进程间通信开销,
// 同时利用 pluginlib 实现动态加载

void read::onInit() {
    // 在独立线程中启动主处理循环,避免阻塞 ROS 回调
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    initThread_ = std::thread(std::bind(&read::reda, this, nh));
}

read::~read() {
    if (initThread_.joinable()) {
        initThread_.join();  // 等待处理线程结束,防止资源泄漏
    }
}

}//namespace
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(image::read, nodelet::Nodelet);
