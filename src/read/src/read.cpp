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
#include<bezier_predict.h>
#include <map>
using namespace cv;

namespace image{
int flag = 1;
Eigen::Vector3d glo_pos{0.0, 0.0, 0.0};
Eigen::Vector3d glo_vel{0.0, 0.0, 0.0};
Eigen::Vector3d averge_v{0.0, 0.0, 0.0};//sudu
std::vector<double> pos_copy;
std::mutex global_mutex;
int flag1 = 0;
Eigen::Vector3d averge_vel{0.0, 0.0, 0.0};//weihzi
cv::Mat frame;  
double last_deg = 0;
double fin_deg = 0;
int triger2 = 0;
int vins_triger = 0;
int gos_triger = 0;
int map_triger = 16;
Eigen::Vector3d avg{0.0, 0.0, 0.0};

Eigen::Vector3d svo_p{0.0, 0.0, 0.0},svo_v{0.0, 0.0, 0.0};

Eigen::Vector3d vins_p{0.0, 0.0, 0.0};
Eigen::Vector3d last_svo_p{0.0, 0.0, 0.0},last_svo_v{0.0, 0.0, 0.0};
Eigen::Quaterniond svo_q;
std::vector<double> result_x(6,0.0),result_y(6,0.0),result_z(6,0.0),result_vx(6,0.0),result_vy(6,0.0),result_vz(6,0.0);
std::vector<Eigen::Matrix<double, 6, 1>> g_predictedTrajectory(
    _PREDICT_SEG, Eigen::Matrix<double, 6, 1>::Zero()
);
int predict_index = 0;
static bool svo_initialize = false;

Eigen::Vector3d gos_v{0.0, 0.0, 0.0};
Eigen::Vector3d gos_vins_v{0.0, 0.0, 0.0};
double gos_yaw;
double gos_yaw0 = 0;
double gos_vins_yaw0 = 0;
double gos_vins_yaw = 0;
double d_yaw = 0;
Eigen::Vector3d gos_pos{0.0, 0.0, 0.0};
Eigen::Vector3d vins_p0{0.0, 0.0, 0.0};
Eigen::Vector3d gos_p0{0.0, 0.0, 0.0};
Eigen::Vector3d gos_vins_p0{0.0, 0.0, 0.0};
Eigen::Vector3d gos_vins_pos{0.0, 0.0, 0.0};
std::vector<double> yaw_list;
std::vector<std::vector<Eigen::Vector3d>> reflect_list;
int land_triger = 0;

ros::Time start_gos_time;
ros::Time target_start_time = ros::Time::now();

std::vector<Eigen::Vector4d> target_detect_lis;
Bezierpredict predict;

ros::Time predict_time;

double last_gos_vins_vx = -2.0;
double last_gos_vins_vy = -2.0;

double AOA_distance;
double AOA_angle;

double flow_z = 0.0;



MultiKalmanFilter::MultiKalmanFilter(double pos_process_var, double pos_meas_var, double dist_meas_var)
        : process_noise_(Eigen::Matrix3d::Identity() * pos_process_var),
        pos_meas_noise_(Eigen::Matrix3d::Identity() * pos_meas_var),
        dist_meas_noise_(dist_meas_var),
        is_initialized_(false) {}
Eigen::Vector3d  MultiKalmanFilter::filter(const Eigen::Vector3d& pos_meas, double distance_meas) {
    if (!is_initialized_) {
        state_ = pos_meas;
        covariance_ = Eigen::Matrix3d::Identity();
        is_initialized_ = true;
        return state_;
    }

    // Prediction step
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity(); // 状态转移矩阵
    state_ = F * state_;
    covariance_ = F * covariance_ * F.transpose() + process_noise_;

    // Update step with position measurements
    Eigen::Matrix3d H_pos = Eigen::Matrix3d::Identity(); // 位置观测矩阵
    Eigen::Vector3d y_pos = pos_meas - H_pos * state_;
    Eigen::Matrix3d S_pos = H_pos * covariance_ * H_pos.transpose() + pos_meas_noise_;
    Eigen::Matrix3d K_pos = covariance_ * H_pos.transpose() * S_pos.inverse();
    state_ = state_ + K_pos * y_pos;
    covariance_ = (Eigen::Matrix3d::Identity() - K_pos * H_pos) * covariance_;

    // Additional update with distance measurement
    double pred_distance = state_.norm();
    double y_dist = distance_meas - pred_distance;
    
    // 计算雅可比矩阵（距离观测的导数）
    Eigen::Matrix<double, 1, 3> H_dist;
    if (pred_distance > 1e-6) {
        H_dist << state_.x()/pred_distance, 
                  state_.y()/pred_distance,
                  state_.z()/pred_distance;
    } else {
        H_dist.setZero();
    }
    
    // 计算卡尔曼增益
    double S_dist = (H_dist * covariance_ * H_dist.transpose())(0) + dist_meas_noise_;
    Eigen::Matrix<double, 3, 1> K_dist = covariance_ * H_dist.transpose() / S_dist;

    // 更新状态
    state_ = state_ + K_dist * y_dist;
    covariance_ = (Eigen::Matrix3d::Identity() - K_dist * H_dist) * covariance_;

    return state_;
}


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

    // Predict
    double predicted_state = state_;
    double predicted_covariance = covariance_ + process_variance_;

    // Update
    double kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance_);
    state_ = predicted_state + kalman_gain * (measurement - predicted_state);
    covariance_ = (1 - kalman_gain) * predicted_covariance;

    return state_;
}

MedianFilter1::MedianFilter1(size_t size) : size(size) {}


void publishOdometry(const ros::TimerEvent&, ros::Publisher &odom_pub) {

    
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = ros::Time::now();
    odom_msg.header.frame_id = "world";
    if(svo_initialize){
        // std::cout<<"别6了"<<std::endl;
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


    // 位置
    odom_msg.pose.pose.position.x = result_x[predict_index];
    odom_msg.pose.pose.position.y = result_y[predict_index];
    odom_msg.pose.pose.position.z = result_z[predict_index];

    // 方向（四元数）
    odom_msg.pose.pose.orientation.x = svo_q.x();
    odom_msg.pose.pose.orientation.y = svo_q.y();
    odom_msg.pose.pose.orientation.z = svo_q.z();
    odom_msg.pose.pose.orientation.w = svo_q.w();
    if (svo_v.norm() < 4 || vins_triger == 0){    
    // 速度
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
    if(predict_index<=3)
        predict_index+=1;
   }

    // 发布消息
    
}

Eigen::Matrix3d correctYaw(const Eigen::Matrix3d& pose) {
    Eigen::Vector3d eulerAngles = pose.eulerAngles(2, 1, 0); // 2, 1, 0代表Z-Y-X顺序

    // 提取偏航角（yaw），俯仰角（pitch），滚转角（roll）
    double yaw = eulerAngles[0];  // 偏航角（单位：弧度）
    double pitch = eulerAngles[1];  // 俯仰角（单位：弧度）
    double roll = eulerAngles[2];  // 滚转角（单位：弧度）

    // 你可以在这里进行对yaw、pitch、roll的修正，如果需要的话
    // 比如确保yaw在[-pi, pi]范围内
    if (yaw > M_PI) {
        yaw -= 2 * M_PI;
    } else if (yaw < -M_PI) {
        yaw += 2 * M_PI;
    }

    if (yaw > 0) {
        yaw -=  M_PI;
    } else if (yaw < 0) {
        yaw += M_PI;
    }
    
    // 这里将修正后的偏航角重新应用到旋转矩阵中
    Eigen::Matrix3d correctedPose = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                                    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix() *
                                    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();

    // 返回修正后的旋转矩阵
    return correctedPose;

}

void gos_pos_pub(const ros::TimerEvent&, ros::Publisher &odom_pub) {

    if(gos_triger == 1){
        nav_msgs::Odometry gos_msg;

        // Set the frame ID and timestamp
        gos_msg.header.stamp = ros::Time::now();
        gos_msg.header.frame_id = "world";
        // if(svo_initialize)
        //     std::cout<<g_predictedTrajectory.size()<<std::endl;
        std::vector<Eigen::Vector3d> gos_p_list;
        std::vector<Eigen::Vector3d> gos_v_list;
        std::vector<double> gos_yaw_list;
        for(int i = 0;i<reflect_list.size();i++){
            std::vector<Eigen::Vector3d> vec = reflect_list[i];
            Eigen::Vector3d dog_p0 = vec[0];
            Eigen::Vector3d dog_vins_p0 = vec[1];
            double r_yaw = vec[2].x();
            Eigen::Vector3d r_p = gos_pos - dog_p0;
            double cos_theta = cos(r_yaw);
            double sin_theta = sin(r_yaw);
            double rotated_x = cos_theta * r_p.x() - sin_theta * r_p.y();
            double rotated_y = sin_theta * r_p.x() + cos_theta * r_p.y();
            r_p.x() = rotated_x;
            r_p.y() = rotated_y;
            Eigen::Vector3d dog_vins_pos = r_p + dog_vins_p0;
            double dog_vins_yaw = gos_yaw + r_yaw;
            double rotated_vx = cos_theta * gos_v.x() - sin_theta * gos_v.y();
            double rotated_vy = sin_theta * gos_v.x() + cos_theta * gos_v.y();
            Eigen::Vector3d  dog_vins_v = gos_v;
            dog_vins_v.x() = rotated_vx;
            dog_vins_v.y() = rotated_vy;
            gos_p_list.push_back(dog_vins_pos);
            gos_v_list.push_back(dog_vins_v);
            gos_yaw_list.push_back(dog_vins_yaw);

        }
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
                // odom_msg.pose.pose.position.x = predict_state_list[0](0);
                // odom_msg.pose.pose.position.y = predict_state_list[0](1);
                // odom_msg.pose.pose.position.z = predict_state_list[0](2);
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
        if(fabs(gos_vins_pos.z() - svo_p.z()) < 3)
            gos_msg.pose.pose.position.z = round_to_decimal_places(gos_vins_pos.z(), 2)+0.05;
        else
            gos_msg.pose.pose.position.z = svo_p.z();
        // gos_msg.twist.twist.linear.x = gos_vins_v.x();
        // gos_msg.twist.twist.linear.y = gos_vins_v.y();
        // gos_msg.twist.twist.linear.z = 0;
        gos_msg.pose.pose.orientation.w = gos_vins_yaw;
        gos_msg.pose.pose.orientation.x = 0;
        gos_msg.pose.pose.orientation.y = 0;
        gos_msg.pose.pose.orientation.z = 0;
        // std::cout<<gos_pos.x()<<std::endl; 

        if( land_triger == 1)
            return;
        // odom_pub.publish(gos_msg);
    }

}

std::vector<double> predictPositions(double pos1, double pos2) {
    const double dt_input = 1.0 / 60;   // 输入位置的时间间隔
    const double dt_predict = 1.0 / 200;  // 预测位置的时间间隔

    // 假设匀速运动，计算速度
    double velocity = (pos2 - pos1) / dt_input;  
    
    std::vector<double> predictedPositions;
    predictedPositions.push_back(pos2);
    // 依次计算下5个位置
    for (int i = 1; i <= 5; ++i) {
        double timeElapsed = i * dt_predict;  // 从 pos2 开始经过的时间
        double newPos = pos2 + velocity * timeElapsed;
        predictedPositions.push_back(newPos);
    }
    
    return predictedPositions;
}

double MedianFilter1::update(double new_value){
    // 将新值添加到窗口
        if (window.size() >= size) {
            window.pop_front();  // 保持窗口大小不超过设定值
        }
        window.push_back(new_value);

        // 复制当前窗口用于计算中值
        std::vector<double> sorted_window(window.begin(), window.end());

        // 计算中值
        std::sort(sorted_window.begin(), sorted_window.end());
        size_t mid = sorted_window.size() / 2;

        if (sorted_window.size() % 2 == 0) {
            return (sorted_window[mid - 1] + sorted_window[mid]) / 2.0; // 偶数个元素时，取中间两个元素的平均值
        } else {
            return sorted_window[mid]; // 奇数个元素时，取中间值
        }
}

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

        if (list.size() == 2) {
            // 计算平均位置 T1
            T1 = std::accumulate(list.begin(), list.end(), Eigen::Vector3d(0,0,0),
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a + b;
            });

            // 检查 list.size() 是否为 0，防止除以 0 的错误
            if (list.size() > 0) {
                T1 /= static_cast<double>(list.size());
            } else {
                T1 = Eigen::Vector3d::Zero(); // 如果 list 为空，T1 设为零向量
            }

            // 获取四元数并转化为旋转矩阵 R1
            tf::Quaternion quat(
                msg->pose.pose.orientation.x,
                msg->pose.pose.orientation.y,
                msg->pose.pose.orientation.z,
                msg->pose.pose.orientation.w
            );
            trigger_condition_met = true;
            if (T1[0] != 0 && T1[2] != 0){
                gos_pos = T1;
                gos_yaw = msg->pose.pose.orientation.w;
                gos_v = current_v;}

            // 清空列表
            list.clear();
        }
    }

void gos_listener::pose_cb_land(const geometry_msgs::PoseStamped::ConstPtr& msg){
    land_triger = 1;
}

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
        //pose_sub = nh.subscribe<nav_msgs::Odometry>("/vins_fusion/imu", 3, &UAVStateListener1::pose_cb, this);
    }


void UAVStateListener1::AOA_callback(const nav_msgs::Odometry::ConstPtr& msg){
    AOA_distance = msg->pose.pose.position.x;
    AOA_angle = msg->pose.pose.orientation.x;
}

void UAVStateListener1::flow_callback(const nav_msgs::Odometry::ConstPtr& msg){
    flow_z = msg->pose.pose.position.z;
}


void UAVStateListener1::pose_cb(const nav_msgs::Odometry::ConstPtr& msg) {
    
    // 只保存必要的位置、速度和角度信息，而不是完整的消息
    vins_states.push_back(VINSState(msg));
    
    // 动态维护vins_states长度：超过延迟数量5倍时，删除前面3倍
    if (vins_states.size() > FIXED_DELAY * 5) {
        vins_states.erase(vins_states.begin(), vins_states.begin() + FIXED_DELAY * 3);
    }
    
    // 增加VINS输入计数器
    vins_input_count++;
    
    // 每5次输入更新一次T1和R1
    if (vins_input_count >= UPDATE_INTERVAL && vins_states.size() > FIXED_DELAY + UPDATE_INTERVAL) {
        // 直接在这里更新T1和R1

        // 计算延迟后的起始索引
        int start_idx = vins_states.size() - FIXED_DELAY - UPDATE_INTERVAL;
        int end_idx = vins_states.size() - FIXED_DELAY;
        
        // 确保索引在有效范围内
        if (start_idx < 0) {
            start_idx = 0;
            end_idx = std::min(UPDATE_INTERVAL, static_cast<int>(vins_states.size()));
        }
        
        // 计算延迟后的平均位置（5阶平均）- 使用std::accumulate避免for循环
        T1 = std::accumulate(
            vins_states.begin() + start_idx, 
            vins_states.begin() + end_idx, 
            Eigen::Vector3d(0,0,0),  // 使用与能行代码相同的初始值
            [](const Eigen::Vector3d& sum, const VINSState& state) {
                return sum + state.position;
            }
        );
        T1 /= (end_idx - start_idx);
        
        // 取中间时刻的值提取方向
        int rotation_idx = start_idx + (end_idx - start_idx) / 2;
        if (rotation_idx >= vins_states.size()) {
            rotation_idx = vins_states.size() - 1;
        }
        
        // 更新T1和R1
        R1 = vins_states[rotation_idx].orientation.toRotationMatrix();
        
        vins_input_count = 0;  // 重置计数器
    }
}


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
        // static bool svo_initialize = false;
        static std::vector<Eigen::Matrix<double,6,1>> svo_predict_list;
        if(!svo_initialize){
            svo_list.push_back(state);        
            if(svo_list.size()>=_MAX_SEG){
                svo_initialize=1;
            }            
        }
        else{
            svo_list.erase(svo_list.begin());
            svo_list.push_back(state);  

            int bezier_flag = svopredict.TrackingGeneration(5,5,svo_list);
            if(bezier_flag==0){
                svo_predict_list = svopredict.getStateListFromBezier(_PREDICT_SEG);
                g_predictedTrajectory = svo_predict_list;
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

double round_to_decimal_places(double value, int decimal_places) {
    double factor = std::pow(10.0, decimal_places);
    return std::round(value * factor) / factor;
}

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

void imageCallback(const sensor_msgs::ImageConstPtr& msg){
    try
    {
        // 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 格式
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        // 将转换后的图像赋值给全局变量
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

double radians_to_degrees(double radians) {
    return radians * 180.0 / M_PI;
}

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
    double z_degrees = radians_to_degrees(z);

    // 如果 z 角度大于 180，则减去 360，确保角度在 [-180, 180] 范围内
    if (z_degrees > 180.0) {
        z_degrees -= 360.0;
    }

    return z_degrees;  // 返回绕 Z 轴的角度（度数）
}

void read::reda(ros::NodeHandle& nh) {
    std::cout<<"here"<<std::endl;
    target_pose_pub = nh.advertise<nav_msgs::Odometry>("/target_ekf_odom", 10);
#ifdef DELAY_TEST
    // 发布延迟补偿后的VINS消息
    delayed_vins_pub = nh.advertise<nav_msgs::Odometry>("/vins_delayed", 10);
#endif

    ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/vins_fusion/imu_propagate", 10);
    ros::Timer timer = nh.createTimer(ros::Duration(1.0 / 200.0), 
                                     boost::bind(publishOdometry, _1, boost::ref(odom_pub)));

    ros::Timer gos_timer = nh.createTimer(ros::Duration(1.0 / 30.0), 
                                     boost::bind(gos_pos_pub, _1, boost::ref(target_pose_pub)));

    static KalmanFilter kf_x(0.01, 0.0025);
    static KalmanFilter kf_y(0.01, 0.0025);
    static KalmanFilter kf_z(0.01, 0.0025);
    // MultiKalmanFilter kf_3d(0.01, 0.05, 0.02);

    sleep(6);
    double glo_deg;
    // Eigen::Matrix3d M;
    // M << 0, 0, 1,
    //     -1, 0, 0,
    //      0, -1, 0;
    // Eigen::Matrix3d M1;
    // M1 << 0, -1, 0,
    //       0, 0, -1,
    //       1, 0, 0;

    // camera:
    // x: -inf left->right +inf
    // y: -inf up  ->down  +inf
    // z: -inf back->front   +inf

    // drone:
    // x: -inf back ->front +inf
    // y: -inf right->left  +inf
    // z: -inf down ->up    +inf

    // tag deg :
    // left: green
    // front: red
    // up: blue

    Eigen::Matrix3d M_camera2drone;
    M_camera2drone <<  1, 0, 0,
                       0, -1, 0,
                       0, 0, -1;
    Eigen::Matrix3d M_tag2camera;
    M_tag2camera  <<  0, 1, 0,
                      -1, 0, 0,
                      0, 0, 1;

    nh_ = nh;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    Eigen::Vector3d pre_vel{0.0,0.0,0.0};
    double fx = 734.804843, fy = 734.561878;
    double cx = 590.087740, cy = 422.783965;

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
    // cap.set(cv::CAP_PROP_EXPOSURE, -4);
    cap.set(cv::CAP_PROP_FPS, 30);
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "当前帧率: " << fps << " FPS" << std::endl;
   
#ifdef SAVE_VIDEO
    // 视频保存参数
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

    while (true)
    {
        int triger_ = 8;
        nav_msgs::Odometry odom_msg;

        // Set the frame ID and timestamp
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = "world";
        cap >> frame;

        if(triger_ != 1)
            triger2 = 0;
        
        if (frame.empty()) {
            continue;
        }
        // 转换为灰度图像，非必要
        cv::Mat gray;
        // frame.convertTo(frame, -1, 1.1, 20); // 轻微提高对比度和亮度
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> T1_R1_pair = uav_state_listener.get_T1_R1();
        T1 = T1_R1_pair.first;
        R1 = T1_R1_pair.second;
        
#ifdef DELAY_TEST
        // 发布延迟补偿后的VINS消息
        nav_msgs::Odometry delayed_vins_msg;
        delayed_vins_msg.header.stamp = ros::Time::now();
        delayed_vins_msg.header.frame_id = "world";
        delayed_vins_msg.child_frame_id = "vins_delayed";
        
        // 设置位置
        delayed_vins_msg.pose.pose.position.x = T1[0];
        delayed_vins_msg.pose.pose.position.y = T1[1];
        delayed_vins_msg.pose.pose.position.z = T1[2];
        
        // 设置旋转（从旋转矩阵转换为四元数）
        Eigen::Quaterniond delayed_q(R1);
        delayed_vins_msg.pose.pose.orientation.w = delayed_q.w();
        delayed_vins_msg.pose.pose.orientation.x = delayed_q.x();
        delayed_vins_msg.pose.pose.orientation.y = delayed_q.y();
        delayed_vins_msg.pose.pose.orientation.z = delayed_q.z();
        
        // 发布消息
        delayed_vins_pub.publish(delayed_vins_msg);
#endif
        cv::putText(frame, std::string("vins(delayed): ") + 
                    "x=" + std::to_string(T1[0]) + 
                    " y=" + std::to_string(T1[1]) + 
                    " z=" + std::to_string(T1[2]),
                    cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);


        if (markerIds.size() > 0)
        {
            flag1+=1;

            cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            std::vector<cv::Vec3d> rvecs1, tvecs1;
            std::vector<cv::Vec3d> rvecs0, tvecs0;
            std::vector<cv::Vec3d> rvecs2, tvecs2;

            // 检查是否包含29号二维码
            bool hasMarkerMain = false;
            for (int i = 0; i < markerIds.size(); ++i) {
                if (markerIds[i] == 29) {
                    hasMarkerMain = true;
                    break;
                }
            }

            // 如果有29号二维码，只计算0.15的尺寸
            if (hasMarkerMain) {
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, cameraMatrix, distCoeffs, rvecs1, tvecs1);
            } else {
                // 如果没有29号二维码，计算所有尺寸
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.0165, cameraMatrix, distCoeffs, rvecs0, tvecs0);
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.15, cameraMatrix, distCoeffs, rvecs1, tvecs1);
                cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.06, cameraMatrix, distCoeffs, rvecs2, tvecs2);
            }

            Eigen::Vector3d position{0, 0, 0};
            Eigen::Vector3d averagePosition{0, 0, 0};
            double averageDeg = 0;
            double weight_count = 0;

            for (int i = 0; i < markerIds.size(); ++i)
            {
                int currentMarkerId = markerIds[i];  // 获取当前标签ID
                
                // 如果有29号二维码，跳过其他二维码的处理
                if (hasMarkerMain && (currentMarkerId != 29)) {
                    continue;
                }
                
                Eigen::Vector3d T2;
                Eigen::Matrix3d R2;
                cv::Vec3d rvec;
                cv::Vec3d tvec;
                if(currentMarkerId == 29){
                    rvec = rvecs1[i];
                    tvec = tvecs1[i];
                }
                else if(currentMarkerId == 33){
                    rvec = rvecs0[i];
                    tvec = tvecs0[i];
                }
                else if(currentMarkerId == 0 || currentMarkerId == 1 || currentMarkerId == 2 || currentMarkerId == 3 || currentMarkerId == 4 || currentMarkerId == 5){
                    rvec = rvecs2[i];
                    tvec = tvecs2[i];
                }
                else
                    continue;

                T2 << tvec[0], tvec[1], tvec[2];
                cv::Mat rotationMatrix;
                cv::Rodrigues(rvec, rotationMatrix);
                
                for (int i = 0; i < 3; ++i){
                    for (int j = 0; j < 3; ++j)
                        R2(i, j) = rotationMatrix.at<double>(i, j);}
                //   Eigen::Vector3d T3(T2[2], -T2[0], -T2[1]);
                Eigen::Vector3d T3 = M_camera2drone * T2;
                //   T3[2] += 0.05;
                //位置计算
                R2 = M_camera2drone * M_tag2camera * R2;
                position = R1 * T3 + T1;
#ifdef DELAY_TEST
                position = R1 * T3;
#endif

                //偏航角计算
                Eigen::Matrix3d pose = R1 * R2;
                // Eigen::Matrix3d pose = R2;
                // Eigen::Quaterniond q(correctYaw(pose));
                double deg = rotationMatrixToEulerAngles(pose);
                glo_deg = deg;

                if (currentMarkerId == 29)
                    averageDeg += (deg * 0.8);
                else if (currentMarkerId == 33)
                    averageDeg += (deg * 0.5);
                else if (currentMarkerId == 0 || currentMarkerId == 1 || currentMarkerId == 2 || currentMarkerId == 3 || currentMarkerId == 4 || currentMarkerId == 5)
                    averageDeg += (deg * 0.65);

                if (currentMarkerId == 29){
                    position.x() = position.x() + 0.0*std::sin(fin_deg);
                    position.y() = position.y() - 0.0*std::cos(fin_deg);
                    averagePosition += (position * 0.8);
                }
                else if (currentMarkerId == 33){
                    position.x() = position.x() + 0.0*std::sin(fin_deg);
                    position.y() = position.y() - 0.0*std::cos(fin_deg);
                    averagePosition += (position * 0.5);
                }
                else if (currentMarkerId == 0){
                    position.x() = position.x() - 0.205*std::cos(fin_deg) + 0.14*std::sin(fin_deg);
                    position.y() = position.y() - 0.205*std::sin(fin_deg) - 0.14*std::cos(fin_deg);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 1){
                    position.x() = position.x() + 0.205*std::cos(fin_deg) + 0.14*std::sin(fin_deg);
                    position.y() = position.y() + 0.205*std::sin(fin_deg) - 0.14*std::cos(fin_deg);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 2){
                    position.x() = position.x() + 0.205*std::cos(fin_deg) - 0.14*std::sin(fin_deg);
                    position.y() = position.y() + 0.205*std::sin(fin_deg) + 0.14*std::cos(fin_deg);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 3){
                    position.x() = position.x() - 0.205*std::cos(fin_deg) - 0.14*std::sin(fin_deg);
                    position.y() = position.y() - 0.205*std::sin(fin_deg) + 0.14*std::cos(fin_deg);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 4){
                    position.x() = position.x() + 0.14*std::sin(fin_deg);
                    position.y() = position.y() - 0.14*std::cos(fin_deg);
                    averagePosition += (position * 0.65);
                }
                else if (currentMarkerId == 5){
                    position.x() = position.x() + 0.14*std::sin(fin_deg);
                    position.y() = position.y() - 0.14*std::cos(fin_deg);
                    averagePosition += (position * 0.65);
                }

                if(currentMarkerId == 29)
                    weight_count += 0.8;
                else if(currentMarkerId == 33)
                    weight_count += 0.5;
                else if(currentMarkerId == 0 || currentMarkerId == 1 || currentMarkerId == 2 || currentMarkerId == 3 || currentMarkerId == 4 || currentMarkerId == 5)
                    weight_count += 0.65;

            }

            double this_deg = averageDeg / weight_count;
            if (std::fabs(this_deg-fin_deg) > 3 &&
            std::fabs(this_deg-fin_deg + 360) > 3 &&
            std::fabs(this_deg-fin_deg - 360) > 3)
            {
                // 这里不再使用mf1滤波，改为类似callback中yaw offset的滤波方式
                double delta_yaw = this_deg * M_PI / 180 - fin_deg;
                // 处理角度环绕问题
                if (delta_yaw > M_PI) {
                    delta_yaw -= 2 * M_PI;
                } else if (delta_yaw < -M_PI) {
                    delta_yaw += 2 * M_PI;
                }
                fin_deg += 0.5 * delta_yaw; // 0.3为滤波系数，可根据实际调整
            }

            if(fin_deg > M_PI)
                fin_deg -= 2 * M_PI;
            else if(fin_deg < -M_PI)
                fin_deg += 2 * M_PI;

            position = averagePosition / weight_count;
            if((ros::Time::now() - target_start_time).toSec() > 1.0)
              {
                  pos.clear();
              }
            pos.push_back({position[0], position[1], position[2]});
            target_start_time = ros::Time::now();

            if(pos.size() > 3){
                pos.erase(pos.begin());
                Eigen::Vector3d sum(0.0, 0.0, 0.0);
                for (const auto& p : pos) {
                    sum += p;
                }
        
                avg = sum / pos.size();
            }


            //机械狗标定
            if(map_triger>24){
                gos_p0 =  gos_pos;
                gos_vins_p0 = position;
                if(pos.size() >= 3)
                    gos_vins_p0 = avg;
                // if(flow_z > 0.3)
                //     gos_vins_p0.z() = T1[2] - (flow_z - 0.41);
                gos_vins_yaw0 = fin_deg;
                gos_yaw0 = gos_yaw;
                d_yaw = gos_vins_yaw0 - gos_yaw0 ;
                Eigen::Vector3d reflect;
                std::vector<Eigen::Vector3d> reflect_kid;
                reflect.x() = d_yaw;
                reflect.y() = d_yaw;
                reflect.z() = d_yaw;
                reflect_kid.push_back(gos_p0);
                reflect_kid.push_back(gos_vins_p0);
                reflect_kid.push_back(reflect);
                reflect_list.push_back(reflect_kid);
                if (reflect_list.size() > 10) {
                    reflect_list.erase(reflect_list.begin());
                }
                gos_triger = 1;
                map_triger = 0;
              }
              map_triger+=1;


            //降落位置识别
              averge_vel = {0, 0, 0};
              glo_pos = position;
              if (pos.size() >= 3 || true) {
                    double filtered_x = kf_x.filter(avg.x());
                    double filtered_y = kf_y.filter(avg.y());
                    double filtered_z = kf_z.filter(avg.z());
                    // double filtered_x = position.x();
                    // double filtered_y = position.y();
                    // double filtered_z = position.z();

                    // 基于VINS的机体朝向，将最终位置向机体左侧偏移2cm
                    // 从VINS旋转矩阵R1获取yaw（机体朝向，世界系）
                    double yaw_vins = std::atan2(R1(1,0), R1(0,0));
                    filtered_x -= 0.02 * std::sin(yaw_vins);
                    filtered_y += 0.02 * std::cos(yaw_vins);

                    odom_msg.pose.pose.position.x = round_to_decimal_places(filtered_x, 2);
                    odom_msg.pose.pose.position.y = round_to_decimal_places(filtered_y, 2);
                    odom_msg.pose.pose.position.z = round_to_decimal_places(filtered_z, 2);
                    odom_msg.twist.twist.linear.x = 1.0*averge_v[0];
                    odom_msg.twist.twist.linear.y = 1.0*averge_v[1];
                    odom_msg.twist.twist.linear.z = 0;
                    odom_msg.pose.pose.orientation.w = fin_deg;
                    odom_msg.pose.pose.orientation.x = 0;
                    odom_msg.pose.pose.orientation.y = 0;
                    odom_msg.pose.pose.orientation.z = 0;                    

                    static bool initialize = false;
                    static std::vector<Eigen::Matrix<double,6,1>> predict_state_list;
                    Eigen::Vector4d state(odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y,odom_msg.pose.pose.position.z,ros::Time::now().toSec());
                    if(!initialize){
                        target_detect_list.push_back(state);        
                        if(target_detect_list.size()>=_MAX_SEG){
                            initialize=1;
                        }            
                    }
                    else{
                        target_detect_list.erase(target_detect_list.begin());
                        target_detect_list.push_back(state);  

                        int bezier_flag = tgpredict.TrackingGeneration(5, 5, target_detect_list);  // 从(5,5)改为(3,3)，减少控制点和预测段数，提高响应速度
                        if(bezier_flag==0){
                            predict_state_list = tgpredict.getStateListFromBezier(_PREDICT_SEG);
                            odom_msg.twist.twist.linear.x = 1.0*predict_state_list[2](3);
                            odom_msg.twist.twist.linear.y = 1.0*predict_state_list[2](4);
                            odom_msg.twist.twist.linear.z = 0;
                            pre_vel[0] = predict_state_list[0](3);
                            pre_vel[1] = predict_state_list[0](4);
                        }      
                    }
                    target_pose_pub.publish(odom_msg);

                }
                // 绘制相对位姿
            for (int i = 0; i < markerIds.size(); ++i)
            {
                int currentMarkerId1 = markerIds[i];  // 获取当前标签ID
                cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs1[i], tvecs1[i], 0.1);
            }
        }

            

        // 显示速度 averge_v
        cv::putText(frame, std::string("c_pos: ") + 
                    "x=" + std::to_string(avg[0]-vins_p[0]) + 
                    " y=" + std::to_string(avg[1]-vins_p[1]) + 
                    " z=" + std::to_string(avg[2]-vins_p[2]),
                    cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        // 显示 vins 的位姿 T1
        
        cv::putText(frame, std::string("deg: ") + std::to_string(glo_deg),
                    cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::putText(frame, std::string("last_deg: ") + std::to_string(last_deg),
                    cv::Point(10, 230), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::putText(frame, std::string("pre_vel: ") + 
                    "x=" + std::to_string(pre_vel[0]) + 
                    " y=" + std::to_string(pre_vel[1]) + 
                    " z=" + std::to_string(pre_vel[2]),
                    cv::Point(10, 270), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

# ifdef SAVE_VIDEO
        out.write(frame);
# endif
# ifdef SCREEN_SHOW
        // 显示图像
        cv::imshow("ArUco Detection", frame);
# endif
        // 按下 'q' 键退出循环
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

# ifdef SAVE_VIDEO
    // 释放资源
    out.release();
    cv::destroyAllWindows();
# endif
}

void read::onInit() {
    // std::cout<<"begin"<<std::endl;
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    initThread_ = std::thread(std::bind(&read::reda, this, nh));
  }

read::~read() {
    if (initThread_.joinable()) {
        initThread_.join();  // 等待线程结束，防止资源泄漏
    }
}

}//namespace
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(image::read, nodelet::Nodelet);
// #include <pluginlib/class_list_macros.h>
// PLUGINLIB_EXPORT_CLASS(image::read, nodelet::Nodelet);