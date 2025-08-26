#ifndef READ_H
#define READ_H

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "quadrotor_msgs/PositionCommand.h"
#include "quadrotor_msgs/TakeoffLand.h"
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <Eigen/Core>
#include <atomic>
#include <thread>
#include <cstdlib>
#include <algorithm>

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
#include <geometry_msgs/PoseStamped.h>

#include<bezier_predict.h>

using namespace cv;

// #define SIMULATE
# define SAVE_VIDEO
# define SCREEN_SHOW

namespace image{
class MedianFilter1 {
public:
    // 构造函数，设置窗口大小
    MedianFilter1(size_t size = 3);

    // 更新函数，传入新值，返回当前窗口的中值
    double update(double new_value);
private:
    size_t size;  // 窗口大小
    std::deque<double> window;  // 存储窗口内的值
};


class MultiKalmanFilter {
    public:
        MultiKalmanFilter(double pos_process_var, double pos_meas_var, double dist_meas_var);
    
        Eigen::Vector3d filter(const Eigen::Vector3d& pos_meas, double distance_meas); 
    
    private:
        Eigen::Vector3d state_;
        Eigen::Matrix3d covariance_;
        Eigen::Matrix3d process_noise_;
        Eigen::Matrix3d pos_meas_noise_;
        double dist_meas_noise_;
        bool is_initialized_;
    };

class UAVStateListener1 {
public:
    UAVStateListener1() ;

    void pose_cb(const nav_msgs::Odometry::ConstPtr& msg) ;
    void AOA_callback(const nav_msgs::Odometry::ConstPtr& msg) ;
    void flow_callback(const nav_msgs::Odometry::ConstPtr& msg) ;
#ifndef SIMULATE
    void svo_callback(const boost::shared_ptr<const geometry_msgs::PoseWithCovarianceStamped>& msg);
#else
    void svo_callback(const boost::shared_ptr<const geometry_msgs::PoseStamped>& msg);
#endif
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> get_T1_R1() const ;

private:
    ros::NodeHandle nh;
    ros::Subscriber svo_sub;
    ros::Subscriber pose_sub;
    ros::Subscriber AOA_sub;
    ros::Subscriber flow_sub;

    bool trigger_condition_met;
    int c_flag;
    Eigen::Vector3d T1;
    Eigen::Matrix3d R1;
    std::vector<Eigen::Vector3d> list;

    std::vector<Eigen::Vector4d> svo_list;
    Bezierpredict svopredict;
};

class gos_listener {
public:
    gos_listener() ;

    void pose_cb(const nav_msgs::Odometry::ConstPtr& msg) ;
    void pose_cb_land(const geometry_msgs::PoseStamped::ConstPtr& msg) ;

private:
    ros::NodeHandle nh;
    ros::Subscriber pose_sub;
    ros::Subscriber land_sub;

    bool trigger_condition_met;
    int c_flag;
    Eigen::Vector3d T1;
    Eigen::Matrix3d R1;
    std::vector<Eigen::Vector3d> list;
};


class read : public nodelet::Nodelet{
private:
    ros::NodeHandle nh_;
    std::thread initThread_;
    UAVStateListener1 uav_state_listener;
    gos_listener gos_sub;
    std::vector<Eigen::Vector3d> pos; // 定义pos的存储结构
    Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d T1{0.0, 0.0, 0.0};
    Eigen::Vector3d T8{0.0, 0.0, 0.0};
    Eigen::Vector3d position{0.0, 0.0, 0.0};
    Eigen::Matrix3d pose = Eigen::Matrix3d::Identity(); 
    MedianFilter1 mf1;
    ros::Publisher target_pose_pub;
    void reda(ros::NodeHandle& nh);

    
    Bezierpredict tgpredict;
    std::vector<Eigen::Vector4d> target_detect_list; 

    double im_kp = 0;
public:
    void onInit();
    ~read();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class KalmanFilter {
    public:
        KalmanFilter(double process_variance, double measurement_variance);
            
    
        double filter(double measurement);
    
    private:
        double process_variance_;
        double measurement_variance_;
        bool is_initialized_;
        double state_;
        double covariance_;
    };
    

double round_to_decimal_places(double value, int decimal_places);

double correlation_coefficient(const std::vector<double>& times, const std::vector<double>& data);

void imageCallback(const sensor_msgs::ImageConstPtr& msg);

bool isRotationMatrix(const Eigen::Matrix3d& R);

double radians_to_degrees(double radians);

double rotationMatrixToEulerAngles(const Eigen::Matrix3d& R);

void publishOdometry(const ros::TimerEvent&, ros::Publisher &odom_pub) ;

std::vector<double> predictPositions(double pos1, double pos2);

Eigen::Matrix3d correctYaw(const Eigen::Matrix3d& pose);
}
#endif