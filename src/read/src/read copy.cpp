//#include <read.h>

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

#include<bezier_predict.h>

#include <fstream>   // 用于文件操作  // 用于 std::vector
#include <string>

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
int triger2 = 0;

MedianFilter1::MedianFilter1(size_t size) : size(size) {}

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

TagVelocityEstimator1::TagVelocityEstimator1()
        : initial_time(0.0),
          last_msg_time(ros::Time::now().toSec()) ,
          timeout_duration(0.8),
          max_samples(10){

        // 初始化定时器
        timer = nh.createTimer(ros::Duration(timeout_duration), &TagVelocityEstimator1::check_timeout, this);
        timer1 = nh.createTimer(ros::Duration(1.0), &TagVelocityEstimator1::set_trigger, this);
    }

void TagVelocityEstimator1::pose_callback() {

        // 获取当前时间
        double current_time = ros::Time::now().toSec();
        triger2 = 1;
        // 如果是第一次接收消息，则设置初始时间
        if (initial_time == 0.0) {
            initial_time = current_time;
        }

        // 更新最后接收消息的时间
        last_msg_time = current_time;

        // 计算相对于初始时间的时间差
        double time_diff = current_time - initial_time;
        // std::cout<<"time_diff"<<time_diff<<std::endl;

        // 记录位置
        Eigen::Vector3d position = glo_pos;
        positions.push_back(position);
        times.push_back(time_diff);

        // 保留最新的 `max_samples` 个样本
        if (positions.size() > max_samples) {
            positions.erase(positions.begin());
            times.erase(times.begin());
        }

        // 如果有足够的样本，则计算速度
        if (positions.size() == max_samples) {
            compute_and_publish_velocity();
        }
    }

void TagVelocityEstimator1::compute_and_publish_velocity() {

        // 提取每个轴的数据
        std::vector<double> x_data, y_data, z_data;
        for (const auto& pos : positions) {
            x_data.push_back(pos.x());
            y_data.push_back(pos.y());
            z_data.push_back(pos.z());
        }
        // 计算各轴的速度斜率
        double x_slope = least_squares_slope(times,x_data);
        double y_slope = least_squares_slope(times,y_data);
        double z_slope = least_squares_slope(times,z_data);
        // 计算拟合值
        // 计算相关系数 R²
        double r_squared_x = correlation_coefficient(times,x_data);
        double r_squared_y = correlation_coefficient(times,y_data);
        // 将速度值保存在 glo_vel 中
        // std::cout<<"x_v:"<<x_slope<<std::endl;
        // std::cout<<"y_v:"<<y_slope<<std::endl;
        // std::cout<<"x_r:"<<r_squared_x<<std::endl;
        // std::cout<<"y_r:"<<r_squared_y<<std::endl;
        glo_vel(0) = (std::fabs(r_squared_x) > 0.6) ? x_slope : glo_vel(0);
        glo_vel(1) = (std::fabs(r_squared_y) > 0.5) ? y_slope : glo_vel(1);
        glo_vel(2) = z_slope; // 根据具体需要可以设置不同的阈值
        glo_vel = 0.7*last_vel + 0.3*glo_vel;
        last_vel = glo_vel;

        // 速度阈值，低于该值的速度视为 0
        double velocity_threshold = 0.35;

        for (int i = 0; i < 3; ++i) {
            if (std::abs(glo_vel(i)) < velocity_threshold) {
                glo_vel(i) = 0.0;
            }
        }
    }

void TagVelocityEstimator1::check_timeout(const ros::TimerEvent& event) {


        double current_time = ros::Time::now().toSec();
        if (current_time - last_msg_time > timeout_duration) {
            // 超时，没有新的消息，清空列表
            positions.clear();
            times.clear();
            initial_time = 0;
            averge_v = Eigen::Vector3d(0,0,0);
            // pos_copy.clear(); // 视情况使用
        }
    }

void TagVelocityEstimator1::set_trigger(const ros::TimerEvent& event) { 
        

        double current_time = ros::Time::now().toSec();
        if (current_time - last_msg_time > 0.4 && triger2 == 1) {
            ros::param::set("/drone0/planning/triger_", 0.0);
            triger2 = 0;
        }
    }

VelocitySubscriber1:: VelocitySubscriber1() 
        : velocities(3, 0.0), average_velocity(3, 0.0),
          sample_count(0), cumulative_time(0.0), flag(0), last_vel(0.0) {
        last_update_time = std::chrono::steady_clock::now();
    }

void VelocitySubscriber1::callback() {
        std::lock_guard<std::mutex> lock(global_mutex); // 确保多线程安全

        // 提取线速度数据
        std::vector<double> current_velocities = {
            round(glo_vel(0) * 100.0) / 100.0,
            round(glo_vel(1) * 100.0) / 100.0,
            round(glo_vel(2) * 100.0) / 100.0
        };

        if (sample_count < 3) {
            // 收集3次速度样本
            velocity_samples.push_back(current_velocities);
            sample_count++;

            if (sample_count == 3) {
                // 计算平均速度
                for (int i = 0; i < 3; ++i) {
                    average_velocity[i] = 0.0;
                    for (const auto& sample : velocity_samples) {
                        average_velocity[i] += sample[i];
                    }
                    average_velocity[i] /= 3.0;
                }

                // 发布ROS参数
                ros::param::set("/drone0/planning/perching_vx", round_to_decimal_places(average_velocity[0] ,2));
                ros::param::set("/drone0/planning/perching_vy", round_to_decimal_places(average_velocity[1] ,2));
                ros::param::set("/drone0/planning/perching_vz", 0.00);

                // 更新全局变量
                averge_v = Eigen::Vector3d(average_velocity[0], average_velocity[1], 0.0);

                // 清空样本和计数器
                velocity_samples.clear();
                sample_count = 0;
            }
        }

        // 获取当前时间（当前未使用该变量）
    }

void VelocitySubscriber1::change() {
        cumulative_time = 0;
        sample_count = 0;
    }

UAVStateListener1::UAVStateListener1() 
        : trigger_condition_met(false),  T1(Eigen::Vector3d::Zero()), R1(Eigen::Matrix3d::Identity()) {
        //pose_sub = nh.subscribe<nav_msgs::Odometry>("/vins_fusion/imu_propagate", 3, &UAVStateListener1::pose_cb, this);
        pose_sub = nh.subscribe<nav_msgs::Odometry>("/vins_fusion/imu", 3, &UAVStateListener1::pose_cb, this);
    }

void UAVStateListener1::pose_cb(const nav_msgs::Odometry::ConstPtr& msg) {
        Eigen::Vector3d current_position(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        list.push_back(current_position);

        if (list.size() == 5) {
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
            tf::Matrix3x3 tf_matrix(quat);

// 将 tf::Matrix3x3 转换为 Eigen::Matrix3d
            R1 << tf_matrix[0][0], tf_matrix[0][1], tf_matrix[0][2],
                tf_matrix[1][0], tf_matrix[1][1], tf_matrix[1][2],
                tf_matrix[2][0], tf_matrix[2][1], tf_matrix[2][2];

            // 触发条件达成
            trigger_condition_met = true;

            // 清空列表
            list.clear();
        }
    }

std::pair<Eigen::Vector3d, Eigen::Matrix3d>  UAVStateListener1::get_T1_R1() const {
        return std::make_pair(T1, R1);
    }

double round_to_decimal_places(double value, int decimal_places) {
    double factor = std::pow(10.0, decimal_places);
    return std::round(value * factor) / factor;
}

double least_squares_slope(const std::vector<double>& times, const std::vector<double>& data) {
    size_t n = data.size();
    if (n < 2 || times.size() != n) {
        throw std::invalid_argument("Data size too small or time size does not match data size");
    }

    double sum_x = std::accumulate(times.begin(), times.end(), 0.0);
    double sum_y = std::accumulate(data.begin(), data.end(), 0.0);
    double sum_xy = 0.0;
    double sum_xx = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_xy += times[i] * data[i];
        sum_xx += times[i] * times[i];
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    
    return slope;
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
        return 0;
        throw std::runtime_error("Denominator is zero, cannot calculate correlation coefficient.");
    }

    double r = numerator / denominator;
    return r;
}

void set_ros_param(const std::string& param_key, float param_value) {
    ros::NodeHandle nh;
    nh.setParam(param_key, param_value);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
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
    double z_degrees = radians_to_degrees(z)+180;

    // 如果 z 角度大于 180，则减去 360，确保角度在 [-180, 180] 范围内
    if (z_degrees > 180.0) {
        z_degrees -= 360.0;
    }

    return z_degrees;  // 返回绕 Z 轴的角度（度数）
}

void read::reda(ros::NodeHandle& nh) {
    // std::cout<<"here"<<std::endl;
    target_pose_pub = nh.advertise<nav_msgs::Odometry>("/target_ekf_odom", 10);
    double glo_deg;
    Eigen::Matrix3d M;
    M << 0, 0, 1,
        -1, 0, 0,
         0, -1, 0;
    Eigen::Matrix3d M1;
    M1 << 0, -1, 0,
          0, 0, -1,
          1, 0, 0;
    Eigen::Matrix3d M2;
    std::vector<double> myVector_y = {};
    std::vector<double> myVector_x = {};
    M2 << 1, 0, 0,
          0, -1, 0,
          0, 0, -1;
    nh_ = nh;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    parameters ->adaptiveThreshConstant = 3;
    parameters ->minMarkerPerimeterRate = 0.01;
    // 读取相机标定参数
    //cv::FileStorage fs("calibration.xml", cv::FileStorage::READ);
    //cv::Mat cameraMatrix, distCoeffs;
    //fs["camera_matrix"] >> cameraMatrix;
    //fs["distortion_coefficients"] >> distCoeffs;
    // double fx = 927.7442016601562, fy =  926.8687744140625;
    // double cx = 658.57080078125, cy =  361.7791442871094;
    double fx = 193.99, fy = 193.99;
    double cx = 336.0, cy = 188.0;
    // double fx = 601.7569580078125;
    // double fy = 601.7569580078125;
    // double cx = 314.7174987792969;
    // double cy = 241.04949951171875;
    // double k1 = 0.039106;
    // double k2 = -0.056494;
    // double p1 = -0.000824;
    // double p2 = 0.092161;
    // double k3 = 0.0;

    double k1 = 0.0;
    double k2 = 0.0;
    double p1 = 0.0;
    double p2 = 0.0;
    double k3 = 0.0;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
    fx, 0, cx,
    0, fy, cy,
    0, 0, 1);

    cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);


    // 打开相机
    // std::cout << "sblc" << std::endl;
    auto time_now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm* tm_now = std::localtime(&time_now);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", tm_now);
    std::string time_video(buffer);
    std::string unique_filename = "/home/ros/Fast-Perching/output_video_" + time_video + ".avi";

    // 视频保存参数
    int fourcc_video = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter out(unique_filename, fourcc_video, 30.0, cv::Size(1280, 720));
    //ros::Subscriber sub = nh.subscribe("/camera/color/image_raw", 1, imageCallback);  
    //ros::Subscriber sub = nh.subscribe("/camera/infra1/image_rect_raw", 1, imageCallback);      
    //change
    
    ros::Subscriber sub = nh.subscribe("/airsim/front_right_custom/color", 1, imageCallback);

    // 检查视频输出是否打开成功
    if (!out.isOpened()) {
        std::cerr << "无法保存视频文件" << std::endl;
        exit(1);
    }

    Eigen::Vector3d T3_1;
    while (true)
    {
        // std::cout << "sblc" << std::endl;
        //change
        int triger_ = 8;
        nav_msgs::Odometry odom_msg;

        // Set the frame ID and timestamp
        odom_msg.header.stamp = ros::Time::now();
        odom_msg.header.frame_id = "world";

        // Set the position (example values)
       
        //nh.getParam("/drone0/planning/triger_", triger_);
        // 读取相机帧
        if(triger_ != 1)
            triger2 = 0;
        
        if (frame.empty()) {
            continue;
        }
        // 转换为灰度图像，非必要
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    //     cv::Mat sharpening_kernel = (cv::Mat_<float>(3, 3) <<
    //     0, -1, 0,
    //     -1, 5, -1,
    //     0, -1, 0);

    // // 对图像进行锐化处理
    //     cv::Mat sharpened;
    //     cv::filter2D(gray, sharpened, -1, sharpening_kernel);
        // 检测ArUco二维码
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        //cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> T1_R1 = uav_state_listener.get_T1_R1();
        Eigen::Vector3d TT= T1_R1.first;
        cv::putText(frame, std::string("vins: ") + 
                    "x=" + std::to_string(TT[0]) + 
                    " y=" + std::to_string(TT[1]) + 
                    " z=" + std::to_string(TT[2]),
                    cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        // 绘制检测结果
        if (markerIds.size() > 0)
        {
            flag1+=1;
            // std::cout<<"flag1"<<std::endl;
            std::pair<Eigen::Vector3d, Eigen::Matrix3d> T1_R1_pair = uav_state_listener.get_T1_R1();
            // 手动解包
            T1 = T1_R1_pair.first;
            R1 = T1_R1_pair.second;
            //change
            R1 = M2 * R1 * M2;

            // std::cout<<"T1"<<T1<<std::endl;

            cv::aruco::drawDetectedMarkers(frame, markerCorners, markerIds);

            // 估计相机姿态
            /*********************************************************
            void cv::aruco::estimatePoseSingleMarkers(
                const std::vector<std::vector<cv::Point2f>>& corners,  // 检测到的 ArUco 标记角点的二维图像坐标
                float markerLength,                                      // ArUco 标记的边长（单位任意，但与标定过程中的相机参数保持一致）
                const cv::Mat& cameraMatrix,                             // 相机的内参数矩阵
                const cv::Mat& distCoeffs,                               // 相机的畸变参数
                std::vector<cv::Vec3d>& rvecs,                           // 输出的每个标记的旋转向量
                std::vector<cv::Vec3d>& tvecs                            // 输出的每个标记的平移向量
            );
            *********************************************************/
            std::vector<cv::Vec3d> rvecs, tvecs;
            // float arucoLength = 0.10; //aruco二维码边长
            float arucoLength = 0.30;
            cv::aruco::estimatePoseSingleMarkers(markerCorners, arucoLength, cameraMatrix, distCoeffs, rvecs, tvecs);

            for (int i = 0; i < markerIds.size(); ++i)
            {
              int currentMarkerId = markerIds[i];  // 获取当前标签ID
              std::cout << "当前检测到的ArUco标签ID: " << currentMarkerId << std::endl;
              if(currentMarkerId != 6)
                continue;
              Eigen::Vector3d T2;
              Eigen::Matrix3d R2;
              cv::Vec3d rvec = rvecs[i];
              cv::Vec3d tvec = tvecs[i];
              T2 << tvec[0], tvec[1], tvec[2];
              cv::Mat rotationMatrix;
              cv::Rodrigues(rvec, rotationMatrix);
              for (int i = 0; i < 3; ++i){
                for (int j = 0; j < 3; ++j)
                  R2(i, j) = rotationMatrix.at<double>(i, j);}
              Eigen::Vector3d T3(T2[2], -T2[0], -T2[1]);
            //   T3[1] += 0.05;
            //   double distance = sqrt(T3[0]*T3[0] + T3[1]*T3[1]);
            //   double dif_d = -0.0128*distance*distance + (-0.0478 * distance) + 0.0051;
            //   T3[1] -= dif_d;
            //   myVector_y.push_back(T3[1]);
            //   myVector_x.push_back(T1[0]);
              T3_1 = T3;
              R2 = M * R2 * M1;
              // 计算标签的世界坐标
              Eigen::Vector3d position = R1 * T3 + T1;

              // 调整位置信息
               position[0] -= 0.08;
            //   position[1] += 0.08;
              //position[2] -= 0.10;
              
              Eigen::Matrix3d pose = R1 * R2;
              double deg = rotationMatrixToEulerAngles(pose);
              glo_deg = deg;
              double fin_deg = 0.35*last_deg + 0.65*deg;
              fin_deg = mf1.update(fin_deg);
            //   if (std::fabs(fin_deg-last_deg)>20)
            //     fin_deg = last_deg;
              if(std::fabs(fin_deg-last_deg)<3)
                fin_deg = last_deg;
              else
                last_deg = fin_deg;
            
              if(flag1%10 == 0)
                nh.setParam("/drone0/planning/target_yaw_deg", round_to_decimal_places(fin_deg,4));
            //   std::cout<<"deg:"<<last_deg<<std::endl;
              averge_vel = {position[0], position[1], position[2]};

              pos.push_back({position[0], position[1], position[2]});
              glo_pos = position;
              // std::cout<<"position:"<<glo_pos<<std::endl;
              v_sub.pose_callback();
              velocity_subscriber.callback();
              if (flag1 % 1 == 0) {
                    std::vector<double> avg_pos(3, 0.0);
                    for (const auto &p : pos) {
                        avg_pos[0] += p[0];
                        avg_pos[1] += p[1];
                        avg_pos[2] += p[2];
                    }
                    avg_pos[0] /= pos.size();
                    avg_pos[1] /= pos.size();
                    avg_pos[2] /= pos.size();
                    if (pos_copy.size() > 0) {
                        avg_pos[0] = 0.2 * pos_copy[0] + 0.8 * avg_pos[0];
                        avg_pos[1] = 0.2 * pos_copy[1] + 0.8 * avg_pos[1];
                        avg_pos[2] = 0.2 * pos_copy[2] + 0.8 * avg_pos[2];
                    }
                    odom_msg.pose.pose.position.x = round_to_decimal_places(avg_pos[0], 2);
                    odom_msg.pose.pose.position.y = round_to_decimal_places(avg_pos[1], 2);
                    odom_msg.pose.pose.position.z = round_to_decimal_places(avg_pos[2], 2);
                    odom_msg.twist.twist.linear.x = 1.0*averge_v[0];
                    odom_msg.twist.twist.linear.y = 1.0*averge_v[1];
                    odom_msg.twist.twist.linear.z = 0;
                    //odom_msg.pose.pose.position.z = 0.0;

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

                        int bezier_flag = tgpredict.TrackingGeneration(5,5,target_detect_list);
                        if(bezier_flag==0){
                            predict_state_list = tgpredict.getStateListFromBezier(_PREDICT_SEG);
                            odom_msg.pose.pose.position.x = predict_state_list[0](0);
                            odom_msg.pose.pose.position.y = predict_state_list[0](1);
                            odom_msg.pose.pose.position.z = predict_state_list[0](2);
                            odom_msg.twist.twist.linear.x = 1.0*predict_state_list[0](3);
                            odom_msg.twist.twist.linear.y = 1.0*predict_state_list[0](4);
                            odom_msg.twist.twist.linear.z = 0;
                        }      
                    }
                    


                    target_pose_pub.publish(odom_msg);
                    nh.setParam("/drone0/planning/perching_px", round_to_decimal_places(avg_pos[0], 2));
                    nh.setParam("/drone0/planning/perching_py", round_to_decimal_places(avg_pos[1], 2));
                    nh.setParam("/drone0/planning/perching_pz", round_to_decimal_places(avg_pos[2], 2)-1.0);
                   // nh.setParam("/drone0/planning/perching_pz", 0);

                    pos_copy = avg_pos;
                    pos.clear();
                    if (triger_ == 0){
                        nh.setParam("/drone0/planning/triger_", 1.0);}

                }
              // std::cout << "Marker ID: " << markerIds[i] << std::endl;
              // std::cout << "rvec: " << rvec << std::endl;
              // std::cout << "tvec: " << tvec << std::endl;
            }

            // 绘制相对位姿
            for (int i = 0; i < markerIds.size(); ++i)
            {
                int currentMarkerId1 = markerIds[i];  // 获取当前标签ID
                if(currentMarkerId1 != 6)
                   continue;
                cv::aruco::drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
            }
        }
        cv::putText(frame, std::string("triger_: ") + std::to_string(triger_),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        // 显示平均速度 averge_vel
        cv::putText(frame, std::string("position: ") + 
                    "x=" + std::to_string(averge_vel[0]) + 
                    " y=" + std::to_string(averge_vel[1]) + 
                    " z=" + std::to_string(averge_vel[2]),
                    cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        // 显示速度 averge_v
        cv::putText(frame, std::string("vel: ") + 
                    "x=" + std::to_string(averge_v[0]) + 
                    " y=" + std::to_string(averge_v[1]) + 
                    " z=" + std::to_string(averge_v[2]),
                    cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        // 显示 vins 的位姿 T1
        
        cv::putText(frame, std::string("deg: ") + std::to_string(glo_deg),
                    cv::Point(10, 190), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        cv::putText(frame, std::string("last_deg: ") + std::to_string(last_deg),
                    cv::Point(10, 230), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        out.write(frame);
        // 显示图像
        cv::imshow("ArUco Detection", frame);
        // 按下 'q' 键退出循环
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }

    // 释放资源
    out.release();
    cv::destroyAllWindows();
    
    // std::ofstream outFile_y("/home/ros/Fast-Perching/output_y.txt");

    // // 检查文件是否成功打开

    // // 将 vector 中的值写入文件
    // for (size_t i = 0; i < myVector_y.size(); ++i) {
    //     outFile_y << myVector_y[i] << "\n";
    // }

    // // 关闭文件流
    // outFile_y.close();
    // std::ofstream outFile_x("/home/ros/Fast-Perching/output_x.txt");

    // // 检查文件是否成功打开

    // // 将 vector 中的值写入文件
    // for (size_t i = 0; i < myVector_x.size(); ++i) {
    //     outFile_x << -myVector_x[i]-0.15<< "\n";
        
    // }

    // // 关闭文件流
    // outFile_x.close();


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