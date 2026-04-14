#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/TakeoffLand.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>

// #include <traj_opt/poly_traj_utils.hpp>
#include <geometry_msgs/PoseStamped.h>

#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <unordered_map>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <array>
#include <tuple>
#include <fstream>

#include "callback.h"

ros::Publisher pos_cmd_pub_;
ros::Publisher land_pub_;
ros::Publisher mode_pub_;
int land_lock_timer = 0;
ros::Publisher debug_pub_;  // 添加调试信息发布器
ros::Time heartbeat_time_;
ros::Publisher yaw_offset_pub;  // 发布yaw offset差值


// 保存上一次发布的命令速度、位置和加速度，用于加速度和jerk限制
Eigen::Vector3d last_cmd_velocity{0, 0, 0};
Eigen::Vector3d last_cmd_position{0, 0, 0};
Eigen::Vector3d last_cmd_acceleration{0, 0, 0};
bool last_cmd_initialized = false;

// 距离日志，按需懒加载
static std::ofstream distance_log_stream;
static const char kDistanceLogPath[] = "/home/pc/Fast-Drone-250/distance_log.txt";
static std::vector<std::tuple<double, double, double>> distance_history;  // {time, distance, height}

static inline void logDistance(double distance, double height) {
  if (!distance_log_stream.is_open()) {
    distance_log_stream.open(kDistanceLogPath, std::ios::app);
    if (!distance_log_stream.is_open()) {
      ROS_WARN_STREAM("Failed to open distance log file: " << kDistanceLogPath);
      return;
    }
  }
  const double now_sec = ros::Time::now().toSec();
  distance_history.emplace_back(now_sec, distance, height);
  distance_log_stream << std::fixed << now_sec << "," << distance << std::endl;
}

int triger_mode = -1;
Eigen::Vector3d vins_p{0,0,0},vins_v{0,0,0};
double vins_yaw = 0;
// double last_target_yaw = 0;
// double target_yaw = 0;

Eigen::Vector3d target_p{0,0,0},target_v{0,0,0};
double target_dog_yaw = 0;
int last_target_timer = 0;
unsigned int target_count = 0;
unsigned int last_target_count = 0;
unsigned int last_precise_target_count = 0;
bool target_receive = false;
int last_target_loss_timer = 0;
unsigned int last_target_loss_count = 0;

// dog_pos 相关变量（只保留速度信息）
bool hc14_dog_pos_received = false;
unsigned int hc14_dog_pos_count = 0;
unsigned int last_hc14_dog_pos_count = 0;
int last_hc14_dog_pos_timer = 0;  // dog_pos专用的timer

// 处理后的hc14_dog信息
Eigen::Vector3d hc14_dog_vel{0,0,0};
Eigen::Vector3d hc14_dog_pos{0,0,0};
double hc14_dog_yaw = 0.0;
double hc14_dog_yaw_rate = 0.0;  // 狗通信角速度
Eigen::Vector2d hc14_dog_acc{0,0};  // 狗加速度（世界坐标系，x和y）
double hc14_perception_confidence = 0.0;  // 感知置信度（0-1之间）
bool hc14_offset_yaw_ready = false;  // hc14_dog信息是否可用
bool hc14_offset_pos_ready = false;  // hc14_dog位置信息是否可用

double kp = 1.2;
double tracking_dist_ = 1.5;
double target_receive_triger = 1;
double mode_vins_z = 0;
double mode_vins_vel_z = 0;
bool reflight_complete = false;
ros::Time target_lost_time;
Eigen::Vector3d target_lost_p = {0,0,0};
Eigen::Vector3d target_lost_v = {0,0,0};
Eigen::Vector3d last_target_v = {0,0,0};

double AOA_x = 10;
double AOA_w = 0;
double flow_z = -1;
double last_error_x = 0;
double last_error_y = 0;
double last_error_z = 0;
double alpha = 0.3;
double filtered_d_error_x = 0;
double filtered_d_error_y = 0;
double filtered_d_error_z = 0;
double intergral_x = 0;
double intergral_y = 0;
double intergral_z = 0;

double intergral_targetz = 0;
double intergral_targetx = 0;  // 世界系积分项（x方向）
double intergral_targety = 0;  // 世界系积分项（y方向）
double intergral_targetx_body = 0;  // 平台系积分项（x方向）
double intergral_targety_body = 0;  // 平台系积分项（y方向）
double last_error_targetx = 0;
double last_error_targety = 0;
double last_error_targetz = 0;

// 均值滤波器相关变量
std::deque<double> error_targetx_buffer;
std::deque<double> error_targety_buffer;
std::deque<double> error_targetz_buffer;

int target_land_flag = 0;


// 角速度控制相关参数
double yaw_kp = 0.3;  // 偏航角速度控制增益
double max_yaw_rate = 0.8;  // 最大偏航角速度限制 (rad/s)

double yaw_rate_pos_gain = 0.0;  // 角速度前馈增益（侧面分量）
double yaw_rate_vel_gain = 0.0;  // 角速度前馈增益（侧面分量）
double yaw_rate_pos_gain_forward = 0.0;  // 角速度前馈增益（前方分量）
double yaw_rate_vel_gain_forward = 0.00;  // 角速度前馈增益（前方分量）

// 加速度和jerk限制参数
const double max_accel = 1.2;  // 最大加速度限制 (m/s^2)
const double max_jerk = 100.0;   // 最大jerk限制 (m/s^3)
const double accel_dt = 0.01;  // 时间间隔 (s)
const double max_accel_z = 1.0;  // z方向最大加速度限制 (m/s^2)

// 降落速度参数（先快后慢）
const double land_vel_max_high = -0.6;  // 高度较高时的最大下降速度 (m/s)
const double land_vel_max_low = -0.6;   // 高度较低时的最大下降速度 (m/s)
const double land_height_fast = 1.5;    // 高于此高度时使用快速下降 (m)
const double land_height_slow = 0.8;    // 低于此高度时使用慢速下降 (m)

double x_p = 0.3;
double x_i = 0.83;
double x_d = 0.0;
double x_d_max = 0.12;
double v_offset_x = 0.0;
double integral_limit_x = 0.7;
double integral_decay = 0.0;  // 积分衰减系数（1/s），防止积分 windup
double acc_gain_x = 0.0; // 0.6

double y_p = 0.3;
double y_i = 0.83;
double y_d = 0.0;
double y_d_max = 0.12;
double v_offset_y = 0.0;
double integral_limit_y = 0.7;
double acc_gain_y = 0.0; // 0.6

double z_p = 0.25;
double z_i = 0.0;
double z_d = 0.0;
double z_d_max = 0.1;
double integral_limit_z = 0.1;

double mpc_x_p = 0.3;
double mpc_x_i = 0.83;
double mpc_x_d = 0.0;
double mpc_x_d_max = 0.12;

double mpc_y_p = 0.3;
double mpc_y_i = 0.83;
double mpc_y_d = 0.0;
double mpc_y_d_max = 0.12;

double mpc_z_p = 0.3;
double mpc_z_i = 0.0;
double mpc_z_d = 0.0;
double mpc_z_d_max = 0.12;


// 位置控制PID参数，感觉没用
double pos_x_p = 0.0;
double pos_x_i = 0.0;
double pos_x_d = 0.0;
double pos_x_d_max = 0.1;
double pos_integral_limit_x = 0.5;

double pos_y_p = 0.0;
double pos_y_i = 0.0;
double pos_y_d = 0.0;
double pos_y_d_max = 0.1;
double pos_integral_limit_y = 0.5;

double mpc_pos_x_p = 0.0;
double mpc_pos_x_i = 0.0;
double mpc_pos_x_d = 0.0;
double mpc_pos_x_d_max = 0.1;

double mpc_pos_y_p = 0.0;
double mpc_pos_y_i = 0.0;
double mpc_pos_y_d = 0.0;
double mpc_pos_y_d_max = 0.1;


#ifdef SIMULATE
v_offset_x = 0.0;
v_offset_y = 0.0;
#endif

std::vector<double> land_height_limit = {1.2, 1.5};

std::vector<Eigen::Vector3d> target_p_list,target_v_list;

ros::Time traj_start_time;
bool traj_initialized = false;
std::vector<Eigen::Vector3d> trajectory_points;
std::vector<Eigen::Vector3d> trajectory_v_points;
std::vector<Eigen::Vector3d> trajectory_a_points;

Eigen::Vector3d mpc_p{0,0,0},mpc_v{0,0,0},mpc_a{0,0,0};

int BPNN_count = 60;
ros::Time traj_sub_time;

double land_timer = 0;

// 参数定义
const double blsc_lambda = 1.2;          // 平衡收敛速度与稳定性
const double blsc_boundary_layer = 0.5;  // 抑制抖振同时保证精度
const double blsc_max_horiz_vel = 1.5;   // 安全水平速度限制（m/s）
const double blsc_max_vert_vel = 0.8;    // 安全垂直速度限制（m/s）

using namespace Eigen;

struct PIDParams {
  double Kp;
  double Ki;
  double Kd;
};

class BPNeuralNetwork {
  private:
      int input_size, hidden_size, output_size;
      double learning_rate;
      std::vector<std::vector<double>> input_to_hidden_weights;  // ���뵽���ز��Ȩ��
      std::vector<std::vector<double>> hidden_to_output_weights; // ���ز㵽������Ȩ��
      std::vector<double> hidden_bias, output_bias;
      std::vector<double> hidden_output, network_output;
  
  public:
      BPNeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate = 2)
          : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) {
          // ��ʼ��Ȩ�غ�ƫ��
          input_to_hidden_weights = std::vector<std::vector<double>>(input_size, std::vector<double>(hidden_size));
          hidden_to_output_weights = std::vector<std::vector<double>>(hidden_size, std::vector<double>(output_size));
          hidden_bias = std::vector<double>(hidden_size, 0.0);
          output_bias = std::vector<double>(output_size, 0.0);
          hidden_output = std::vector<double>(hidden_size, 0.0);
          network_output = std::vector<double>(output_size, 0.0);
  
          // �����ʼ��Ȩ��
          for (int i = 0; i < input_size; ++i)
              for (int j = 0; j < hidden_size; ++j)
                  input_to_hidden_weights[i][j] = (rand() % 1000) / 1000.0;  // ��ʼ��Ϊ [0, 1] ����������
          for (int i = 0; i < hidden_size; ++i)
              for (int j = 0; j < output_size; ++j)
                  hidden_to_output_weights[i][j] = (rand() % 1000) / 1000.0;  // ��ʼ��Ϊ [0, 1] ����������
      }
  
      // �������sigmoid��
      double sigmoid(double x) {
          return 1.0 / (1.0 + exp(-x));
      }
  
      // ������ĵ�����sigmoid�ĵ�����
      double sigmoid_derivative(double x) {
          return x * (1.0 - x);
      }
  
      // ǰ�򴫲�
      std::vector<double> forward(const std::vector<double>& input) {
          // ���뵽���ز�
          for (int i = 0; i < hidden_size; ++i) {
              hidden_output[i] = 0.0;
              for (int j = 0; j < input_size; ++j)
                  hidden_output[i] += input[j] * input_to_hidden_weights[j][i];
              hidden_output[i] += hidden_bias[i];
              hidden_output[i] = sigmoid(hidden_output[i]);
          }
  
          // ���ز㵽�����
          for (int i = 0; i < output_size; ++i) {
              network_output[i] = 0.0;
              for (int j = 0; j < hidden_size; ++j)
                  network_output[i] += hidden_output[j] * hidden_to_output_weights[j][i];
              network_output[i] += output_bias[i];
              network_output[i] = sigmoid(network_output[i]);
          }
  
          return network_output;
      }
  
      // ���򴫲�
      void backward(const std::vector<double>& input, const std::vector<double>& target) {
          // ��������
          std::vector<double> output_errors(output_size);
          for (int i = 0; i < output_size; ++i)
              output_errors[i] = target[i] - network_output[i];
  
          // ���ز����
          std::vector<double> hidden_errors(hidden_size);
          for (int i = 0; i < hidden_size; ++i) {
              hidden_errors[i] = 0.0;
              for (int j = 0; j < output_size; ++j)
                  hidden_errors[i] += output_errors[j] * hidden_to_output_weights[i][j];
              hidden_errors[i] *= sigmoid_derivative(hidden_output[i]);
          }
  
          // ���������Ȩ��
          for (int i = 0; i < output_size; ++i) {
              for (int j = 0; j < hidden_size; ++j) {
                  hidden_to_output_weights[j][i] += learning_rate * output_errors[i] * hidden_output[j];
              }
              output_bias[i] += learning_rate * output_errors[i];
          }
  
          // �������ز�Ȩ��
          for (int i = 0; i < hidden_size; ++i) {
              for (int j = 0; j < input_size; ++j) {
                  input_to_hidden_weights[j][i] += learning_rate * hidden_errors[i] * input[j];
              }
              hidden_bias[i] += learning_rate * hidden_errors[i];
          }
      }
};

class BPNeuralNetworkPIDController {
  private:
      double Kp, Ki, Kd;
      BPNeuralNetwork nn;
      double prev_error = 0;
      double integral = 0;
  
      double clamp(double value, double min_val, double max_val) {
          return std::max(min_val, std::min(value, max_val));
      }
  
  public:
      BPNeuralNetworkPIDController(double Kp_init, double Ki_init, double Kd_init)
          : Kp(Kp_init), Ki(Ki_init), Kd(Kd_init), nn(4, 5, 3) {}
  
      PIDParams compute(double error, double delta_error,double target_v) {

          integral = clamp(integral + 0.01*error, -100.0, 100.0); // ���ƻ�����
  
          // �����������Ϊ�������仯��
          std::vector<double> input = { error, integral, delta_error ,target_v };
          std::vector<double> output = nn.forward(input);
  
          // ʹ���������������PID����
          // ����PID���������Ų����Ʒ�Χ��
          Kp = clamp(Kp + output[0] * 0.06, 0.0, 1.5);
          Ki = clamp(Ki + output[1] * 0.05, 0.0, 3.0);
          Kd = clamp(Kd + output[2] * 0.1, 0.0, 1.5);
  
          // ����ѵ��Ŀ�꣨���������ٷ���
          std::vector<double> target(3, 0.0);
          target[0] = -error * 0.1;
          target[1] = -integral * 0.1;
          target[2] = -delta_error * 0.1;
  
          // ���򴫲�����������
          nn.backward(input, target);
  
  
          // ��������ź�
          double control_signal = Kp * error + Ki * error + Kd * delta_error;
          return {Kp, Ki, Kd};
      }
      PIDParams get_parm() {
        return {Kp, Ki, Kd};
      }

};

BPNeuralNetworkPIDController pid_x(0.6, 0.0, 0.3),pid_y(0.6, 0.0, 0.3),pid_z(0.3, 0.1, 0.01);

class SmoothedDeltaError {
  private:
      const double delta_t;
      std::deque<double> error_history;
      const int window_size = 5;
      const std::vector<double> coefficients = { -2.0, -1.0, 0.0, 1.0, 2.0 };
  
      // 新增：低通滤波器参数
      double alpha;          // 滤波器系数
      double filtered_delta; // 上一帧滤波输出
      double error;
      double last_error;
      double last_last_error;

      const double max_delta_error = 0.5;  // 限幅
  
  public:
      SmoothedDeltaError(double dt = 0.015, double lpf_cutoff_freq = 12.0)
          : delta_t(dt), filtered_delta(0.0) {
          if (delta_t <= 0) {
              throw std::invalid_argument("delta_t must be positive");
          }
          // 设置低通滤波器系数 (简单一阶离散RC滤波器)
          double rc = 1.0 / (2 * M_PI * lpf_cutoff_freq);  // RC常数
          alpha = delta_t / (delta_t + rc);
      }
  
      double update_error(double new_error) {
          error = 0.3 * new_error + 0.4 * last_error + 0.3 * last_error;
          last_last_error = last_error;
          last_error = error;

          error_history.push_back(error);
          if (error_history.size() > window_size) {
              error_history.pop_front();
          }
          if (error_history.size() < window_size) {
              return 0.0;
          }
  
          // 原始四元差分
          double raw_delta = calculate_quad_differential();
  
          // 应用低通滤波器
          filtered_delta = alpha * raw_delta + (1 - alpha) * filtered_delta;
          filtered_delta = std::max(-max_delta_error, std::min(filtered_delta, max_delta_error));
          return filtered_delta;
      }
  
      void reset() {
          error_history.clear();
          filtered_delta = 0.0;
      }
  
  private:
      double calculate_quad_differential() {
          double weighted_sum = 0.0;
          for (size_t i = 0; i < window_size; ++i) {
              weighted_sum += coefficients[i] * error_history[i];
          }
          return weighted_sum / (10.0 * delta_t);
      }
};
  
SmoothedDeltaError diff_x, diff_y, diff_z;

// Saturation函数（参数命名更清晰）
double sat(double value, double epsilon) {
  if (std::abs(value) > epsilon)
      return (value > 0 ? 1.0 : -1.0);
  else
      return value / epsilon;  // 边界层内线性化
}

// 改进的BLSC控制器（修正误差方向与滑模面）
void BLSCController(
  const Eigen::Vector3d& current_pos,
  const Eigen::Vector3d& current_vel,
  const Eigen::Vector3d& desired_pos,
  const Eigen::Vector3d& desired_vel,
  Eigen::Vector3d& output_vel) {
  // 1. 修正位置误差方向：desired - current
  Eigen::Vector3d pos_error = desired_pos - current_pos;
  Eigen::Vector3d vel_error = desired_vel - current_vel;  // 保持一致性

  // 2. 改进滑模面：s = vel_error + λ * pos_error (关键修改点)
  Eigen::Vector3d s = vel_error + blsc_lambda * pos_error;

  // 3. 计算saturation项（逐元素处理）
  Eigen::Vector3d sat_s;
  for (int i = 0; i < 3; ++i) {
      sat_s[i] = sat(s[i], blsc_boundary_layer);
  }

  // 4. 生成控制指令（修正公式符号）
  output_vel = desired_vel + blsc_lambda * pos_error - sat_s;

  // 5. 速度限幅（严格约束）
  output_vel.x() = std::max(-blsc_max_horiz_vel, std::min(output_vel.x(), blsc_max_horiz_vel));
  output_vel.y() = std::max(-blsc_max_horiz_vel, std::min(output_vel.y(), blsc_max_horiz_vel));
  output_vel.z() = std::max(-blsc_max_vert_vel, std::min(output_vel.z(), blsc_max_vert_vel));
}


class TrajectoryVisualizer {
    public:
        explicit TrajectoryVisualizer(const std::string& frame_id = "world") : frame_id_(frame_id) {}
    
        void visualizeTraj(const Eigen::MatrixXd& position_array, double dt, const std::string& topic = "traj") {
            publishPath(position_array, topic);
            publishWaypoints(position_array, topic + "_wayPts");
        }
    
    private:
        ros::NodeHandle nh_;
        std::string frame_id_;
        std::unordered_map<std::string, ros::Publisher> publishers_;
    
        template<typename T>
        ros::Publisher& getPublisher(const std::string& topic, uint32_t queue_size = 10) {
            if (publishers_.find(topic) == publishers_.end()) {
                publishers_[topic] = nh_.advertise<T>(topic, queue_size);
            }
            return publishers_[topic];
        }
    
        void publishPath(const Eigen::MatrixXd& positions, const std::string& topic) {
            nav_msgs::Path path_msg;
            path_msg.header.frame_id = frame_id_;
            path_msg.header.stamp = ros::Time::now();
    
            for (int i = 0; i < positions.rows(); ++i) {
                geometry_msgs::PoseStamped pose;
                pose.header = path_msg.header;
                pose.pose.position.x = positions(i, 0);
                pose.pose.position.y = positions(i, 1);
                pose.pose.position.z = positions(i, 2);
                path_msg.poses.push_back(pose);
            }
    
            getPublisher<nav_msgs::Path>(topic).publish(path_msg);
        }
    
        void publishWaypoints(const Eigen::MatrixXd& positions, const std::string& topic) {
            sensor_msgs::PointCloud2 cloud_msg;
            cloud_msg.header.frame_id = frame_id_;
            cloud_msg.header.stamp = ros::Time::now();
    
            cloud_msg.height = 1;
            cloud_msg.width = positions.rows();
            cloud_msg.is_bigendian = false;
            cloud_msg.is_dense = true;
    
            sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            modifier.resize(positions.rows());
    
            sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
            sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
            sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    
            for (int i = 0; i < positions.rows(); ++i, ++iter_x, ++iter_y, ++iter_z) {
                *iter_x = static_cast<float>(positions(i, 0));
                *iter_y = static_cast<float>(positions(i, 1));
                *iter_z = static_cast<float>(positions(i, 2));
            }
    
            getPublisher<sensor_msgs::PointCloud2>(topic).publish(cloud_msg);
        }
};

class VelocityEstimator {
  private:
      Eigen::Vector3d last_position;
      ros::Time last_time;
      bool initialized;
      double timeout_threshold;  // 0.5s超时清除
  
  public:
      VelocityEstimator(double timeout_sec = 0.5)
          : initialized(false), timeout_threshold(timeout_sec) {
          last_position.setZero();
      }
  
      // 更新函数，返回速度估计值（单位：m/s）
      Eigen::Vector3d update(const Eigen::Vector3d& new_position, const ros::Time& current_time) {
          if (!initialized) {
              last_position = new_position;
              last_time = current_time;
              initialized = true;
              return Eigen::Vector3d::Zero();
          }
  
          double dt = (current_time - last_time).toSec();
  
          if (dt <= 1e-6 || dt > timeout_threshold) {
              // 时间无效或超时，重置状态
              last_position = new_position;
              last_time = current_time;
              return Eigen::Vector3d::Zero();  // 丢弃估计
          }
  
          Eigen::Vector3d velocity = (new_position - last_position) / dt;
  
          // 更新历史记录
          last_position = new_position;
          last_time = current_time;
  
          return velocity;
      }
  
      // 重置
      void reset() {
          initialized = false;
          last_position.setZero();
          last_time = ros::Time(0);
      }
};

VelocityEstimator drone_v_estimator,target_v_estimator;

Eigen::MatrixXd predictTargetTrajectory(const Eigen::Vector3d& pos, const Eigen::Vector3d& vel, int N, double dt) {
      Eigen::MatrixXd trajectory(N + 1, 6);
      for (int t = 0; t <= N; ++t) {
          trajectory.block<1,3>(t, 0) = pos + vel * (t * dt);
          trajectory.block<1,3>(t, 3) = vel;
      }
      return trajectory;
}


  // 独立函数：通过二次插值获取轨迹中的状态
std::pair<Eigen::VectorXd, Eigen::Vector3d> trajGetState(const std::vector<Eigen::VectorXd>& x_traj,
                                                           const std::vector<Eigen::VectorXd>& u_traj,
                                                           double future_time, double dt) {
      double steps = future_time / dt;
      int idx_base = static_cast<int>(std::floor(steps));
      double alpha = steps - idx_base;
  
      idx_base = std::min(std::max(0, idx_base), static_cast<int>(x_traj.size()) - 3);
  
      const Eigen::VectorXd& x0 = x_traj[idx_base];
      const Eigen::VectorXd& x1 = x_traj[idx_base + 1];
      const Eigen::VectorXd& x2 = x_traj[idx_base + 2];
  
      Eigen::VectorXd state = (0.5 * alpha * (alpha - 1.0)) * x0
                            + (0.5 * alpha * (alpha + 1.0) - 0.5 * (alpha - 1.0) * (alpha + 1.0)) * x1
                            + (0.5 * (alpha - 1.0) * alpha) * x2;
  
      int accel_idx = std::min(idx_base, static_cast<int>(u_traj.size()) - 1);
      Eigen::Vector3d accel = u_traj[accel_idx].segment<3>(0);
  
      return {state, accel};
}



void cmdCallback(const ros::TimerEvent &e) {
  // if (!(hc14_dog_pos_received && hc14_offset_yaw_ready))
  //   return;
  if (triger_mode != -1 && triger_mode != 0 && triger_mode != 1 && triger_mode != 2)
    return;
  if (triger_mode == -1)
    return;
  if (triger_mode == 0 && !traj_initialized)
    return;
  if ((triger_mode == 1 || triger_mode == 2) && !(hc14_dog_pos_received && hc14_offset_yaw_ready))
    return;
  
  quadrotor_msgs::PositionCommand cmd;
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = 0;
  
  double targetx, targety, targetz;
  double target_vx, target_vy, target_vz;
  double error_targetx, error_targety, error_targetz;
  double target_yaw;
  double angle_diff = 0;


  // 目前mpc没有控制yaw，还使用目标的yaw
  target_yaw = hc14_dog_yaw;

  static int yaw_mode = 0;
  if (triger_mode == 2){
    // yaw_mode = 2;
    yaw_mode = 1;
  } else if ((triger_mode == 0) && (Eigen::Vector2d(vins_p.x() - target_p.x(), vins_p.y() - target_p.y()).norm() > 5.0)){
    yaw_mode = 0;
  } else if (Eigen::Vector2d(vins_p.x() - target_p.x(), vins_p.y() - target_p.y()).norm() < 3.0){
    yaw_mode = 1;
  }

  if (yaw_mode == 0) {
    // MPC模式：计算飞机对着狗的方向
    double yaw_to_dog = std::atan2(target_p.y() - vins_p.y(), target_p.x() - vins_p.x());
    angle_diff = yaw_to_dog - vins_yaw;
  } else if (yaw_mode == 1) {
    angle_diff = target_yaw - vins_yaw;
  } else if (yaw_mode == 2) {
    angle_diff = 0.0;
  }

  if (angle_diff > M_PI) {
    angle_diff -= 2 * M_PI;
  } else if (angle_diff < -M_PI) {
    angle_diff += 2 * M_PI;
  }

  if (triger_mode == 0) {
    target_vx = mpc_v.x();
    target_vy = mpc_v.y();
    target_vz = mpc_v.z();

    targetx = mpc_p.x();
    targety = mpc_p.y();
    targetz = mpc_p.z();

  } else if (triger_mode == 1 || triger_mode == 2) {
    target_vx = hc14_dog_vel.x();
    target_vy = hc14_dog_vel.y();

    // 角速度前馈：在狗速度法向（侧面）和切向（前方）添加位置和速度补偿
    double dog_vel_norm = std::sqrt(hc14_dog_vel.x() * hc14_dog_vel.x() + hc14_dog_vel.y() * hc14_dog_vel.y());
    
    // 侧面分量补偿
    double compensation_pos_lateral = hc14_dog_yaw_rate * yaw_rate_pos_gain;
    double compensation_vel_lateral = hc14_dog_yaw_rate * yaw_rate_vel_gain;
    
    // 前方分量补偿
    double compensation_pos_forward = std::fabs(hc14_dog_yaw_rate) * yaw_rate_pos_gain_forward;
    double compensation_vel_forward = std::fabs(hc14_dog_yaw_rate) * yaw_rate_vel_gain_forward;

    // 计算狗速度的法向单位向量（侧面方向，垂直方向）
    double normal_x = 0.0, normal_y = 0.0;
    // 计算狗速度的切向单位向量（前方方向）
    double forward_x = 0.0, forward_y = 0.0;
    
    if (dog_vel_norm > 0.001) {  // 避免除零
      // 法向向量（逆时针90度，侧面方向）
      normal_x = -hc14_dog_vel.y() / dog_vel_norm;
      normal_y = hc14_dog_vel.x() / dog_vel_norm;
      
      // 切向向量（前方方向，与速度方向一致）
      forward_x = hc14_dog_vel.x() / dog_vel_norm;
      forward_y = hc14_dog_vel.y() / dog_vel_norm;
    }
    
    // 计算侧面位置补偿向量
    double pos_compensation_lateral_x = normal_x * compensation_pos_lateral;
    double pos_compensation_lateral_y = normal_y * compensation_pos_lateral;
    
    // 计算前方位置补偿向量
    double pos_compensation_forward_x = forward_x * compensation_pos_forward;
    double pos_compensation_forward_y = forward_y * compensation_pos_forward;
    
    // 计算侧面速度补偿向量
    double vel_compensation_lateral_x = normal_x * compensation_vel_lateral;
    double vel_compensation_lateral_y = normal_y * compensation_vel_lateral;
    
    // 计算前方速度补偿向量
    double vel_compensation_forward_x = forward_x * compensation_vel_forward;
    double vel_compensation_forward_y = forward_y * compensation_vel_forward;

    targetx = hc14_dog_pos.x();
    targety = hc14_dog_pos.y();
    
    // 添加侧面和前方的位置补偿
    targetx += pos_compensation_lateral_x + pos_compensation_forward_x;
    targety += pos_compensation_lateral_y + pos_compensation_forward_y;
    
    // 添加侧面和前方的速度补偿
    target_vx += vel_compensation_lateral_x + vel_compensation_forward_x;
    target_vy += vel_compensation_lateral_y + vel_compensation_forward_y;    

    if (triger_mode == 1)
    {
      if (reflight_complete){
        targetz = std::min(target_p.z() + land_height_limit[1], std::max(target_p.z() + land_height_limit[0], vins_p.z()));
        target_vz = target_v.z();
      } else {
        // 计算目标速度
        if (mode_vins_z < target_p.z() + land_height_limit[0]){
          mode_vins_vel_z = 0.05;
          if (vins_p.z() > target_p.z() + land_height_limit[0]){
            reflight_complete = true;
          }
        } else if (mode_vins_z > target_p.z() + land_height_limit[1]){
          mode_vins_vel_z = -0.05;
          if (vins_p.z() < target_p.z() + land_height_limit[1]){
            reflight_complete = true;
          }
        }
        else {
          reflight_complete = true;
        }
        
        // 持续更新mode_vins_z（速度乘以dt）
        mode_vins_z += mode_vins_vel_z * accel_dt;
        target_vz = target_v.z() + mode_vins_vel_z;
        targetz = mode_vins_z;
      }
    }
    else if (triger_mode == 2)
    {
      // 降落模式：根据高度差分段设置下降速度（先快后慢）
      double height_diff = vins_p.z() - target_p.z();

      // 计算目标速度
      double target_vel_z;
      if (height_diff >= land_height_fast) {
        target_vel_z = land_vel_max_high;  // 高度高，快降
      } else if (height_diff <= land_height_slow) {
        target_vel_z = land_vel_max_low;   // 高度低，慢降
      } else {
        double ratio = (height_diff - land_height_slow) / (land_height_fast - land_height_slow);
        target_vel_z = land_vel_max_low + ratio * (land_vel_max_high - land_vel_max_low);
      }
    
      // 计算目标加速度（限制在max_accel_z内）
      double desired_accel_z = (target_vel_z - mode_vins_vel_z) / accel_dt;
      desired_accel_z = std::max(-max_accel_z, std::min(max_accel_z, desired_accel_z));
      
      // 持续更新mode_vins_vel_z（加速度乘以dt，限制速度变化不超过dt*max_accel）
      double vel_change = desired_accel_z * accel_dt;
      mode_vins_vel_z += vel_change;
      
      // 持续更新mode_vins_z（速度乘以dt）
      mode_vins_z += mode_vins_vel_z * accel_dt;
      target_vz = target_v.z() + mode_vins_vel_z;
      targetz = mode_vins_z;

    }

  }

  error_targetx = targetx - vins_p.x();
  error_targety = targety - vins_p.y();
  error_targetz = targetz - vins_p.z();

  if(last_error_targetx == 0){
    last_error_targetx = error_targetx;
    last_error_targety = error_targety;
    last_error_targetz = error_targetz;
  }

  double cos_yaw = cos(vins_yaw);
  double sin_yaw = sin(vins_yaw);

  // 坐标系变换到平台体系（狗体系）
  double error_x_body =  error_targetx * cos_yaw - error_targety * sin_yaw;
  double error_y_body =  error_targetx * sin_yaw + error_targety * cos_yaw;
  double last_error_x_body = last_error_targetx * cos_yaw - last_error_targety * sin_yaw;
  double last_error_y_body = last_error_targetx * sin_yaw + last_error_targety * cos_yaw;
  double target_vx_body = target_vx * cos_yaw - target_vy * sin_yaw;
  double target_vy_body = target_vx * sin_yaw + target_vy * cos_yaw;
  double target_x_body = targetx * cos_yaw - targety * sin_yaw;
  double target_y_body = targetx * sin_yaw + targety * cos_yaw;

  double current_x_p, current_y_p, current_z_p, current_pos_x_p, current_pos_y_p;
  double current_x_i, current_y_i, current_z_i, current_pos_x_i, current_pos_y_i;
  double current_x_d, current_y_d, current_z_d, current_pos_x_d, current_pos_y_d;
  double current_x_d_max, current_y_d_max, current_z_d_max, current_pos_x_d_max, current_pos_y_d_max;
  
  if (triger_mode == 0) {
    current_x_p = mpc_x_p;
    current_y_p = mpc_y_p;
    current_z_p = mpc_z_p;
    current_pos_x_p = mpc_pos_x_p;
    current_pos_y_p = mpc_pos_y_p;
    current_x_i = mpc_x_i;
    current_y_i = mpc_y_i;
    current_z_i = mpc_z_i;
    current_pos_x_i = mpc_pos_x_i;
    current_pos_y_i = mpc_pos_y_i;
    current_x_d = mpc_x_d;
    current_y_d = mpc_y_d;
    current_z_d = mpc_z_d;
    current_pos_x_d = mpc_pos_x_d;
    current_pos_y_d = mpc_pos_y_d;
    current_x_d_max = mpc_x_d_max;
    current_y_d_max = mpc_y_d_max;
    current_z_d_max = mpc_z_d_max;
    current_pos_x_d_max = mpc_pos_x_d_max;
    current_pos_y_d_max = mpc_pos_y_d_max;
  } else if (triger_mode == 1 || triger_mode == 2) {  
    current_x_p = x_p;
    current_y_p = y_p;
    current_z_p = z_p;
    current_pos_x_p = pos_x_p;
    current_pos_y_p = pos_y_p;
    current_x_i = x_i;
    current_y_i = y_i;
    current_z_i = z_i;
    current_pos_x_i = pos_x_i;
    current_pos_y_i = pos_y_i;
    current_x_d = x_d;
    current_y_d = y_d;
    current_z_d = z_d;
    current_pos_x_d = pos_x_d;
    current_pos_y_d = pos_y_d;
    current_x_d_max = x_d_max;
    current_y_d_max = y_d_max;
    current_z_d_max = z_d_max;
    current_pos_x_d_max = pos_x_d_max;
    current_pos_y_d_max = pos_y_d_max;
  }

  // 根据trigger_mode选择使用世界系还是body系的积分项
  double integral_x_used, integral_y_used;
  if (triger_mode == 0) {
    // trigger_mode=0时使用世界系积分项，需要转换到body系
    integral_x_used = intergral_targetx * cos_yaw - intergral_targety * sin_yaw;
    integral_y_used = intergral_targetx * sin_yaw + intergral_targety * cos_yaw;
  } else {
    // trigger_mode=1或2时使用body系积分项
    integral_x_used = intergral_targetx_body;
    integral_y_used = intergral_targety_body;
  }

  // 狗体系下PID
  double vx_body = target_vx_body + current_x_p * error_x_body + current_x_i * integral_x_used + std::max(std::min(current_x_d * (error_x_body - last_error_x_body), current_x_d_max), -current_x_d_max);
  double vy_body = target_vy_body + current_y_p * error_y_body + current_y_i * integral_y_used + std::max(std::min(current_y_d * (error_y_body - last_error_y_body), current_y_d_max), -current_y_d_max);
  double vz = target_vz + current_z_p * error_targetz + current_z_i * intergral_targetz + std::max(std::min(current_z_d * (error_targetz - last_error_targetz), current_z_d_max), -current_z_d_max);

  double x_body = target_x_body + current_pos_x_p * error_x_body + current_pos_x_i * integral_x_used + std::max(std::min(current_pos_x_d * (error_x_body - last_error_x_body), current_pos_x_d_max), -current_pos_x_d_max);
  double y_body = target_y_body + current_pos_y_p * error_y_body + current_pos_y_i * integral_y_used + std::max(std::min(current_pos_y_d * (error_y_body - last_error_y_body), current_pos_y_d_max), -current_pos_y_d_max);
  
  // 再转回世界系
  cmd.velocity.x = vx_body * cos_yaw + vy_body * sin_yaw;
  cmd.velocity.y = - vx_body * sin_yaw + vy_body * cos_yaw;
  cmd.velocity.z = vz;
  
  cmd.position.x = x_body * cos_yaw + y_body * sin_yaw;
  cmd.position.y = - x_body * sin_yaw + y_body * cos_yaw;
  cmd.position.z = targetz;

  // 在精确模式下，设置加速度前馈
  if ((triger_mode == 1 || triger_mode == 2)) {
    cmd.acceleration.x = hc14_dog_acc.x() * acc_gain_x;
    cmd.acceleration.y = hc14_dog_acc.y() * acc_gain_y;
  }

  // 角度限制逻辑，防止一次转角太大
  cmd.yaw = std::max(std::min(angle_diff, 0.5), -0.5) + vins_yaw;

  // P控制器计算角速度
  // cmd.yaw_dot = std::max(-max_yaw_rate, std::min(max_yaw_rate, yaw_kp * (cmd.yaw - vins_yaw)));
  cmd.yaw_dot = 0.0;

  if(triger_mode == 2){
    // 降落过程中需要实时更新target信息
    // if (target_count == last_precise_target_count) {
    //   if (target_receive_triger == 1){
    //     target_receive_triger = 0;
    //     target_lost_time = ros::Time::now();
    //     target_lost_p.x() = cmd.position.x;
    //     target_lost_p.y() = cmd.position.y;
    //     target_lost_v.x() = cmd.velocity.x;
    //     target_lost_v.y() = cmd.velocity.y;
    //   }

    //   // Landing模式下的velocity前馈控制
    //   target_lost_v.x() += (target_vx - last_target_v.x()) * 0.75;
    //   target_lost_v.y() += (target_vy - last_target_v.y()) * 0.75;
    //   cmd.velocity.x = target_lost_v.x();
    //   cmd.velocity.y = target_lost_v.y();

    //   target_lost_p.x() += cmd.velocity.x * accel_dt;
    //   target_lost_p.y() += cmd.velocity.y * accel_dt;

    //   cmd.position.x = target_lost_p.x();
    //   cmd.position.y = target_lost_p.y();
    // }
    // else{
    //   target_receive_triger = 1;
    // }
    
    // 丢失目标之后如果target_p变了怎么办？
    if ((flow_z < 0.1) && (flow_z > 0.0) && (std::fabs(vins_p.z() - target_p.z()) < 0.5))
        land_lock_timer += 1;
    else
        land_lock_timer = std::max(land_lock_timer - 0.5, 0.0);

    // if (land_lock_timer > 1 && 
    //   std::fabs(vins_p.x() - target_p.x()) < 0.1 &&
    //   std::fabs(vins_p.y() - target_p.y()) < 0.1)
    if (land_lock_timer > 8)
    {
      quadrotor_msgs::TakeoffLand land;
      land.takeoff_land_cmd = 2;
      land_pub_.publish(land);

      // 发布到ROS话题
      std_msgs::Float64 yaw_diff_msg;
      yaw_diff_msg.data = angle_diff;
      yaw_offset_pub.publish(yaw_diff_msg);
      std::cout << "Published yaw diff: " << angle_diff * 180.0 / M_PI << " degrees" << std::endl;
      

      geometry_msgs::PoseStamped mode_msg;
      mode_msg.header.stamp = ros::Time::now();
      mode_msg.pose.orientation.w = -1;
      mode_pub_.publish(mode_msg);
    }
  }

  cmd.velocity.x += v_offset_x * cos(vins_yaw) - v_offset_y * sin(vins_yaw);
  cmd.velocity.y += v_offset_x * sin(vins_yaw) + v_offset_y * cos(vins_yaw);

  // 做上下限限制（z 方向统一使用固定饱和，降落模式的速度规划已在上游计算）
  cmd.velocity.x = std::max(-1.5, std::min(1.5, cmd.velocity.x));
  cmd.velocity.y = std::max(-1.5, std::min(1.5, cmd.velocity.y));
  cmd.velocity.z = std::max(-0.8, std::min(0.8, cmd.velocity.z));

  // 加速度和jerk限制：限制加速度变化（jerk）、速度变化（加速度）和位置变化（对 x、y、z 一起生效）
  if (last_cmd_initialized) {
    // 计算期望的速度变化
    Eigen::Vector3d vel_change(
        cmd.velocity.x - last_cmd_velocity.x(),
        cmd.velocity.y - last_cmd_velocity.y(),
        cmd.velocity.z - last_cmd_velocity.z());
    
    // 计算期望的加速度（基于速度变化）
    Eigen::Vector3d desired_accel = vel_change / accel_dt;
    
    // 计算加速度变化（jerk）
    Eigen::Vector3d accel_change = desired_accel - last_cmd_acceleration;
    double max_accel_change = max_jerk * accel_dt;
    
    // 限制jerk：限制加速度变化不超过 max_jerk * dt
    if (accel_change.norm() > max_accel_change) {
      accel_change = accel_change.normalized() * max_accel_change;
    }
    
    // 得到限制后的加速度
    Eigen::Vector3d limited_accel = last_cmd_acceleration + accel_change;
    
    // 限制加速度：限制加速度幅度不超过 max_accel
    if (limited_accel.norm() > max_accel) {
      limited_accel = limited_accel.normalized() * max_accel;
    }
    
    // 基于限制后的加速度，计算限制后的速度变化
    Eigen::Vector3d limited_vel_change = limited_accel * accel_dt;
    
    // 更新速度
    cmd.velocity.x = last_cmd_velocity.x() + limited_vel_change.x();
    cmd.velocity.y = last_cmd_velocity.y() + limited_vel_change.y();
    cmd.velocity.z = last_cmd_velocity.z() + limited_vel_change.z();
    
    // 限制位置变化：不能超过限制后的速度 * dt
    Eigen::Vector3d pos_change(
        cmd.position.x - last_cmd_position.x(),
        cmd.position.y - last_cmd_position.y(),
        cmd.position.z - last_cmd_position.z());
    Eigen::Vector3d limited_vel(cmd.velocity.x, cmd.velocity.y, cmd.velocity.z);
    double max_pos_change = limited_vel.norm() * accel_dt * 3.5;

    if (max_pos_change > 0.001 && pos_change.norm() > max_pos_change) {
      pos_change = pos_change.normalized() * max_pos_change;
      cmd.position.x = last_cmd_position.x() + pos_change.x();
      cmd.position.y = last_cmd_position.y() + pos_change.y();
      cmd.position.z = last_cmd_position.z() + pos_change.z();
    }
    
    // 更新保存的加速度（用于下一次jerk限制）
    last_cmd_acceleration = limited_accel;
  } else {
    last_cmd_initialized = true;
    cmd.velocity.x = vins_v.x();
    cmd.velocity.y = vins_v.y();
    cmd.velocity.z = vins_v.z();
    cmd.position.x = vins_p.x();
    cmd.position.y = vins_p.y();
    cmd.position.z = vins_p.z();
    // 初始化加速度为0
    last_cmd_acceleration = Eigen::Vector3d(0, 0, 0);
  }
  
  // 更新保存的上一次命令值
  last_cmd_velocity = Eigen::Vector3d(cmd.velocity.x, cmd.velocity.y, cmd.velocity.z);
  last_cmd_position = Eigen::Vector3d(cmd.position.x, cmd.position.y, cmd.position.z);


  last_target_v.x() = target_vx;
  last_target_v.y() = target_vy;
  last_target_v.z() = target_vz;

  last_error_targetx = error_targetx;
  last_error_targety = error_targety;
  last_error_targetz = error_targetz;

  last_precise_target_count = target_count;

  // 积分项带衰减，避免windup
  // 根据trigger_mode选择在世界系还是body系累积积分项
  double decay_factor = 1.0 - integral_decay * accel_dt;
  decay_factor = std::max(0.0, std::min(1.0, decay_factor));  // 防止异常

  if (triger_mode == 0) {
    // trigger_mode=0时在世界系累积x和y的积分项
    intergral_targetx = intergral_targetx * decay_factor + accel_dt * (cmd.position.x - vins_p.x());
    intergral_targety = intergral_targety * decay_factor + accel_dt * (cmd.position.y - vins_p.y());
  } else {
    // trigger_mode=1或2时在body系累积x和y的积分项
    intergral_targetx_body = intergral_targetx_body * decay_factor + accel_dt * ((cmd.position.x - vins_p.x()) * cos_yaw - (cmd.position.y - vins_p.y()) * sin_yaw);
    intergral_targety_body = intergral_targety_body * decay_factor + accel_dt * ((cmd.position.x - vins_p.x()) * sin_yaw + (cmd.position.y - vins_p.y()) * cos_yaw);
  }
  
  // z轴积分项仍在世界系累积（因为z轴不需要坐标变换）
  intergral_targetz = intergral_targetz * decay_factor + accel_dt * (cmd.position.z - vins_p.z());

  // 限制积分项
  if (triger_mode == 0) {
    intergral_targetx = std::max(std::min(intergral_targetx, integral_limit_x), -integral_limit_x);
    intergral_targety = std::max(std::min(intergral_targety, integral_limit_y), -integral_limit_y);
  } else {
    intergral_targetx_body = std::max(std::min(intergral_targetx_body, integral_limit_x), -integral_limit_x);
    intergral_targety_body = std::max(std::min(intergral_targety_body, integral_limit_y), -integral_limit_y);
  }
  intergral_targetz = std::max(std::min(intergral_targetz, integral_limit_z), -integral_limit_z);


  // 发布调试信息到plotjuggler
  // quadrotor_msgs::PositionCommand debug_msg;
  // debug_msg.header.stamp = ros::Time::now();
  // debug_msg.header.frame_id = "world";
  // debug_msg.position.x = error_targetx;  // 使用position字段存储error_targetx
  // debug_msg.position.y = error_targety;  // 使用position字段存储error_targety
  // debug_msg.position.z = error_targetz;  // 使用position字段存储error_targetz
  // debug_msg.velocity.x = intergral_targetx;  // 使用velocity字段存储积分项
  // debug_msg.velocity.y = intergral_targety;
  // debug_msg.velocity.z = intergral_targetz;
  // debug_msg.yaw = last_error_targetx;  // 使用yaw字段存储last_error_targetx
  // debug_pub_.publish(debug_msg);

  pos_cmd_pub_.publish(cmd);
  return;
}

void traj_callback(const nav_msgs::Path::ConstPtr& msg) {

    // 每次完全更新 trajectory_points，只保留起点一致的新轨迹
    traj_sub_time = ros::Time::now();
    trajectory_points.clear();
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d pos;
        pos << pose.pose.position.x,
               pose.pose.position.y,
               pose.pose.position.z;
        trajectory_points.push_back(pos);
    }
}

void traj_v_callback(const nav_msgs::Path::ConstPtr& msg) {
    if (!traj_initialized && !msg->poses.empty()) {
        traj_initialized = true;
    }

    // 每次完全更新 trajectory_points，只保留起点一致的新轨迹
    trajectory_v_points.clear();
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d pos;
        pos << pose.pose.position.x,
               pose.pose.position.y,
               pose.pose.position.z;
        trajectory_v_points.push_back(pos);
    }
}

void traj_a_callback(const nav_msgs::Path::ConstPtr& msg) {
    // 处理加速度轨迹
    trajectory_a_points.clear();
    for (const auto& pose : msg->poses) {
        Eigen::Vector3d acc;
        acc << pose.pose.position.x,
               pose.pose.position.y,
               pose.pose.position.z;
        trajectory_a_points.push_back(acc);
    }
}


void flag_and_hc14_process_callback(const ros::TimerEvent &event) {
    // 检查target_receive状态（模仿dog_pos_received的逻辑）
    if (target_count != last_target_count) {
        last_target_timer++;
        if (last_target_timer >= 5) {
            target_receive = true;
        }
        last_target_count = target_count;
        last_target_loss_timer = 0;
    } else {
        last_target_loss_timer++;
        if (last_target_loss_timer >= 5) {  // 连续5次没有新包才重置
            target_receive = false;
        }
        last_target_timer -= 0.2;
    }
    
    // 检查dog_pos_received状态（模仿target_count的逻辑）
    if (hc14_dog_pos_count != last_hc14_dog_pos_count) {
        hc14_dog_pos_received = true;
        last_hc14_dog_pos_count = hc14_dog_pos_count;
        last_hc14_dog_pos_timer = 0;  // 重置dog_pos的timer
    } else {
        last_hc14_dog_pos_timer++;
        if (last_hc14_dog_pos_timer >= 5)  // 连续5次没有新包才重置
        hc14_dog_pos_received = false;
    }

    double angle_diff = target_dog_yaw - vins_yaw;
    if (angle_diff > M_PI) {
      angle_diff -= 2 * M_PI;
    } else if (angle_diff < -M_PI) {
      angle_diff += 2 * M_PI;
    }

    // 处理triger_mode切换逻辑
    if (triger_mode == 0){
      if (hc14_offset_pos_ready && 
          hc14_offset_yaw_ready && 
          target_receive && 
          vins_p.z() > target_p.z() + land_height_limit[0] &&
          Eigen::Vector2d(vins_p.x() - target_p.x(), vins_p.y() - target_p.y()).norm() < 0.8 && 
          angle_diff < 30.0/180.0 * M_PI)
      {
          geometry_msgs::PoseStamped mode_msg;
          mode_msg.header.stamp = ros::Time::now();
          mode_msg.pose.orientation.w = 1;
          mode_pub_.publish(mode_msg);
          std::cout << "precise_mode: true" << std::endl;
      }
    }
    else if (triger_mode == 1){
      if (hc14_offset_yaw_ready && 
          hc14_offset_pos_ready &&
          std::fabs(target_v.x() - vins_v.x()) < 0.2 && 
          std::fabs(target_v.y() - vins_v.y()) < 0.2 && 
          std::fabs(target_p.x() - vins_p.x()) < 0.4 &&
          std::fabs(target_p.y() - vins_p.y()) < 0.4 &&
          angle_diff < 15.0/180.0 * M_PI &&
          hc14_perception_confidence > 0.6)
      {
        land_timer ++;
        if (land_timer > 10)
        {
          geometry_msgs::PoseStamped mode_msg;
          mode_msg.header.stamp = ros::Time::now();
          mode_msg.pose.orientation.w = 2;
          mode_pub_.publish(mode_msg);
          std::cout << "land_mode: true" << std::endl;
          land_timer = 0;
        }
      }else if (Eigen::Vector2d(vins_p.x() - target_p.x(), vins_p.y() - target_p.y()).norm() > 2.0){ // 可能遇到障碍时切回mpc避障
        geometry_msgs::PoseStamped mode_msg;
        mode_msg.header.stamp = ros::Time::now();
        mode_msg.pose.orientation.w = 0;
        mode_pub_.publish(mode_msg);
        std::cout << "precise_mode: false" << std::endl;
        land_timer = std::max(0.0, land_timer - 0.5);
      }else{
        land_timer = std::max(0.0, land_timer - 0.5);
      }
    }
    else if (triger_mode == 2){
      if (!(
            hc14_offset_pos_ready &&
            hc14_offset_yaw_ready &&
            std::fabs(target_v.x() - vins_v.x()) < 0.65 && 
            std::fabs(target_v.y() - vins_v.y()) < 0.65 && 
            std::fabs(target_p.x() - vins_p.x()) < 0.55 &&
            std::fabs(target_p.y() - vins_p.y()) < 0.55 &&
            vins_p.z() > target_p.z() - 0.1 &&
            angle_diff < 25.0/180.0 * M_PI &&
            (hc14_perception_confidence > 0.4 || vins_p.z() > target_p.z() + 0.3)))
      {
        geometry_msgs::PoseStamped mode_msg;
        mode_msg.header.stamp = ros::Time::now();
        mode_msg.pose.orientation.w = 1;
        mode_pub_.publish(mode_msg);
        std::cout << "land_mode: false" << std::endl;
      }
    }
  

}

void mpc_callback(const ros::TimerEvent &event){
  if (!traj_initialized)
  {
    return;
  }
  ros::Time now = ros::Time::now();
  double dt = (now - traj_sub_time).toSec();
  double dt_step = 0.1;  // 时间步长
  
  dt += 0.0;  // 提前取未来的点
  
  // 计算浮点索引
  double idx_float = dt / dt_step;
  
  // 找到相邻的两个索引
  int idx_down = static_cast<int>(std::floor(idx_float));
  int idx_up = idx_down + 1;
  
  // 边界处理：限制在有效范围内
  int max_idx = static_cast<int>(trajectory_points.size() - 1);
  idx_down = std::max(0, std::min(idx_down, max_idx));
  idx_up = std::max(0, std::min(idx_up, max_idx));
  
  // 如果超出范围，使用最后一个点
  if (idx_down >= max_idx) {
    mpc_p = trajectory_points[max_idx];
    mpc_v = trajectory_v_points[max_idx];
    // 如果有加速度轨迹，也使用最后一个点
    if (!trajectory_a_points.empty() && max_idx < static_cast<int>(trajectory_a_points.size())) {
      mpc_a = trajectory_a_points[max_idx];
    }
    return;
  }
  
  // 计算权重：idx_float距离idx_down越近，weight_down越大
  double weight_up = idx_float - idx_down;  // 距离下界的距离
  double weight_down = 1.0 - weight_up;     // 距离上界的距离
  
  // 加权求和
  mpc_p = weight_down * trajectory_points[idx_down] + weight_up * trajectory_points[idx_up];
  mpc_v = weight_down * trajectory_v_points[idx_down] + weight_up * trajectory_v_points[idx_up];
  
  // 如果有加速度轨迹，也进行插值
  if (!trajectory_a_points.empty() && 
      idx_down < static_cast<int>(trajectory_a_points.size()) && 
      idx_up < static_cast<int>(trajectory_a_points.size())) {
    mpc_a = weight_down * trajectory_a_points[idx_down] + weight_up * trajectory_a_points[idx_up];
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle nh("~");

  ros::Subscriber heartbeat_sub = nh.subscribe("heartbeat", 10, heartbeatCallback);
  ros::Subscriber triger_sub_ = nh.subscribe("/mode_manager", 10, mode_callback);
  ros::Subscriber odom_sub_ = nh.subscribe("/vins_fusion/imu_propagate", 10, odom_callback);
  ros::Subscriber target_sub_ = nh.subscribe("/target_ekf_odom", 10, target_callback);
  ros::Subscriber AOA_sub_ = nh.subscribe("/AOA_Tag_data", 10, AOA_callback);
  ros::Subscriber flow_sub_ = nh.subscribe("/flow_data", 10, flow_callback);

  // ros::Subscriber sub = nh.subscribe("/pid", 1000, pid_callback);

  ros::Subscriber traj_sub_ = nh.subscribe<nav_msgs::Path>("/drone2/planning/traj", 10, traj_callback);
  ros::Subscriber traj_v_sub_ = nh.subscribe<nav_msgs::Path>("/traj_v", 10, traj_v_callback);
  ros::Subscriber traj_a_sub_ = nh.subscribe<nav_msgs::Path>("/traj_a", 10, traj_a_callback);
  
  // 订阅处理后的dog_pos话题
  ros::Subscriber dog_pos_sub_ = nh.subscribe<nav_msgs::Odometry>("/dog_pos_processed", 10, dog_pos_callback);
  

  pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
  land_pub_ = nh.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land",1);
  mode_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/mode_manager", 10);
  debug_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/debug_info", 50); // 添加调试信息发布器
  
  ros::Timer init_timer = nh.createTimer(ros::Duration(2.0), initCallback);

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.015), cmdCallback);


  ros::Timer mpc_timer = nh.createTimer(ros::Duration(0.01), mpc_callback);

  ros::Timer flag_and_hc14_process_timer = nh.createTimer(ros::Duration(0.1), flag_and_hc14_process_callback);

  yaw_offset_pub = nh.advertise<std_msgs::Float64>("/yaw_diff_preset", 10); // 发布飞机和狗的yaw差值


  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}