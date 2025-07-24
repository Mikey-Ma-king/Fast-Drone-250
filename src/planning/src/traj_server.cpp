#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/TakeoffLand.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
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

#include "callback.h"

ros::Publisher pos_cmd_pub_;
ros::Publisher land_pub_;
int land_test_count = 0;
ros::Publisher land_mark_pub_;
ros::Publisher test_mark_pub_;
ros::Publisher land_triger_pub;
ros::Time heartbeat_time_;
bool receive_traj_ = false;
bool flight_start_ = false;
Eigen::Vector3d last_p_;
double last_yaw_ = 0;

double Kt = 0.6;
bool triger_received_ = false;
bool land_triger_received_ = false;
bool stop_triger_received_ = false; 
Eigen::Vector3d vins_p{0,0,0},vins_v{0,0,0};
Eigen::Vector3d target_p{0,0,0},target_v{0,0,0};
double vins_yaw = 0;
double last_target_yaw = 0;
double target_yaw = 0;

bool target_receive = false;
double target_dog_yaw = 0;
double kp = 1.2;
double tracking_dist_ = 1.5;
double landtriger = 0;

double AOA_x = 10;
double AOA_w = 0;
double flow_z = -1;
ros::Time flow_timer;
bool flow_detect = false;
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

double intergral_mpcx = 0;
double intergral_mpcy = 0;
double intergral_mpcz = 0;
double last_error_mpcx = 0;
double last_error_mpcy = 0;
double last_error_mpcz = 0;

int target_land_flag = 0;

std::vector<Eigen::Vector3d> target_p_list,target_v_list;

ros::Time traj_start_time;
bool traj_initialized = false;
double traj_dt = 0.2;
std::vector<Eigen::Vector3d> trajectory_points;
std::vector<Eigen::Vector3d> trajectory_v_points;

Eigen::Vector3d mpc_p{0,0,0},mpc_v{0,0,0};

int BPNN_count = 60;
ros::Time traj_sub_time;

double land_count = 0;

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

Eigen::Vector3d get_traj_point_at() {
    if (!traj_initialized || trajectory_points.size() < 2) {
        ROS_WARN("Trajectory not initialized or too short");
        return Eigen::Vector3d::Zero();
    }

    double t_now = (ros::Time::now() - traj_start_time).toSec() + 0.15;
    double idx_f = t_now / traj_dt;
    size_t idx = static_cast<size_t>(idx_f);
    double alpha = idx_f - idx;

    if (idx >= trajectory_points.size() - 1) {
        return trajectory_points.back();
    }

    Eigen::Vector3d p0 = trajectory_points[idx];
    Eigen::Vector3d p1 = trajectory_points[idx + 1];

    return (1.0 - alpha) * p0 + alpha * p1;
}

Eigen::Vector3d get_traj_v_point_at() {
    if (!traj_initialized || trajectory_v_points.size() < 2) {
        ROS_WARN("Velocity trajectory not initialized or too short");
        return Eigen::Vector3d::Zero();
    }

    double t_now = (ros::Time::now() - traj_start_time).toSec() + 0.15;
    double idx_f = t_now / traj_dt;
    size_t idx = static_cast<size_t>(idx_f);
    double alpha = idx_f - idx;

    if (idx >= trajectory_v_points.size() - 1) {
        return trajectory_v_points.back();
    }

    Eigen::Vector3d v0 = trajectory_v_points[idx];
    Eigen::Vector3d v1 = trajectory_v_points[idx + 1];

    return (1.0 - alpha) * v0 + alpha * v1;
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
  if (!target_receive)
    return;
  if (!triger_received_)
    return;
    if (stop_triger_received_)
    return;
  if (!receive_traj_ || true)
  {
    double mid_yaw = 0;
    double angle_diff = atan2(sin(target_dog_yaw - vins_yaw), cos(target_dog_yaw - vins_yaw));
    mid_yaw = 0.8*angle_diff + vins_yaw;
    quadrotor_msgs::PositionCommand cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.header.frame_id = "world";
    cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
    cmd.trajectory_id = 0;
    
    if (traj_initialized)
    {
      Eigen::Vector2d land_mpc_velocity;
      double error_mpcx = mpc_p.x() - vins_p.x();
      double error_mpcy = mpc_p.y() - vins_p.y();
      double error_mpcz = mpc_p.z() - vins_p.z();
      // error_mpcx = 0;
      // error_mpcy = 0;
      error_mpcz = 0;
      if(last_error_mpcx == 0){
        last_error_mpcx = error_mpcx;
        last_error_mpcy = error_mpcy;
        last_error_mpcz = error_mpcz;
      }
      cmd.position.x = mpc_p.x();
      cmd.position.y = mpc_p.y();
      cmd.position.z = mpc_p.z();
      land_mpc_velocity.x() = mpc_v.x() + 0.4 * error_mpcx + 1.8 * (error_mpcx - last_error_mpcx);
      land_mpc_velocity.y() = mpc_v.y() + 0.6 * error_mpcy + 0.15 * intergral_mpcy;

    //   BPNN_count ++;
    //   if (BPNN_count > 4) {
    //   auto params_x = pid_x.compute(error_mpcx, error_mpcx - last_error_mpcx, mpc_v.x());
    //   auto params_y = pid_y.compute(error_mpcy, error_mpcy - last_error_mpcy, mpc_v.y());
    //   auto params_z = pid_z.compute(error_mpcz,, 0.0);
    //   BPNN_count -= 4;
    //   // std::cout << "derror_x: " << delta_error_x << " derror_y: " << delta_error_y << std::endl;
    //   }
    //   const double adaptive_kp_x = pid_x.get_parm().Kp;
    //   const double adaptive_ki_x = pid_x.get_parm().Ki;
    //   const double adaptive_kd_x = pid_x.get_parm().Kd;

    
    //   const double adaptive_kp_y = pid_y.get_parm().Kp;
    //   const double adaptive_ki_y = pid_y.get_parm().Ki;
    //   const double adaptive_kd_y = pid_y.get_parm().Kd;

    
    //   const double adaptive_kp_z = pid_z.get_parm().Kp;
    //   const double adaptive_ki_z = pid_z.get_parm().Ki;
    //   const double adaptive_kd_z = pid_z.get_parm().Kd;
    
      if(!land_triger_received_){
        cmd.velocity.x = mpc_v.x() + 0.4 * error_mpcx + std::max(std::min(1.8 * (error_mpcx - last_error_mpcx),0.12),-0.12);
        cmd.velocity.y = mpc_v.y() + 0.6 * error_mpcy + 0.3 * intergral_mpcy + std::max(std::min(2.5 * (error_mpcy - last_error_mpcy),0.12),-0.12);
        cmd.velocity.z = mpc_v.z() + 0.3 * error_mpcz + 0.3 * intergral_mpcz + std::max(std::min(1.8 * (error_mpcz - last_error_mpcz),0.12),-0.12);

        // Eigen::Vector3d BLSC_velocity;
        // BLSCController(vins_p, vins_v, mpc_p, mpc_v, BLSC_velocity);
        // cmd.velocity.x = BLSC_velocity.x();
        // cmd.velocity.y = BLSC_velocity.y();
        // cmd.velocity.z = BLSC_velocity.z();
      }
      else{
        cmd.velocity.x = land_mpc_velocity.x();
        cmd.velocity.y = land_mpc_velocity.y();
        cmd.velocity.z = mpc_v.z() + land_mpc_velocity.norm() * 0.0 + 0.3* error_mpcz + 1.8 * (error_mpcz - last_error_mpcz);
        // Eigen::Vector3d BLSC_velocity;
        // BLSCController(vins_p, vins_v, mpc_p, mpc_v, BLSC_velocity);
        // cmd.velocity.x = BLSC_velocity.x();
        // cmd.velocity.y = BLSC_velocity.y();
        // cmd.velocity.z = BLSC_velocity.z();

      }
      last_error_mpcx = error_mpcx;
      last_error_mpcy = error_mpcy;
      last_error_mpcz = error_mpcz;
      intergral_mpcx += 0.01 * error_mpcx;
      intergral_mpcy += 0.01 * error_mpcy;
      intergral_mpcz += 0.01 * error_mpcz;
      if (landtriger == 1 || land_triger_received_ && flow_detect && (ros::Time::now() - flow_timer).toSec() > 0.12  || AOA_x < 0.05)
      {
        landtriger = 1;
        cmd.position.z = vins_p.z() - 1;
        cmd.velocity.z = -0.8;
        cmd.position.x = vins_p.x();
        cmd.position.y = vins_p.y();
        cmd.velocity.x = target_v.x();
        cmd.velocity.y = target_v.y();
        land_test_count += 1;
        if (land_test_count > 40 && land_test_count < 200)
        {
          quadrotor_msgs::TakeoffLand land;
          land.takeoff_land_cmd = 2;
          land_pub_.publish(land);
        }
      }
    }
    else {
      return;
    }

    cmd.yaw = vins_yaw;

    cmd.yaw = 0.4*angle_diff + vins_yaw;
    pos_cmd_pub_.publish(cmd);
    if (land_triger_received_ && landtriger != 1)
      land_mark_pub_.publish(cmd);
    return;
  }
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
        traj_start_time = ros::Time::now();
        traj_initialized = true;
        ROS_INFO("Trajectory initialized at t = %.3f", traj_start_time.toSec());
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

void mpc_callback(const ros::TimerEvent &event){
  if (!traj_initialized)
  {
    return;
  }
  ros::Time now = ros::Time::now();
  int raw_index = static_cast<int>(((now - traj_sub_time).toSec() + 0.1)/ 0.1);
  int index_down = std::min(std::max(raw_index, 0), 15);
  int index_up = std::min(std::max(raw_index + 1, 0), 15);
  double weight_down = ((now - traj_sub_time).toSec() + 0.1)/ 0.1 - raw_index;
  double weight_up = 1 - weight_down;
  mpc_p = weight_down * trajectory_points[index_down] + weight_up * trajectory_points[index_up];
  mpc_v = weight_down * trajectory_v_points[index_down] + weight_up * trajectory_v_points[index_up];
}

void bppidCallback(const ros::TimerEvent &event) {
  if (triger_received_) {
    std::cout << "Final PID parameters for X axis: Kp=" << pid_x.get_parm().Kp 
          << ", Ki=" << pid_x.get_parm().Ki << ", Kd=" << pid_x.get_parm().Kd << std::endl;

      std::cout << "Final PID parameters for Y axis: Kp=" << pid_y.get_parm().Kp 
                << ", Ki=" << pid_y.get_parm().Ki << ", Kd=" << pid_y.get_parm().Kd << std::endl;

      std::cout << "Final PID parameters for Z axis: Kp=" << pid_z.get_parm().Kp 
                << ", Ki=" << pid_z.get_parm().Ki << ", Kd=" << pid_z.get_parm().Kd << std::endl;

  }
}

void auto_landing_detect(const ros::TimerEvent &event) {
  if (triger_received_) 
  {
    land_count += 1;
  }
  else return;
  Eigen::Vector3d delta_p = {target_p.x() - tracking_dist_ * std::cos(target_dog_yaw) - vins_p.x(), target_p.y() - tracking_dist_ * std::sin(target_dog_yaw) - vins_p.y(), target_p.z() - vins_p.z() + 0.05};
  
  // 创建绕 Z 轴的 2D 逆时针旋转
  Eigen::Rotation2D<double> rot(target_dog_yaw);

  // 只对 x 和 y 分量旋转
  Eigen::Vector2d rotated_xy = rot * delta_p.head<2>();
  Eigen::Vector3d rotated_delta_p(rotated_xy.x(), rotated_xy.y(), delta_p.z());

  if (!land_triger_received_ && std::fabs(rotated_delta_p[1]) < 0.2 && std::fabs(delta_p[2]) < 0.1 && delta_p[2] < 0.0)
  {
    Eigen::Vector3d delta_v = {target_v.x() - vins_v.x(), target_v.y() - vins_v.y(), target_v.z() - vins_v.z()};
    if (std::fabs(delta_v[0]) < 0.2 && std::fabs(delta_v[1]) < 0.2 && std::fabs(delta_v[2]) < 0.2 && land_count > 25)
    {
      std::cout << "Auto landing triggered!" << std::endl;
      // land_triger_received_ = true;
      geometry_msgs::PoseStamped land_pose;
      land_pose.pose.position.x = 0.0;
      land_triger_pub.publish(land_pose);
    
    }
  }
  // else {
  //   land_count -= 0.5;
  // }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle nh("~");

  ros::Subscriber heartbeat_sub = nh.subscribe("heartbeat", 10, heartbeatCallback);
  ros::Subscriber triger_sub_ = nh.subscribe("/triger", 10, triger_callback);
  ros::Subscriber land_triger_sub_ = nh.subscribe("/land_triger", 10, land_triger_callback);
  ros::Subscriber stop_triger_sub_ = nh.subscribe("/stop_triger", 10, stop_triger_callback);
  ros::Subscriber odom_sub_ = nh.subscribe("/vins_fusion/imu_propagate", 10, odom_callback);
  ros::Subscriber target_sub_ = nh.subscribe("/target_ekf_odom", 10, target_callback);
  // ros::Subscriber mpc_sub_ = nh.subscribe("/mpc", 10, mpc_callback);
  ros::Subscriber AOA_sub_ = nh.subscribe("/AOA_Tag_data", 10, AOA_callback);
  ros::Subscriber flow_sub_ = nh.subscribe("/flow_data", 10, flow_callback);

  ros::Subscriber traj_sub_ = nh.subscribe<nav_msgs::Path>("/drone2/planning/traj", 10, traj_callback);
  ros::Subscriber traj_v_sub_ = nh.subscribe<nav_msgs::Path>("/traj_v", 10, traj_v_callback);
  

  pos_cmd_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/position_cmd", 50);
  land_pub_ = nh.advertise<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land",1);
  land_mark_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/land_mark", 50);
  test_mark_pub_ = nh.advertise<quadrotor_msgs::PositionCommand>("/test_mark", 50);
  land_triger_pub = nh.advertise<geometry_msgs::PoseStamped>("/land_triger", 50);
  
  ros::Timer init_timer = nh.createTimer(ros::Duration(2.0), initCallback);

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.015), cmdCallback);

  // ros::Timer bppid_timer = nh.createTimer(ros::Duration(1), bppidCallback);

  ros::Timer mpc_timer = nh.createTimer(ros::Duration(0.01), mpc_callback);

  ros::Timer auto_land_timer = nh.createTimer(ros::Duration(0.2), auto_landing_detect);

  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}