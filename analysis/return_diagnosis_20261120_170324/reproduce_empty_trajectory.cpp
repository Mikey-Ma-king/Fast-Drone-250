// Offline reproduction using callback bodies extracted verbatim from traj_server.cpp.
// No ROS master, publishers, controller, or hardware is started.
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
namespace ros {
struct TimerEvent {};
struct Duration { double value; double toSec() const { return value; } };
struct Time { double value=0; static Time now() { return Time{1.0}; } };
Duration operator-(Time a, Time b) { return {a.value-b.value}; }
}
namespace nav_msgs {
struct Path {
 using ConstPtr=std::shared_ptr<const Path>;
 struct Pose { struct { struct {double x=0,y=0,z=0;} position; } pose; };
 std::vector<Pose> poses;
};
}
bool traj_initialized=false;
ros::Time traj_sub_time;
std::vector<Eigen::Vector3d> trajectory_points,trajectory_v_points,trajectory_a_points;
Eigen::Vector3d mpc_p,mpc_v,mpc_a;
void traj_v_callback(const nav_msgs::Path::ConstPtr& msg) {
    if (!traj_initialized && !msg->poses.empty()) {
        traj_initialized = true;
    }
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
  double dt = (now - traj_sub_time).toSec();  // 距离轨迹接收的经过时间
  double dt_step = 0.1;                        // 轨迹点时间步长(对应 MPC 的 dt=0.1s)

  dt += 0.0;  // 预留的提前量参数(可调整为正值以取未来点)

  // 计算浮点索引(可能带小数)
  double idx_float = dt / dt_step;

  // 找到相邻的两个整数索引
  int idx_down = static_cast<int>(std::floor(idx_float));
  int idx_up = idx_down + 1;

  // 边界处理
  int max_idx = static_cast<int>(trajectory_points.size() - 1);
  idx_down = std::max(0, std::min(idx_down, max_idx));
  idx_up = std::max(0, std::min(idx_up, max_idx));

  // 超出轨迹范围: 保持在最后一点
  if (idx_down >= max_idx) {
    mpc_p = trajectory_points[max_idx];
    mpc_v = trajectory_v_points[max_idx];
    if (!trajectory_a_points.empty() && max_idx < static_cast<int>(trajectory_a_points.size())) {
      mpc_a = trajectory_a_points[max_idx];
    }
    return;
  }

  // 线性插值: 根据小数权重在两个相邻点之间加权
  double weight_up = idx_float - idx_down;      // 上权重(离上一个点多远)
  double weight_down = 1.0 - weight_up;          // 下权重(离下一个点多远)

  // 位置和速度的加权插值
  mpc_p = weight_down * trajectory_points[idx_down] + weight_up * trajectory_points[idx_up];
  mpc_v = weight_down * trajectory_v_points[idx_down] + weight_up * trajectory_v_points[idx_up];

  // 加速度同理
  if (!trajectory_a_points.empty() &&
      idx_down < static_cast<int>(trajectory_a_points.size()) &&
      idx_up < static_cast<int>(trajectory_a_points.size())) {
    mpc_a = weight_down * trajectory_a_points[idx_down] + weight_up * trajectory_a_points[idx_up];
  }
}
int main() {
 auto velocity=std::make_shared<nav_msgs::Path>();
 velocity->poses.resize(2);
 traj_v_callback(velocity); // Velocity arrives before the first position Path.
 mpc_callback(ros::TimerEvent{}); // Timer runs before the position callback.
}
