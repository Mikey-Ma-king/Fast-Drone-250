#include "callback.h"

// 告诉编译器：这些变量在别处定义，我只是来用它的
extern Eigen::Vector3d vins_p;
extern Eigen::Vector3d vins_v;
extern double vins_yaw;

extern Eigen::Vector3d target_p;
extern Eigen::Vector3d target_v;
extern double target_yaw;
extern double last_target_yaw;
extern double target_dog_yaw;
extern bool target_receive;
extern int target_count;

extern Eigen::Vector3d hc14_dog_vel;
extern int hc14_dog_pos_count;

extern double AOA_x;
extern double AOA_w;
double last_flow_z;
extern double flow_z;
extern ros::Time flow_timer;
extern bool flow_detect;

extern double x_p;
extern double x_i;
extern double x_d;
extern double y_p;
extern double y_i;
extern double y_d;
extern double z_p;
extern double z_i;
extern double z_d;

extern bool triger_received_;
extern bool land_triger_received_;
extern bool stop_triger_received_;
extern ros::Time heartbeat_time_;
bool odom_received_ = false;

void initCallback(const ros::TimerEvent &event) {
    if (!odom_received_)
        std::cout<< "no odom!" << std::endl;
}

void odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    vins_p.x() = msg->pose.pose.position.x;
    vins_p.y() = msg->pose.pose.position.y;
    vins_p.z() = msg->pose.pose.position.z;
    vins_v.x() = msg->twist.twist.linear.x;
    vins_v.y() = msg->twist.twist.linear.y;
    vins_v.z() = msg->twist.twist.linear.z;
    
    double vins_q_w = msg->pose.pose.orientation.w;
    double vins_q_x = msg->pose.pose.orientation.x;
    double vins_q_y = msg->pose.pose.orientation.y;
    double vins_q_z = msg->pose.pose.orientation.z;
    // 计算偏航角
    double siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y);
    double cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z);
    vins_yaw = std::atan2(siny_cosp, cosy_cosp);
    if (!odom_received_)
        odom_received_ = true;
}

class YawSmoother {
    private:
        double alpha;
        double smoothed_yaw;
        bool initialized;
        double last_raw_yaw;    // 上一次输入的原始yaw
        double hold_duration;
        double timeout_threshold;
    
        static constexpr double PI = 3.14159265358979323846;
        static constexpr double TWO_PI = 2.0 * PI;
    
    public:
        YawSmoother(double lpf_cutoff_freq = 5.0, double dt = 1.0 / 30.0)
            : smoothed_yaw(0.0), initialized(false), hold_duration(0.0), timeout_threshold(0.5) {
            if (dt <= 0.0) {
                throw std::invalid_argument("dt must be positive");
            }
            double rc = 1.0 / (2.0 * M_PI * lpf_cutoff_freq);
            alpha = dt / (dt + rc);
        }
    
        double update(double raw_yaw) {
            if (!initialized) {
                smoothed_yaw = raw_yaw;
                last_raw_yaw = raw_yaw;
                initialized = true;
                return smoothed_yaw;
            }
    
            double unwrapped_yaw = unwrap(raw_yaw, last_raw_yaw);
            last_raw_yaw = unwrapped_yaw;  // 更新上一帧
    
            smoothed_yaw = alpha * unwrapped_yaw + (1.0 - alpha) * smoothed_yaw;
            hold_duration = 0.0;
            return smoothed_yaw;
        }
    
        double tick(double dt) {
            if (!initialized) {
                return 0.0;
            }
            hold_duration += dt;
            if (hold_duration > timeout_threshold) {
                return smoothed_yaw;
            }
            smoothed_yaw = (1.0 - alpha) * smoothed_yaw;
            return smoothed_yaw;
        }
    
        void reset() {
            initialized = false;
            smoothed_yaw = 0.0;
            hold_duration = 0.0;
        }
    
    private:
        // 角度unwrap函数
        double unwrap(double new_yaw, double reference_yaw) {
            double diff = new_yaw - reference_yaw;
            if (diff > PI) {
                new_yaw -= TWO_PI;
            } else if (diff < -PI) {
                new_yaw += TWO_PI;
            }
            return new_yaw;
        }
};

YawSmoother yaw_filter(5.0, 1.0/30.0);

void target_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    target_p.x() = msg->pose.pose.position.x + 0.00*std::sin(target_dog_yaw);
    target_p.y() = msg->pose.pose.position.y - 0.00*std::cos(target_dog_yaw);
    target_p.z() = std::max(msg->pose.pose.position.z,vins_p.z()-5.0);
    target_v.x() = msg->twist.twist.linear.x;
    target_v.y() = msg->twist.twist.linear.y;
    target_v.z() = 0;
    if (target_v.norm() > 2)
    {
      target_v.x() = 0;
      target_v.y() = 0;
      target_v.z() = 0;
    }
    double vins_q_w = msg->pose.pose.orientation.w;
    // double vins_q_x = msg->pose.pose.orientation.x;
    // double vins_q_y = msg->pose.pose.orientation.y;
    // double vins_q_z = msg->pose.pose.orientation.z;
    // 计算偏航角
    // double siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y);
    // double cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z);
    // target_yaw = std::atan2(siny_cosp, cosy_cosp);
    // target_yaw = 0.6* last_target_yaw + 0.4 * target_yaw;
    // target_yaw = 0;
    // last_target_yaw = target_yaw;
    // double yaw = 2 * std::atan2(vins_q_z, vins_q_w);
    // yaw -= 3.141593;
    // std::cout<<"yaw:"<<target_yaw<<std::endl;
    target_count ++;
    // target_dog_yaw = vins_q_w;
    target_dog_yaw = yaw_filter.update(vins_q_w);
}

void AOA_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    AOA_x = msg->pose.pose.position.x;
    AOA_w = msg->pose.pose.orientation.w;
}

void flow_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    flow_z = msg->pose.pose.position.z;
    if (last_flow_z == 0)
        last_flow_z = flow_z;
    if ((last_flow_z - flow_z > 0.3 || flow_detect) && (vins_p - target_p).norm() < 3 && land_triger_received_) {
        flow_detect = true;
        last_flow_z = flow_z;
        std::cout << "flow detected" << std::endl;
        return;
    }
    flow_detect = false;
    flow_timer = ros::Time::now();
    last_flow_z = flow_z;
}

void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    triger_received_ = true;
}

void land_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    land_triger_received_ = true;
}

void stop_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    stop_triger_received_ = true;
}

void heartbeatCallback(const std_msgs::EmptyConstPtr &msg) {
    heartbeat_time_ = ros::Time::now();
}

void pid_callback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
    ROS_INFO("Received PID Gains:");
    ROS_INFO("  X: Kp=%f, Ki=%f, Kd=%f", msg->data[0], msg->data[1], msg->data[2]);
    ROS_INFO("  Y: Kp=%f, Ki=%f, Kd=%f", msg->data[3], msg->data[4], msg->data[5]);
    ROS_INFO("  Z: Kp=%f, Ki=%f, Kd=%f", msg->data[6], msg->data[7], msg->data[8]);
    x_p = msg->data[0];
    x_i = msg->data[1];
    x_d = msg->data[2];
    y_p = msg->data[3];
    y_i = msg->data[4];
    y_d = msg->data[5];
    z_p = msg->data[6];
    z_i = msg->data[7];
    z_d = msg->data[8];
}

void dog_pos_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 更新dog的速度信息
    hc14_dog_vel.x() = msg->twist.twist.linear.x;
    hc14_dog_vel.y() = msg->twist.twist.linear.y;
    hc14_dog_vel.z() = msg->twist.twist.linear.z;
    
    // 给狗的速度加上上下限
    const double max_dog_velocity = 1.5;  // 最大速度限制 (m/s)
    const double min_dog_velocity = -1.5; // 最小速度限制 (m/s)
    
    hc14_dog_vel.x() = std::max(min_dog_velocity, std::min(max_dog_velocity, hc14_dog_vel.x()));
    hc14_dog_vel.y() = std::max(min_dog_velocity, std::min(max_dog_velocity, hc14_dog_vel.y()));
    hc14_dog_vel.z() = std::max(min_dog_velocity, std::min(max_dog_velocity, hc14_dog_vel.z()));
    
    hc14_dog_pos_count++;  // 收到包时计数器+1
}