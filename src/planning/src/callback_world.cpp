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
extern Eigen::Vector3d hc14_dog_pos;
extern int hc14_dog_pos_count;
extern double hc14_dog_yaw;
extern double hc14_dog_yaw_rate;  // 狗通信角速度
extern bool hc14_offset_yaw_ready;
extern bool hc14_offset_pos_ready;

extern double AOA_x;
extern double AOA_w;
extern double flow_z;

extern double xy_p;
extern double xy_i;
extern double xy_d;
extern double z_p;
extern double z_i;
extern double z_d;

extern double intergral_targetx;
extern double intergral_targety;
extern double intergral_targetz;

extern int triger_mode;
extern int land_lock_timer;
extern double mode_vins_z;
extern double mode_vins_vel_z;
extern bool reflight_complete;
extern bool traj_initialized;
extern bool last_cmd_initialized;
extern ros::Time heartbeat_time_;
bool odom_received_ = false;
extern Eigen::Vector2d hc14_dog_acc;
extern double hc14_perception_confidence;

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
}

void mode_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    int new_triger_mode = msgPtr->pose.orientation.w;
    std::cout << "new_triger_mode: " << new_triger_mode << std::endl;
    if (new_triger_mode == -1){
        intergral_targetx = 0;
        intergral_targety = 0;
        intergral_targetz = 0;
    }else if (new_triger_mode == 0){
        if (triger_mode == -1){
            last_cmd_initialized = false;
        }
        traj_initialized = false;

    }else if (new_triger_mode == 1){
        if (triger_mode == -1){
            last_cmd_initialized = false;
        }
        mode_vins_z = vins_p.z();
        mode_vins_vel_z = vins_v.z();
        reflight_complete = false;
    }else if (new_triger_mode == 2){
        if (triger_mode == -1){
            last_cmd_initialized = false;
        }
        land_lock_timer = 0;
        mode_vins_z = vins_p.z();
        mode_vins_vel_z = vins_v.z();
    }


    triger_mode = new_triger_mode;
}

void heartbeatCallback(const std_msgs::EmptyConstPtr &msg) {
    heartbeat_time_ = ros::Time::now();
}


void dog_pos_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    // 直接使用处理后的dog速度数据（狗坐标系）
    hc14_dog_vel.x() = msg->twist.twist.linear.x;
    hc14_dog_vel.y() = msg->twist.twist.linear.y;
    hc14_dog_vel.z() = msg->twist.twist.linear.z;

    hc14_dog_pos.x() = msg->pose.pose.position.x;
    hc14_dog_pos.y() = msg->pose.pose.position.y;
    hc14_dog_pos.z() = msg->pose.pose.position.z;
    
    // 使用处理后的yaw（在twist.angular.x中）
    hc14_dog_yaw = msg->twist.twist.angular.x;
    
    // 读取狗通信角速度（在twist.angular.y中）
    hc14_dog_yaw_rate = msg->twist.twist.angular.y;
    
    // 从orientation.w和x读取precise_pos_offset_ready和precise_yaw_offset_ready状态
    hc14_offset_pos_ready = (msg->pose.pose.orientation.w > 0.5);
    hc14_offset_yaw_ready = (msg->pose.pose.orientation.x > 0.5);
    
    // 从orientation.y和z读取加速度（世界坐标系）
    hc14_dog_acc.x() = msg->pose.pose.orientation.y;
    hc14_dog_acc.y() = msg->pose.pose.orientation.z;
    
    // 从twist.angular.z读取感知置信度
    hc14_perception_confidence = msg->twist.twist.angular.z;

    hc14_dog_pos_count++;  // 收到包时计数器+1 
}