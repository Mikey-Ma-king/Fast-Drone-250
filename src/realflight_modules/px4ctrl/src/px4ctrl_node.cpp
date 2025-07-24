#include <ros/ros.h>
#include "PX4CtrlFSM.h"
#include <signal.h>
#include <fstream>

double vins_pos_x;
double vins_pos_y;
double vins_pos_z;
double vins_yaw;
double dog_pos_x;
double dog_pos_y;
double dog_pos_z;
double dog_pos_yaw;
bool takeoff_flag = false;

void mySigintHandler(int sig)
{
    ROS_INFO("[PX4Ctrl] exit...");
    ros::shutdown();
}

void vins_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    vins_pos_x = msg->pose.pose.position.x;
    vins_pos_y = msg->pose.pose.position.y;
    vins_pos_z = msg->pose.pose.position.z;
    double vins_q_w = msg->pose.pose.orientation.w;
    double vins_q_x = msg->pose.pose.orientation.x;
    double vins_q_y = msg->pose.pose.orientation.y;
    double vins_q_z = msg->pose.pose.orientation.z;
    // 计算偏航角
    double siny_cosp = 2.0 * (vins_q_w * vins_q_z + vins_q_x * vins_q_y);
    double cosy_cosp = 1.0 - 2.0 * (vins_q_y * vins_q_y + vins_q_z * vins_q_z);
    vins_yaw = std::atan2(siny_cosp, cosy_cosp);
}

void dog_pos_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    dog_pos_x = msg->pose.pose.position.x;
    dog_pos_y = msg->pose.pose.position.y;
    dog_pos_z = msg->pose.pose.position.z;
    dog_pos_yaw = msg->pose.pose.orientation.w;
}

void writeCoordinatesToFile(double vins_p_x, double vins_p_y, double vins_p_z, double yaw1,
    double dog_p_x, double dog_p_y, double dog_p_z, double yaw2,
    const std::string& filename = "/home/pc/Fast-Drone-250/coordinate.txt") {
    std::ofstream file(filename);
    if (!file.is_open()) {
    std::cerr << "无法打开文件 " << filename << " 进行写入。" << std::endl;
    return;
    }

    file << vins_p_x << "," << vins_p_y << "," << vins_p_z << std::endl;
    file << yaw1 << std::endl;
    file << dog_p_x << "," << dog_p_y << "," << dog_p_z << std::endl;
    file << yaw2 << std::endl;

    file.close();
    std::cout << "写入成功: " << filename << std::endl;
}

void takeoff_callback(const quadrotor_msgs::TakeoffLand::ConstPtr& msg) {
    if (takeoff_flag == false && msg->takeoff_land_cmd == 1) {
        takeoff_flag = true;
        writeCoordinatesToFile(vins_pos_x, vins_pos_y, vins_pos_z, dog_pos_yaw, dog_pos_x, dog_pos_y, dog_pos_z, vins_yaw);

    }
    else if (takeoff_flag == true && msg->takeoff_land_cmd == 2) {
        takeoff_flag = false;
    }
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "px4ctrl");
    ros::NodeHandle nh("~");

    signal(SIGINT, mySigintHandler);
    ros::Duration(1.0).sleep();

    Parameter_t param;
    param.config_from_ros_handle(nh);

    // Controller controller(param);
    LinearControl controller(param);
    PX4CtrlFSM fsm(param, controller);


    ros::Subscriber state_sub =
        nh.subscribe<mavros_msgs::State>("/mavros/state",
                                         10,
                                         boost::bind(&State_Data_t::feed, &fsm.state_data, _1));

    ros::Subscriber extended_state_sub =
        nh.subscribe<mavros_msgs::ExtendedState>("/mavros/extended_state",
                                                 10,
                                                 boost::bind(&ExtendedState_Data_t::feed, &fsm.extended_state_data, _1));

    ros::Subscriber odom_sub =
        nh.subscribe<nav_msgs::Odometry>("odom",
                                         100,
                                         boost::bind(&Odom_Data_t::feed1, &fsm.odom_data, _1),
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());
    
    ros::Subscriber vins_sub_ = nh.subscribe<nav_msgs::Odometry>("/vins_fusion/imu_propagate", 10, vins_callback);
    ros::Subscriber dog_pos_sub = nh.subscribe<nav_msgs::Odometry>("/dog_pos",10,dog_pos_callback);
    ros::Subscriber takeoff_sub = nh.subscribe<quadrotor_msgs::TakeoffLand>("/px4ctrl/takeoff_land", 10, takeoff_callback);

    ros::Subscriber flow_sub =
        nh.subscribe<flow_publisher::FlowDataMsg>("/flow_data",
                                         100,
                                         boost::bind(&Odom_Data_t::feed2, &fsm.odom_data, _1),
                                         ros::VoidConstPtr(),
                                         ros::TransportHints().tcpNoDelay());
    // ros::Subscriber flow_imu_sub =
    //     nh.subscribe<sensor_msgs::Imu>("/mavros/imu/data", // Note: do NOT change it to /mavros/imu/data_raw !!!
    //                                    100,
    //                                    boost::bind(&Odom_Data_t::feed3, &fsm.odom_data, _1),
    //                                    ros::VoidConstPtr(),
    //                                    ros::TransportHints().tcpNoDelay());
    ros::Subscriber cmd_sub =
        nh.subscribe<quadrotor_msgs::PositionCommand>("cmd",
                                                      100,
                                                      boost::bind(&Command_Data_t::feed, &fsm.cmd_data, _1),
                                                      ros::VoidConstPtr(),
                                                      ros::TransportHints().tcpNoDelay());

    ros::Subscriber imu_sub =
        nh.subscribe<sensor_msgs::Imu>("/mavros/imu/data", // Note: do NOT change it to /mavros/imu/data_raw !!!
                                       100,
                                       boost::bind(&Imu_Data_t::feed, &fsm.imu_data, _1),
                                       ros::VoidConstPtr(),
                                       ros::TransportHints().tcpNoDelay());

    ros::Subscriber rc_sub;
    if (!param.takeoff_land.no_RC) // mavros will still publish wrong rc messages although no RC is connected
    {
        rc_sub = nh.subscribe<mavros_msgs::RCIn>("/mavros/rc/in",
                                                 10,
                                                 boost::bind(&RC_Data_t::feed, &fsm.rc_data, _1));
    }

    ros::Subscriber bat_sub =
        nh.subscribe<sensor_msgs::BatteryState>("/mavros/battery",
                                                100,
                                                boost::bind(&Battery_Data_t::feed, &fsm.bat_data, _1),
                                                ros::VoidConstPtr(),
                                                ros::TransportHints().tcpNoDelay());

    ros::Subscriber takeoff_land_sub =
        nh.subscribe<quadrotor_msgs::TakeoffLand>("takeoff_land",
                                                  2,
                                                  boost::bind(&Takeoff_Land_Data_t::feed, &fsm.takeoff_land_data, _1),
                                                  ros::VoidConstPtr(),
                                                  ros::TransportHints().tcpNoDelay());

    fsm.ctrl_FCU_pub = nh.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 10);
    fsm.traj_start_trigger_pub = nh.advertise<geometry_msgs::PoseStamped>("/traj_start_trigger", 10);

    fsm.debug_pub = nh.advertise<quadrotor_msgs::Px4ctrlDebug>("/debugPx4ctrl", 10); // debug

    fsm.set_FCU_mode_srv = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    fsm.arming_client_srv = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    fsm.reboot_FCU_srv = nh.serviceClient<mavros_msgs::CommandLong>("/mavros/cmd/command");

    ros::Duration(0.5).sleep();

    if (param.takeoff_land.no_RC)
    {
        ROS_WARN("PX4CTRL] Remote controller disabled, be careful!");
    }
    else
    {
        ROS_INFO("PX4CTRL] Waiting for RC");
        while (ros::ok())
        {
            ros::spinOnce();
            if (fsm.rc_is_received(ros::Time::now()))
            {
                ROS_INFO("[PX4CTRL] RC received.");
                break;
            }
            ros::Duration(0.1).sleep();
        }
    }

    int trials = 0;
    while (ros::ok() && !fsm.state_data.current_state.connected)
    {
        ros::spinOnce();
        ros::Duration(1.0).sleep();
        if (trials++ > 5)
            ROS_ERROR("Unable to connnect to PX4!!!");
    }

    ros::Rate r(param.ctrl_freq_max);
    while (ros::ok())
    {
        r.sleep();
        ros::spinOnce();
        fsm.process(); // We DO NOT rely on feedback as trigger, since there is no significant performance difference through our test.
    }

    return 0;
}
