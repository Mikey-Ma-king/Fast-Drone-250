#pragma once

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
                                                     
#include <Eigen/Core>
#include <atomic>
#include <thread>
#include <cmath>
void initCallback(const ros::TimerEvent &event);
void odom_callback(const nav_msgs::Odometry::ConstPtr& msg);
void target_callback(const nav_msgs::Odometry::ConstPtr& msg);
void AOA_callback(const nav_msgs::Odometry::ConstPtr& msg);
void flow_callback(const nav_msgs::Odometry::ConstPtr& msg);

void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr);
void land_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr);
void stop_triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr);
void heartbeatCallback(const std_msgs::EmptyConstPtr &msg);