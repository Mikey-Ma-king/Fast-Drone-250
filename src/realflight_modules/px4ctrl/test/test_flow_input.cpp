#include <gtest/gtest.h>
#include <limits>
#include "input.h"

namespace {

nav_msgs::Odometry::Ptr flowMessage(double height) {
  nav_msgs::Odometry::Ptr msg(new nav_msgs::Odometry);
  msg->pose.pose.position.x = 156.0;  // Actual publisher uses raw integrals here.
  msg->pose.pose.position.y = -108.0;
  msg->pose.pose.position.z = height;
  return msg;
}

TEST(FlowInput, AcceptsMetresWithoutOverwritingHorizontalStateOrVioFreshness) {
  ros::Time::init();
  Odom_Data_t odom;
  odom.p = Eigen::Vector3d(1.0, 2.0, 3.0);
  odom.v = Eigen::Vector3d(4.0, 5.0, 6.0);
  odom.q.setIdentity();
  odom.w.setZero();
  odom.rcv_stamp = ros::Time(123, 0);
  odom.recv_new_msg = false;
  odom.feed2(flowMessage(0.694));
  EXPECT_DOUBLE_EQ(odom.p.x(), 1.0);
  EXPECT_DOUBLE_EQ(odom.p.y(), 2.0);
  EXPECT_DOUBLE_EQ(odom.p.z(), 0.694);
  EXPECT_DOUBLE_EQ(odom.v.x(), 4.0);
  EXPECT_DOUBLE_EQ(odom.v.y(), 5.0);
  EXPECT_DOUBLE_EQ(odom.v.z(), 6.0);
  EXPECT_DOUBLE_EQ(odom.q.w(), 1.0);
  EXPECT_EQ(odom.rcv_stamp, ros::Time(123, 0));
  EXPECT_FALSE(odom.recv_new_msg);
  EXPECT_TRUE(odom.flow_flag);
}

TEST(FlowInput, RejectsInvalidOrUnavailableHeights) {
  Odom_Data_t odom;
  odom.p = Eigen::Vector3d(1.0, 2.0, 3.0);
  for (double height : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    odom.feed2(flowMessage(height));
    EXPECT_DOUBLE_EQ(odom.p.z(), 3.0);
    EXPECT_FALSE(odom.flow_flag);
    EXPECT_EQ(odom.rcv_stamp, ros::Time(0));
  }
}

TEST(FlowInput, FlowAloneDoesNotClaimCompleteOdometry) {
  Odom_Data_t odom;
  odom.feed2(flowMessage(0.05));
  EXPECT_EQ(odom.rcv_stamp, ros::Time(0));
  EXPECT_FALSE(odom.recv_new_msg);
}

TEST(FlowInput, IgnoresHeightAfterSwitchToVioStage) {
  Odom_Data_t odom;
  odom.p = Eigen::Vector3d(1.0, 2.0, 3.0);
  odom.Odom_State = 2;
  odom.feed2(flowMessage(0.694));
  EXPECT_DOUBLE_EQ(odom.p.z(), 3.0);
  EXPECT_FALSE(odom.flow_flag);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  ros::Time::init();
  return RUN_ALL_TESTS();
}
