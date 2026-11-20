#include <gtest/gtest.h>
#include <svo_odometry.h>
#include <limits>

using image::SvoOdometry;
const Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();

TEST(SvoOdometry, ConstantVelocityAt15_30_60HzAnd200HzQueries) {
  for (double hz : {15.0, 30.0, 60.0}) {
    SvoOdometry odom;
    const double t0 = 1000.0;
    const Eigen::Vector3d velocity(0.2, -0.3, 1.2);
    for (int i = 0; i < 10; ++i) {
      const double elapsed = i / hz;
      const double stamp = t0 + elapsed;
      odom.update(stamp, velocity * elapsed, identity, stamp + 0.004);
      if (i == 0) {
        EXPECT_FALSE(odom.sample(stamp + 0.004).valid);
        continue;
      }
      for (double dt : {0.005, 0.010, 0.015}) {
        const auto state = odom.sample(stamp + dt);
        ASSERT_TRUE(state.valid);
        EXPECT_NEAR((state.velocity - velocity).norm(), 0.0, 1e-10);
        EXPECT_NEAR((state.position - velocity * (elapsed + dt)).norm(), 0.0, 1e-10);
        EXPECT_DOUBLE_EQ(state.source_stamp, stamp);
        EXPECT_DOUBLE_EQ(state.stamp, stamp + dt);
      }
    }
  }
}

TEST(SvoOdometry, StaleMessagesAndRepeatingTimerCannotBecomeFreshState) {
  SvoOdometry odom;
  EXPECT_STREQ(odom.update(100.0, Eigen::Vector3d::Zero(), identity, 102.0), "stale_source");
  EXPECT_FALSE(odom.sample(102.0).valid);
  odom.update(102.0, Eigen::Vector3d::Zero(), identity, 102.0);
  odom.update(102.04, Eigen::Vector3d(0, 0, .04), identity, 102.04);
  EXPECT_TRUE(odom.sample(102.08).valid);
  EXPECT_FALSE(odom.sample(102.15).valid);
  EXPECT_FALSE(odom.sample(110.0).valid);
  EXPECT_STREQ(odom.update(102.06, Eigen::Vector3d(0, 0, .06), identity, 104.0), "stale_source");
  EXPECT_FALSE(odom.sample(104.0).valid);
}

TEST(SvoOdometry, DuplicateAndBackwardsTimestampsDoNotChangeAcceptedPose) {
  SvoOdometry odom;
  odom.update(100.0, Eigen::Vector3d::Zero(), identity, 100.0);
  odom.update(100.04, Eigen::Vector3d(0, 0, .04), identity, 100.04);
  EXPECT_STREQ(odom.update(100.04, Eigen::Vector3d(0, 0, 9), identity, 100.05), "non_monotonic_source");
  EXPECT_STREQ(odom.update(100.02, Eigen::Vector3d(0, 0, 9), identity, 100.05), "non_monotonic_source");
  EXPECT_NEAR(odom.sample(100.06).position.z(), .06, 1e-12);
}

TEST(SvoOdometry, LargeGapRequiresANewVelocityPair) {
  SvoOdometry odom;
  odom.update(100.0, Eigen::Vector3d::Zero(), identity, 100.0);
  odom.update(100.04, Eigen::Vector3d(0, 0, .04), identity, 100.04);
  EXPECT_STREQ(odom.update(101.0, Eigen::Vector3d(0, 0, 5), identity, 101.0), "input_gap");
  EXPECT_FALSE(odom.sample(101.0).valid);
  odom.update(101.04, Eigen::Vector3d(0, 0, 5.04), identity, 101.04);
  ASSERT_TRUE(odom.sample(101.05).valid);
  EXPECT_NEAR(odom.sample(101.05).velocity.z(), 1.0, 1e-12);
}

TEST(SvoOdometry, RejectsInvalidPoseFutureTimeAndExcessiveSpeed) {
  SvoOdometry odom;
  EXPECT_STREQ(odom.update(0, Eigen::Vector3d::Zero(), identity, 100), "invalid_pose");
  EXPECT_STREQ(odom.update(100.1, Eigen::Vector3d::Zero(), identity, 100), "future_source");
  EXPECT_STREQ(odom.update(100, Eigen::Vector3d::Zero(), Eigen::Quaterniond(0,0,0,0), 100), "invalid_pose");
  EXPECT_STREQ(odom.update(100, Eigen::Vector3d(0,0,std::numeric_limits<double>::quiet_NaN()), identity, 100), "invalid_pose");
  odom.update(100, Eigen::Vector3d::Zero(), identity, 100);
  EXPECT_STREQ(odom.update(100.04, Eigen::Vector3d(0,0,1), identity, 100.04), "speed_limit");
  EXPECT_FALSE(odom.sample(100.05).valid);
}

TEST(SvoOdometry, ReversingMotionChangesVelocityWithoutLongHistory) {
  SvoOdometry odom;
  odom.update(100, Eigen::Vector3d(0,0,0), identity, 100);
  odom.update(100.04, Eigen::Vector3d(0,0,.04), identity, 100.04);
  ASSERT_GT(odom.sample(100.04).velocity.z(), 0);
  odom.update(100.08, Eigen::Vector3d(0,0,0), identity, 100.08);
  ASSERT_TRUE(odom.sample(100.08).valid);
  EXPECT_NEAR(odom.sample(100.08).velocity.z(), -1, 1e-12);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
