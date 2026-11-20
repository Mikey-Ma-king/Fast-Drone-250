#ifndef READ_SVO_ODOMETRY_H
#define READ_SVO_ODOMETRY_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace image {

// No ROS I/O: source time controls differentiation and the extrapolation limit.
class SvoOdometry {
 public:
  struct Config {
    double max_source_age_s = 0.10;
    double max_input_gap_s = 0.20;
    double max_future_s = 0.01;
    double max_speed_mps = 4.0;
  };
  struct State {
    bool valid = false;
    const char* reason = "waiting_for_pose";
    double stamp = 0.0;
    double source_stamp = 0.0;
    double receipt_stamp = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
  };

  SvoOdometry() = default;
  explicit SvoOdometry(const Config& config) : config_(config) {
    if (!positive(config.max_source_age_s) || !positive(config.max_input_gap_s) ||
        !positive(config.max_future_s) || !positive(config.max_speed_mps)) {
      throw std::invalid_argument("read odometry limits must be finite and positive");
    }
  }

  const char* update(double stamp, const Eigen::Vector3d& position,
                     const Eigen::Quaterniond& orientation, double receipt) {
    last_input_stamp_ = stamp;
    last_input_receipt_ = receipt;
    if (!positive(stamp) || !positive(receipt) || !position.allFinite() ||
        !orientation.coeffs().allFinite() || orientation.norm() < 1e-6) {
      return last_input_ = "invalid_pose";
    }
    if (receipt - stamp > config_.max_source_age_s) {
      return last_input_ = "stale_source";
    }
    if (stamp - receipt > config_.max_future_s) {
      return last_input_ = "future_source";
    }
    if (have_pose_ && stamp <= state_.source_stamp) {
      return last_input_ = "non_monotonic_source";
    }
    const double dt = stamp - state_.source_stamp;
    bool velocity_valid = false;
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    const char* reason = have_pose_ ? "input_gap" : "waiting_for_velocity";
    if (have_pose_ && dt <= config_.max_input_gap_s) {
      velocity = (position - state_.position) / dt;
      velocity_valid = velocity.allFinite() && velocity.norm() < config_.max_speed_mps;
      reason = velocity_valid ? "ok" : "speed_limit";
    }
    state_.source_stamp = stamp;
    state_.receipt_stamp = receipt;
    state_.position = position;
    state_.orientation = orientation.normalized();
    state_.velocity = velocity;
    state_.valid = velocity_valid;
    state_.reason = reason;
    have_pose_ = true;
    return last_input_ = reason;
  }

  State sample(double now) const {
    State result = state_;
    result.stamp = now;
    if (!have_pose_) return result;
    const double age = now - state_.source_stamp;
    if (!std::isfinite(now) || age > config_.max_source_age_s || age < -config_.max_future_s) {
      result.valid = false;
      result.reason = "stale_source";
    }
    if (result.valid) {
      // Position and velocity use the same causal constant-velocity model.
      result.position += result.velocity * age;
    }
    return result;
  }

  const char* lastInput() const { return last_input_; }
  double lastInputStamp() const { return last_input_stamp_; }
  double lastInputReceipt() const { return last_input_receipt_; }

 private:
  static bool positive(double value) { return std::isfinite(value) && value > 0.0; }
  Config config_;
  State state_;
  bool have_pose_ = false;
  const char* last_input_ = "none";
  double last_input_stamp_ = 0.0;
  double last_input_receipt_ = 0.0;
};

}  // namespace image
#endif
