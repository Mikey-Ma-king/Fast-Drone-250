#!/usr/bin/env python3
"""Exercise the real trajectory callbacks offline, without starting ROS."""
import argparse
from pathlib import Path
import os
import subprocess
import tempfile


def function(source, name):
    start = source.index("void " + name + "(")
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


PREAMBLE = r"""
#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
namespace ros {
struct TimerEvent {};
struct Duration { double value; double toSec() const { return value; } };
double test_now = 10.0;
struct Time { double value=0; static Time now() { return Time{test_now}; } };
Duration operator-(Time a, Time b) { return {a.value-b.value}; }
}
namespace nav_msgs {
struct Path {
  using ConstPtr = std::shared_ptr<const Path>;
  struct Pose { struct { struct {double x=0,y=0,z=0;} position; } pose; };
  std::vector<Pose> poses;
};
}
bool traj_initialized = false, command_passed = false;
bool hc14_dog_pos_received = true, hc14_offset_yaw_ready = true;
int triger_mode = 0;
ros::Time traj_sub_time;
std::vector<Eigen::Vector3d> trajectory_points, trajectory_v_points, trajectory_a_points;
Eigen::Vector3d mpc_p, mpc_v, mpc_a;
"""

CASES = r"""
auto path(int count, double first) {
  auto msg = std::make_shared<nav_msgs::Path>();
  msg->poses.resize(count);
  for (int i=0; i<count; ++i) msg->poses[i].pose.position.x = first+i;
  return msg;
}
void reset() {
  traj_initialized = false;
  trajectory_points.clear(); trajectory_v_points.clear(); trajectory_a_points.clear();
  mpc_p = mpc_v = mpc_a = Eigen::Vector3d::Constant(-99);
  ros::test_now = 10.0; triger_mode = 0;
}
void tick() {
  // Deliberately run interpolation before checking command guards, as at the crash.
  mpc_callback(ros::TimerEvent{});
  command_passed = false;
  cmdCallback(ros::TimerEvent{});
}
void matched() { traj_callback(path(3, 10)); traj_v_callback(path(3, 1)); }
int main() {
  reset(); traj_v_callback(path(3, 1)); tick();
  assert(!command_passed && mpc_p.x() == -99);
  traj_callback(path(3, 10)); tick();
  assert(command_passed && mpc_p.x() == 10 && mpc_v.x() == 1);
  std::cout << "PASS velocity before position; resumes after position arrives\n";

  reset(); traj_callback(path(3, 10)); tick();
  assert(!command_passed && mpc_p.x() == -99);
  traj_v_callback(path(3, 1)); tick(); assert(command_passed && mpc_p.x() == 10);
  std::cout << "PASS position before velocity\n";

  reset(); matched(); tick(); traj_callback(path(0, 0)); tick();
  assert(!command_passed && mpc_p.x() == 10);
  std::cout << "PASS empty position update\n";

  reset(); matched(); tick(); traj_v_callback(path(0, 0)); tick();
  assert(!command_passed && mpc_v.x() == 1);
  std::cout << "PASS empty velocity update\n";

  reset(); matched(); traj_v_callback(path(1, 1)); ros::test_now += 1; tick();
  assert(!command_passed && mpc_p.x() == -99);
  traj_v_callback(path(4, 1)); tick(); assert(!command_passed);
  traj_v_callback(path(3, 1)); tick(); assert(command_passed && mpc_p.x() == 12);
  std::cout << "PASS mismatched lengths; resumes when lengths match\n";

  reset(); matched(); traj_a_callback(path(3, 3)); ros::test_now += 0.05; tick();
  assert(command_passed && std::abs(mpc_p.x()-10.5)<1e-9);
  assert(std::abs(mpc_v.x()-1.5)<1e-9 && std::abs(mpc_a.x()-3.5)<1e-9);
  ros::test_now += 1; tick();
  assert(mpc_p.x() == 12 && mpc_v.x() == 3 && mpc_a.x() == 5);
  std::cout << "PASS normal interpolation and endpoint\n";

  reset(); traj_callback(path(1, 10)); traj_v_callback(path(1, 1)); tick();
  assert(command_passed && mpc_p.x() == 10 && mpc_v.x() == 1);
  std::cout << "PASS single point, optional acceleration absent\n";

  reset(); matched(); traj_a_callback(path(1, 3)); ros::test_now += 0.05; tick();
  assert(command_passed && mpc_a.x() == -99);
  std::cout << "PASS short optional acceleration\n";

  reset(); matched(); traj_initialized = false; tick(); assert(!command_passed);
  std::cout << "PASS reset readiness is preserved\n";
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "src/traj_server.cpp")
    args = parser.parse_args()
    source = args.source.read_text()
    # Use the real command entry guards, stopping before control or publishing.
    command_guards = function(source, "cmdCallback").split(
        "quadrotor_msgs::PositionCommand cmd;", 1)[0] + "command_passed = true;\n}\n"
    callbacks = "\n".join(function(source, name) for name in (
        "traj_callback", "traj_v_callback", "traj_a_callback", "mpc_callback"))
    with tempfile.TemporaryDirectory(prefix="quaddock_trajectory_test_") as directory:
        cpp = Path(directory) / "test.cpp"
        binary = Path(directory) / "test"
        cpp.write_text(PREAMBLE + callbacks + command_guards + CASES)
        subprocess.run(["g++", "-std=c++14", "-O0", "-g", "-fsanitize=address,undefined",
                        "-fno-omit-frame-pointer", "-I/usr/include/eigen3", str(cpp),
                        "-o", str(binary)], check=True)
        # LeakSanitizer cannot inspect threads in the ptrace-based sandbox.
        # Keep AddressSanitizer and UndefinedBehaviorSanitizer enabled.
        env = dict(os.environ)
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
        subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    main()
