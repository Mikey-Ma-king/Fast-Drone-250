#!/usr/bin/env python3
"""Check real mode_callback frame transfers offline, without starting ROS."""
import os
from pathlib import Path
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
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
namespace geometry_msgs {
struct PoseStamped { struct { struct { double w=0; } orientation; } pose; };
using PoseStampedConstPtr = std::shared_ptr<const PoseStamped>;
}
Eigen::Vector3d vins_p{0,0,2}, vins_v{0,0,0.2};
double vins_yaw=-0.7, hc14_dog_yaw=0.0;
double intergral_targetx=0, intergral_targety=0, intergral_targetz=0.1;
double intergral_targetx_body=0, intergral_targety_body=0;
int triger_mode=-1, land_lock_timer=0;
double mode_vins_z=0, mode_vins_vel_z=0;
bool reflight_complete=true, traj_initialized=true, last_cmd_initialized=true;
"""

CASES = r"""
void mode(int value) {
  auto msg=std::make_shared<geometry_msgs::PoseStamped>();
  msg->pose.orientation.w=value;
  mode_callback(msg);
  assert(triger_mode==value);
}
bool near(double a, double b) { return std::abs(a-b)<1e-12; }
int main() {
  // At +90 degrees, world +X is dog -Y; UAV yaw deliberately differs.
  hc14_dog_yaw=M_PI/2;
  for (int world : {0,-2}) for (int body : {1,2}) {
    triger_mode=world; intergral_targetx=0.1; intergral_targety=0.2;
    mode(body);
    assert(near(intergral_targetx_body,0.2) && near(intergral_targety_body,-0.1));
    mode(world);
    assert(near(intergral_targetx,0.1) && near(intergral_targety,0.2));
  }
  std::cout << "PASS all world/dog mode pairs at 90 degrees\n";
  for (double yaw : {0.0,0.31,-1.7,M_PI}) {
    hc14_dog_yaw=yaw; triger_mode=0;
    intergral_targetx=-0.17; intergral_targety=0.24;
    mode(1); mode(-2);
    assert(near(intergral_targetx,-0.17) && near(intergral_targety,0.24));
  }
  std::cout << "PASS round trips preserve world integral\n";
  triger_mode=1; intergral_targetx_body=0.12; intergral_targety_body=-0.09;
  mode(2); mode(1); mode(1);
  assert(near(intergral_targetx_body,0.12) && near(intergral_targety_body,-0.09));
  triger_mode=0; intergral_targetx=0.11; intergral_targety=-0.21;
  mode(-2); mode(0);
  assert(near(intergral_targetx,0.11) && near(intergral_targety,-0.21));
  std::cout << "PASS same-frame changes preserve integral\n";
  mode(-1);
  assert(intergral_targetx==0 && intergral_targety==0 && intergral_targetz==0);
  assert(intergral_targetx_body==0 && intergral_targety_body==0);
  std::cout << "PASS existing stop/reset behavior\n";
}
"""


def main():
    source = (Path(__file__).resolve().parents[1] / "src/callback.cpp").read_text()
    with tempfile.TemporaryDirectory(prefix="quaddock_integral_test_") as directory:
        cpp = Path(directory) / "test.cpp"
        binary = Path(directory) / "test"
        cpp.write_text(PREAMBLE + function(source, "mode_callback") + CASES)
        subprocess.run(["g++", "-std=c++14", "-O0", "-g", "-fsanitize=address,undefined",
                        "-fno-omit-frame-pointer", "-I/usr/include/eigen3", str(cpp),
                        "-o", str(binary)], check=True)
        env = dict(os.environ)
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":detect_leaks=0"
        subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    main()
