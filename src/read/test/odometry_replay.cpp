// stdin: receipt,source_stamp,x,y,z,qx,qy,qz,qw; stdout: causal snapshot CSV.
// No ROS calls, publishers, sensors, or actuators.
#include <svo_odometry.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main() {
  image::SvoOdometry odom;
  std::cout << "receipt,source_stamp,valid,x,y,z,vx,vy,vz,reason,input_result\n" << std::setprecision(17);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream row(line);
    double receipt, stamp, x, y, z, qx, qy, qz, qw;
    if (!(row >> receipt >> stamp >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
    const char* input_result = odom.update(stamp, Eigen::Vector3d(x,y,z), Eigen::Quaterniond(qw,qx,qy,qz), receipt);
    const auto s = odom.sample(receipt);
    std::cout << receipt << ',' << stamp << ',' << s.valid << ','
              << s.position.x() << ',' << s.position.y() << ',' << s.position.z() << ','
              << s.velocity.x() << ',' << s.velocity.y() << ',' << s.velocity.z() << ',' << s.reason << ',' << input_result << '\n';
  }
}
