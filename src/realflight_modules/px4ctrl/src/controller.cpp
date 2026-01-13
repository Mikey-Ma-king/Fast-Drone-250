#include "controller.h"
#include <algorithm>

using namespace std;



double LinearControl::fromQuaternion2yaw(Eigen::Quaterniond q)
{
  double yaw = atan2(2 * (q.x()*q.y() + q.w()*q.z()), q.w()*q.w() + q.x()*q.x() - q.y()*q.y() - q.z()*q.z());
  return yaw;
}

LinearControl::LinearControl(Parameter_t &param) : param_(param)
{
  resetThrustMapping();
}

/* 
  compute u.thrust and u.q, controller gains and other parameters are in param_ 
*/
quadrotor_msgs::Px4ctrlDebug
LinearControl::calculateControl(const Desired_State_t &des,
    const Odom_Data_t &odom,
    const Imu_Data_t &imu, 
    Controller_Output_t &u)
{
  /* WRITE YOUR CODE HERE */
      // 计算期望加速度（两种控制方法都需要）
      Eigen::Vector3d des_acc(0.0, 0.0, 0.0);
      Eigen::Vector3d Kp,Kv;
      Kp << param_.gain.Kp0, param_.gain.Kp1, param_.gain.Kp2;
      Kv << param_.gain.Kv0, param_.gain.Kv1, param_.gain.Kv2;
      des_acc = des.a + Kv.asDiagonal() * (des.v - odom.v) + Kp.asDiagonal() * (des.p - odom.p);
      des_acc += Eigen::Vector3d(0,0,param_.gra);

      // 根据参数选择使用SE(3)控制还是欧拉角控制
      if (param_.use_so3_control)
      {
        // 使用SE(3)控制
        calculateControlSO3(des, odom, imu, u, des_acc);
      }
      else
      {
        // 使用原有的欧拉角控制
        u.thrust = computeDesiredCollectiveThrustSignal(des_acc);
        double roll,pitch,yaw,yaw_imu;
        double yaw_odom = fromQuaternion2yaw(odom.q);
        double sin = std::sin(yaw_odom);
        double cos = std::cos(yaw_odom);
        roll = (des_acc(0) * sin - des_acc(1) * cos )/ param_.gra;
        pitch = (des_acc(0) * cos + des_acc(1) * sin )/ param_.gra;
        
        // Limit roll and pitch angles if max_angle is set (positive value)
        if (param_.max_angle > 0.0)
        {
          roll = std::max(-param_.max_angle, std::min(param_.max_angle, roll));
          pitch = std::max(-param_.max_angle, std::min(param_.max_angle, pitch));
        }
        
        // yaw = fromQuaternion2yaw(des.q);
        yaw_imu = fromQuaternion2yaw(imu.q);
        // Eigen::Quaterniond q = Eigen::AngleAxisd(yaw,Eigen::Vector3d::UnitZ())
        //   * Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX())
        //   * Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY());
        Eigen::Quaterniond q = Eigen::AngleAxisd(des.yaw,Eigen::Vector3d::UnitZ())
          * Eigen::AngleAxisd(pitch,Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(roll,Eigen::Vector3d::UnitX());
        u.q = imu.q * odom.q.inverse() * q;
      }


  /* WRITE YOUR CODE HERE */

  //used for debug
  // debug_msg_.des_p_x = des.p(0);
  // debug_msg_.des_p_y = des.p(1);
  // debug_msg_.des_p_z = des.p(2);
  
  debug_msg_.des_v_x = des.v(0);
  debug_msg_.des_v_y = des.v(1);
  debug_msg_.des_v_z = des.v(2);
  
  debug_msg_.des_a_x = des_acc(0);
  debug_msg_.des_a_y = des_acc(1);
  debug_msg_.des_a_z = des_acc(2);
  
  debug_msg_.des_q_x = u.q.x();
  debug_msg_.des_q_y = u.q.y();
  debug_msg_.des_q_z = u.q.z();
  debug_msg_.des_q_w = u.q.w();
  
  debug_msg_.des_thr = u.thrust;
  
  // Used for thrust-accel mapping estimation
  timed_thrust_.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
  while (timed_thrust_.size() > 100)
  {
    timed_thrust_.pop();
  }
  return debug_msg_;
}

/*
  compute throttle percentage 
*/
double 
LinearControl::computeDesiredCollectiveThrustSignal(
    const Eigen::Vector3d &des_acc)
{
  double throttle_percentage(0.0);
  
  /* compute throttle, thr2acc has been estimated before */
  throttle_percentage = des_acc(2) / thr2acc_;

  return throttle_percentage;
}

bool 
LinearControl::estimateThrustModel(
    const Eigen::Vector3d &est_a,
    const Parameter_t &param)
{
  ros::Time t_now = ros::Time::now();
  while (timed_thrust_.size() >= 1)
  {
    // Choose data before 35~45ms ago
    std::pair<ros::Time, double> t_t = timed_thrust_.front();
    double time_passed = (t_now - t_t.first).toSec();
    if (time_passed > 0.045) // 45ms
    {
      // printf("continue, time_passed=%f\n", time_passed);
      timed_thrust_.pop();
      continue;
    }
    if (time_passed < 0.035) // 35ms
    {
      // printf("skip, time_passed=%f\n", time_passed);
      return false;
    }

    /***********************************************************/
    /* Recursive least squares algorithm with vanishing memory */
    /***********************************************************/
    double thr = t_t.second;
    timed_thrust_.pop();
    
    /***********************************/
    /* Model: est_a(2) = thr1acc_ * thr */
    /***********************************/
    double gamma = 1 / (rho2_ + thr * P_ * thr);
    double K = gamma * P_ * thr;
    thr2acc_ = thr2acc_ + K * (est_a(2) - thr * thr2acc_);
    P_ = (1 - K * thr) * P_ / rho2_;
    //printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr2acc_, gamma, K, P_);
    //fflush(stdout);

    // debug_msg_.thr2acc = thr2acc_;
    return true;
  }
  return false;
}

void 
LinearControl::resetThrustMapping(void)
{
  thr2acc_ = param_.gra / param_.thr_map.hover_percentage;
  P_ = 1e6;
}

// ==================== SE(3)控制实现 ====================
/**
 * @brief SE(3)控制：从期望状态计算期望力和姿态
 * 
 * 采用几何方法直接从期望力方向构建旋转矩阵，避免了欧拉角分解的非线性耦合问题
 * 
 * @param des 期望状态
 * @param odom 里程计数据（当前位置、速度等）
 * @param imu IMU数据（当前姿态等）
 * @param u 控制器输出（姿态和推力）
 * @param des_acc 期望加速度（已在外部计算好）
 */
void
LinearControl::calculateControlSO3(const Desired_State_t &des,
                                    const Odom_Data_t &odom,
                                    const Imu_Data_t &imu,
                                    Controller_Output_t &u,
                                    const Eigen::Vector3d &des_acc)
{

  // 步骤2：计算期望力（世界坐标系）
  // 期望力 = 质量 * 期望加速度
  Eigen::Vector3d force_world = param_.mass * des_acc;

  // 步骤3：限制控制角度（防止力向下）
  // 检查力方向与垂直方向的夹角，如果力向下（z分量<0），需要限制
  double c = cos(M_PI / 4);  // cos(45°) = 0，限制力不能向下
  Eigen::Vector3d f_horizontal = force_world - param_.mass * param_.gra * Eigen::Vector3d(0, 0, 1);
  
  if (Eigen::Vector3d(0, 0, 1).dot(force_world / force_world.norm()) < c)
  {
    // 通过缩放水平力分量，使力的z分量 >= 0
    double nf = f_horizontal.norm();
    double A = c * c * nf * nf - f_horizontal(2) * f_horizontal(2);
    double B = 2 * (c * c - 1) * f_horizontal(2) * param_.mass * param_.gra;
    double C = (c * c - 1) * param_.mass * param_.mass * param_.gra * param_.gra;
    double s = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);
    force_world = s * f_horizontal + param_.mass * param_.gra * Eigen::Vector3d(0, 0, 1);
  }

  // 步骤4：从力的方向构建旋转矩阵（SE(3)控制核心）
  // 机体z轴 = 归一化的力方向（推力方向）
  Eigen::Vector3d b1c, b2c, b3c;
  if (force_world.norm() > 1e-6)
    b3c = force_world.normalized();
  else
    b3c = Eigen::Vector3d(0, 0, 1);

  // 期望的机体x轴方向（在水平面内，由yaw角决定）
  Eigen::Vector3d b1d(cos(des.yaw), sin(des.yaw), 0);

  // 通过叉乘构建正交坐标系
  b2c = b3c.cross(b1d).normalized();  // 机体y轴
  b1c = b2c.cross(b3c).normalized();  // 机体x轴

  // 构建旋转矩阵
  Eigen::Matrix3d R;
  R << b1c, b2c, b3c;

  // 转换为四元数（相对于世界坐标系）
  Eigen::Quaterniond q_world(R);
  
  // 转换为相对于IMU的姿态（与原有方法保持一致）
  u.q = imu.q * odom.q.inverse() * q_world;

  // 步骤5：计算推力（沿机体z轴方向的加速度）
  // 将期望加速度投影到机体z轴方向
  // 注意：R的第三列就是机体z轴在世界坐标系中的方向（b3c）
  Eigen::Vector3d body_z_axis = R.col(2);  // 机体z轴在世界坐标系中的方向 = b3c
  double des_acc_body_z = des_acc.dot(body_z_axis);  // 期望加速度在机体z轴上的投影
  
  // 使用投影后的加速度计算推力
  u.thrust = computeDesiredCollectiveThrustSignal(Eigen::Vector3d(0, 0, des_acc_body_z));
}







