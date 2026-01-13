#include <iostream>
#include <so3_control/SO3Control.h>

#include <ros/ros.h>

SO3Control::SO3Control()
  : mass_(0.5)
  , g_(9.81)
{
  acc_.setZero();
}

void
SO3Control::setMass(const double mass)
{
  mass_ = mass;
}

void
SO3Control::setGravity(const double g)
{
  g_ = g;
}

void
SO3Control::setPosition(const Eigen::Vector3d& position)
{
  pos_ = position;
}

void
SO3Control::setVelocity(const Eigen::Vector3d& velocity)
{
  vel_ = velocity;
}

/**
 * @brief SE(3)控制器：从期望状态计算期望力和姿态
 * 
 * 这是SE(3)控制的核心函数，采用几何方法直接从期望力方向构建旋转矩阵，
 * 避免了欧拉角分解的非线性耦合问题，适用于任意角度。
 * 
 * 算法流程：
 * 1. 输入有效性检查（检查NaN值）
 * 2. 计算总误差和自适应加速度增益
 * 3. 计算期望力（位置、速度、加速度反馈）
 * 4. 限制控制角度（防止力向下）
 * 5. 从力的方向构建旋转矩阵（SE(3)控制的核心）
 * 6. 转换为四元数输出
 * 
 * @param des_pos 期望位置 [x, y, z] (m)
 * @param des_vel 期望速度 [vx, vy, vz] (m/s)
 * @param des_acc 期望加速度 [ax, ay, az] (m/s²)
 * @param des_yaw 期望偏航角 (rad)
 * @param des_yaw_dot 期望偏航角速度 (rad/s, 当前未使用)
 * @param kx 位置反馈增益 [kx_x, kx_y, kx_z] (N/m)
 * @param kv 速度反馈增益 [kv_x, kv_y, kv_z] (N·s/m)
 */
void
SO3Control::calculateControl(const Eigen::Vector3d& des_pos,
                             const Eigen::Vector3d& des_vel,
                             const Eigen::Vector3d& des_acc,
                             const double des_yaw, const double des_yaw_dot,
                             const Eigen::Vector3d& kx,
                             const Eigen::Vector3d& kv)
{
  //  ROS_INFO("Error %lf %lf %lf", (des_pos - pos_).norm(),
  //           (des_vel - vel_).norm(), (des_acc - acc_).norm());

  // ==================== 步骤1：输入有效性检查 ====================
  // 检查输入是否包含NaN值，如果包含则忽略该输入
  // 这样可以处理部分输入缺失的情况（例如只有位置指令，没有速度指令）
  // 如果某个输入为NaN，对应的flag会被设置为false，后续计算中该输入不会被使用
  bool flag_use_pos = !(std::isnan(des_pos(0)) || std::isnan(des_pos(1)) || std::isnan(des_pos(2)));
  bool flag_use_vel = !(std::isnan(des_vel(0)) || std::isnan(des_vel(1)) || std::isnan(des_vel(2)));
  bool flag_use_acc = !(std::isnan(des_acc(0)) || std::isnan(des_acc(1)) || std::isnan(des_acc(2)));

  // ==================== 步骤2：计算总误差和自适应增益 ====================
  // 计算位置、速度、加速度的总误差（用于自适应加速度增益计算）
  // 总误差用于判断系统状态，决定是否需要自适应调整增益
  Eigen::Vector3d totalError(Eigen::Vector3d::Zero());
  if ( flag_use_pos ) totalError.noalias() += des_pos - pos_;
  if ( flag_use_vel ) totalError.noalias() += des_vel - vel_;
  if ( flag_use_acc ) totalError.noalias() += des_acc - acc_;

  // 计算自适应加速度反馈增益 ka
  // 当误差大于3时，增益为0（避免过大反馈导致系统不稳定）
  // 当误差小于3时，增益 = 0.2 * |误差|（自适应调整，误差越大增益越大）
  // 这种设计可以在误差大时减少反馈，避免系统不稳定；误差小时增强反馈，提高精度
  Eigen::Vector3d ka(fabs(totalError[0]) > 3 ? 0 : (fabs(totalError[0]) * 0.2),
                     fabs(totalError[1]) > 3 ? 0 : (fabs(totalError[1]) * 0.2),
                     fabs(totalError[2]) > 3 ? 0 : (fabs(totalError[2]) * 0.2));

  // std::cout << des_pos.transpose() << std::endl;
  // std::cout << des_vel.transpose() << std::endl;
  // std::cout << des_acc.transpose() << std::endl;
  // std::cout << des_yaw << std::endl;
  // std::cout << pos_.transpose() << std::endl;
  // std::cout << vel_.transpose() << std::endl;
  // std::cout << acc_.transpose() << std::endl;
  

  // ==================== 步骤3：计算期望力 ====================
  // 期望力 = 重力 + 位置反馈 + 速度反馈 + 加速度反馈 + 期望加速度
  // 
  // 物理意义：
  // - 重力项：mg[0,0,1] 用于抵消重力，保持悬停
  // - 位置反馈：kx * (des_pos - pos_) 提供位置误差的修正力（比例控制）
  // - 速度反馈：kv * (des_vel - vel_) 提供速度误差的修正力（阻尼，防止振荡）
  // - 加速度反馈：mass * ka * (des_acc - acc_) 提供加速度误差的修正力（自适应）
  // - 期望加速度：mass * des_acc 提供前馈加速度（开环控制，提高响应速度）
  //
  // 注意：虽然最后只使用力的方向（归一化），但kx和kv仍然重要，
  // 因为它们影响force_的方向，进而影响姿态。不同方向的增益会影响力的方向。
  force_ = mass_ * g_ * Eigen::Vector3d(0, 0, 1);  // 重力项：mg[0,0,1]
  if ( flag_use_pos ) force_.noalias() += kx.asDiagonal() * (des_pos - pos_);  // 位置反馈
  if ( flag_use_vel ) force_.noalias() += kv.asDiagonal() * (des_vel - vel_);   // 速度反馈
  if ( flag_use_acc ) force_.noalias() += mass_ * ka.asDiagonal() * (des_acc - acc_) + mass_ * (des_acc);  // 加速度反馈 + 前馈

  // ==================== 步骤4：限制控制角度（防止力向下）====================
  // Limit control angle to 45 degree
  // 为了保证飞机的可操作性，限制推力方向不能向下（z轴方向不能为负）
  // 
  // 原理：
  // - 计算力方向与垂直方向的夹角：cos(θ) = [0,0,1] · (force_ / ||force_||)
  // - 如果 cos(θ) < 0，说明 θ > 90°，力方向向下（z分量 < 0）
  // - 四旋翼的推力只能向上，不能向下，所以必须限制
  // - 通过缩放水平力分量f，使力的z分量 ≥ 0
  //
  // 注意：代码中 theta = M_PI/2 (90度)，c = cos(90°) = 0
  // 这意味着当 force_(2) / ||force_|| < 0 时，即 force_(2) < 0（力向下），需要限制
  // 所以这个限制实际上是在防止推力向下，而不是限制到45度
  double          theta = M_PI / 2;  // 90度：限制力方向不能超过水平（不能向下）
  double          c     = cos(theta);  // cos(90°) = 0：当力方向水平时，与垂直方向夹角为90°
  Eigen::Vector3d f;
  f.noalias() = force_ - mass_ * g_ * Eigen::Vector3d(0, 0, 1);  // 水平力分量（去除重力）
  
  // 检查力方向与垂直方向的夹角
  // 方程：e_z · (force_ / ||force_||) < c
  // 其中 e_z = [0,0,1] 是垂直方向单位向量
  // 当 c = 0 时，条件等价于：force_(2) / ||force_|| < 0，即 force_(2) < 0（力向下）
  // 这是合理的物理限制，因为四旋翼的推力只能向上，不能向下
  if (Eigen::Vector3d(0, 0, 1).dot(force_ / force_.norm()) < c)
  {
    // 通过求解二次方程计算缩放因子s，使得缩放后的力满足角度限制
    // 
    // 设缩放后的力：f_limited = s * f + mg[0,0,1]
    // 约束条件：[0,0,1] · (f_limited / ||f_limited||) = c = 0（刚好水平）
    // 
    // 推导得到二次方程：A * s² + B * s + C = 0
    // 其中：
    // A = c² * ||f||² - f(2)² = -f(2)²（因为c=0）
    // B = 2 * (c² - 1) * f(2) * mg = -2 * f(2) * mg
    // C = (c² - 1) * (mg)² = -(mg)²
    // 
    // 求解：s = (-B + √(B² - 4AC)) / (2A)
    double nf        = f.norm();  // 水平力的大小：||f||
    double A         = c * c * nf * nf - f(2) * f(2);  // A = c² * ||f||² - f(2)²
    double B         = 2 * (c * c - 1) * f(2) * mass_ * g_;  // B = 2 * (c² - 1) * f(2) * mg
    double C         = (c * c - 1) * mass_ * mass_ * g_ * g_;  // C = (c² - 1) * (mg)²
    double s         = (-B + sqrt(B * B - 4 * A * C)) / (2 * A);  // 二次方程求根，取正根
    force_.noalias() = s * f + mass_ * g_ * Eigen::Vector3d(0, 0, 1);  // 缩放后的力：f_limited = s * f + mg[0,0,1]
  }
  // Limit control angle to 45 degree

  // ==================== 步骤5：从力的方向构建旋转矩阵（SE(3)控制核心）====================
  // 这是SE(3)控制的关键：直接从力的方向构建完整的旋转矩阵，无需分解成欧拉角
  //
  // 原理：
  // 旋转矩阵R的列向量就是机体坐标轴在世界坐标系中的方向：
  // R = [b1c  b2c  b3c] = [机体x轴  机体y轴  机体z轴]
  //
  // 构建步骤：
  // 1. b3c = force_.normalized()  // 机体z轴 = 推力方向（力的方向）
  // 2. b1d = [cos(des_yaw), sin(des_yaw), 0]  // 期望的机体x轴方向（水平面内，由yaw决定）
  // 3. b2c = (b3c × b1d).normalized()  // 机体y轴 = z轴 × x轴（保证正交）
  // 4. b1c = (b2c × b3c).normalized()  // 机体x轴 = y轴 × z轴（保证正交）
  //
  // 优势：
  // - 无需小角度假设，适用于任意角度
  // - 无奇异性问题（万向锁）
  // - 几何直观，计算高效
  
  Eigen::Vector3d b1c, b2c, b3c;  // 机体坐标轴在世界坐标系中的方向向量
  
  // 期望的机体x轴方向（在水平面内，由yaw角决定）
  Eigen::Vector3d b1d(cos(des_yaw), sin(des_yaw), 0);

  // 机体z轴 = 归一化的力方向（推力方向）
  // 这是SE(3)控制的核心：直接将期望力方向作为机体z轴
  // 如果力太小（接近0），默认垂直向上，避免数值问题
  if (force_.norm() > 1e-6)
    b3c.noalias() = force_.normalized();  // 归一化得到单位方向向量
  else
    b3c.noalias() = Eigen::Vector3d(0, 0, 1);  // 如果力太小，默认垂直向上

  // 通过叉乘构建正交坐标系（保证三个轴相互垂直）
  // 机体y轴 = z轴 × 期望x轴（保证y轴垂直于z轴和期望x轴）
  b2c.noalias() = b3c.cross(b1d).normalized();
  
  // 机体x轴 = y轴 × z轴（保证x轴垂直于y轴和z轴，形成右手坐标系）
  b1c.noalias() = b2c.cross(b3c).normalized();

  // ==================== 步骤6：构建旋转矩阵并转换为四元数 ====================
  // 将三个坐标轴方向向量组合成旋转矩阵
  // R的每一列对应一个坐标轴方向：[机体x轴  机体y轴  机体z轴]
  Eigen::Matrix3d R;
  R << b1c, b2c, b3c;

  // 将旋转矩阵转换为四元数（用于输出）
  // 四元数表示旋转更紧凑，且无奇异性，是ROS中常用的姿态表示方式
  orientation_ = Eigen::Quaterniond(R);
}

const Eigen::Vector3d&
SO3Control::getComputedForce(void)
{
  return force_;
}

const Eigen::Quaterniond&
SO3Control::getComputedOrientation(void)
{
  return orientation_;
}

void
SO3Control::setAcc(const Eigen::Vector3d& acc)
{
  acc_ = acc;
}
