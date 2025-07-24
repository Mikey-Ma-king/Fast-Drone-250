import math

def get_cmd_yaw(target_pxy, vins_pxy,vins_yaw):
    # 获取无人机和目标点的xy坐标
    x1, y1 = vins_pxy
    x2, y2 = target_pxy
    
    # 计算两点之间的差值
    dx = x2 - x1
    dy = y2 - y1
    
    # 使用atan2计算角度，返回值单位为弧度，转换为角度
    target_yaw = math.atan2(dy, dx)
    # angle_deg = math.degrees(angle_rad)

    yaw  = 0

    if (target_yaw - vins_yaw > 0.2 and target_yaw - vins_yaw<3.14) or target_yaw - vins_yaw <= -3.14:
        yaw = vins_yaw + 0.04
    if (target_yaw - vins_yaw < -0.2 and target_yaw - vins_yaw > -3.14) or target_yaw - vins_yaw >= 3.14:
        yaw = vins_yaw - 0.04
    
    return yaw

# 测试数据
target_pxy = (5, 5)  # 目标点坐标 (x2, y2)
vins_pxy = (1, 1)    # 无人机坐标 (x1, y1)

cmdyaw = get_cmd_yaw(target_pxy,vins_pxy,0.99)
print(cmdyaw)
# print(f"The target_yaw is: {target_yaw:.2f} rad")
