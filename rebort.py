from pymavlink import mavutil
import time

# 连接到飞控
connection = mavutil.mavlink_connection('/dev/ttyACM0', baud=921600)

def reboot_px4():
    """ 发送 MAVLink 命令来重启 PX4 飞控 """
    print("等待心跳包...")
    connection.wait_heartbeat()
    print(f"已连接到系统 {connection.target_system}, 组件 {connection.target_component}")

    # 发送重启命令
    connection.mav.command_long_send(
        connection.target_system,  # 目标系统（飞控）
        connection.target_component,  # 目标组件
        mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,  # 命令类型
        0,  # 确保 command_long 发送格式正确
        1,  # 参数1 = 1 表示重启
        0, 0, 0, 0, 0, 0  # 其他参数设置为 0
    )
    
    print("PX4 重启命令已发送！")

if __name__ == "__main__":
    reboot_px4()
