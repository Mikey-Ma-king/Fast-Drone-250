import subprocess
import time

# 目标蓝牙地址
BLUETOOTH_MAC = "39:93:17:13:48:B8"
RFCOMM_DEVICE = "/dev/rfcomm1"

def connect_bluetooth():
    """ 尝试连接蓝牙设备，并在断开时重新连接 """
    while True:
        try:
            print(f"尝试连接蓝牙设备 {BLUETOOTH_MAC}...")
            # 执行 sudo rfcomm connect 命令
            process = subprocess.Popen(["sudo", "rfcomm", "connect", RFCOMM_DEVICE, BLUETOOTH_MAC], 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 读取并输出连接日志
            for line in iter(process.stdout.readline, b''):
                decoded_line = line.decode().strip()
                print(decoded_line)

                # 如果检测到 "Connection refused" 或 "No route to host"，说明连接失败
                if "Connection refused" in decoded_line or "No route to host" in decoded_line:
                    print("连接失败，1秒后重试...")
                    time.sleep(1)
                    break

            # 等待进程结束
            process.wait()

            # 如果进程意外终止，重新尝试连接
            print("连接断开，5秒后重新尝试...")
            time.sleep(5)

        except KeyboardInterrupt:
            print("用户终止程序。")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    connect_bluetooth()
