import serial

def main():
    # 初始化串口通信
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)

    print("输入 '1' 发送 '+', 输入 '0' 发送 '-'，输入 'q' 退出")

    while True:
        user_input = input("请输入指令 (1/0/q): ").strip()

        if user_input == '1':
            ser.write(b'+')
            print("发送: +")
        elif user_input == '0':
            ser.write(b'-')
            print("发送: -")
        elif user_input.lower() == 'q':
            print("退出程序")
            break
        else:
            print("无效输入，请输入 1、0 或 q")

    # 关闭串口
    ser.close()

if __name__ == "__main__":
    main()
