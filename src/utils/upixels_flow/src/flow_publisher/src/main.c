#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

// 用户定义的协议头文件
#include "flow_decode.h"

// 定义协议类型
// typedef enum {
//     MAVLINK_PX4_NO_TOF,
//     MAVLINK_PX4,
//     MAVLINK_APM,
//     MSP_NO_TOF,
//     MSP,
//     UPIXELS_NO_TOF,
//     UPIXELS
// } PROTOCOL;

// 串口设备名称和波特率
#define SERIAL_PORT "/dev/ttyUSB0"
#define BAUD_RATE B115200

// 串口文件描述符
int serial_fd;

// 串口初始化函数
int init_serial(const char *port, int baud_rate) {
    struct termios tty;

    // 打开串口设备
    serial_fd = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (serial_fd < 0) {
        perror("Failed to open serial port");
        return -1;
    }

    // 获取当前串口设置
    if (tcgetattr(serial_fd, &tty) != 0) {
        perror("Failed to get serial attributes");
        return -1;
    }

    // 配置串口参数
    cfsetospeed(&tty, baud_rate);
    cfsetispeed(&tty, baud_rate);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;  // 设置数据位8位
    tty.c_iflag &= ~IGNBRK;                      // 禁止忽略BREAK信号
    tty.c_lflag = 0;                             // 关闭本地模式
    tty.c_oflag = 0;                             // 关闭输出处理
    tty.c_cc[VMIN] = 1;                          // 最小读取1字节
    tty.c_cc[VTIME] = 1;                         // 读取超时时间（0.1秒）

    // 设置控制模式
    tty.c_cflag |= (CLOCAL | CREAD);  // 开启接收功能
    tty.c_cflag &= ~(PARENB | PARODD);  // 无校验
    tty.c_cflag &= ~CSTOPB;             // 1位停止位
    tty.c_cflag &= ~CRTSCTS;            // 无硬件流控

    // 应用配置
    if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
        perror("Failed to set serial attributes");
        return -1;
    }

    return 0;
}

// 串口读取函数
int read_serial_data(unsigned char *buffer, size_t len) {
    return read(serial_fd, buffer, len);
}

// 错误处理函数
void error_handler(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

// 主函数
int main(void) {
    PROTOCOL protocol = UPIXELS;  // 选择协议
    unsigned char ch;
    int ret;

    // 初始化串口
    if (init_serial(SERIAL_PORT, BAUD_RATE) < 0) {
        error_handler("Serial port initialization failed");
    }

    printf("Serial port initialized, listening for data...\n");

    // 无限循环处理数据
    while (1) {
        // 读取串口数据
        if (read_serial_data(&ch, 1) > 0) {  // 每次读取1字节
            switch (protocol) {
                case MAVLINK_PX4_NO_TOF:
                    ret = px4notof_parse_char(ch);
                    if (!ret) {
                        float integrated_x = px4_flow_data.integrated_x;
                        float integrated_y = px4_flow_data.integrated_y;
                        uint8_t quality = px4_flow_data.quality;
                        printf("integrated_x=%f, integrated_y=%f, quality=%d\n", integrated_x, integrated_y, quality);
                    }
                    break;
                case MAVLINK_PX4:
                    ret = px4_parse_char(ch);
                    if (!ret) {
                        float integrated_x = px4_flow_data.integrated_x;
                        float integrated_y = px4_flow_data.integrated_y;
                        float distance = px4_flow_data.distance;
                        uint8_t quality = px4_flow_data.quality;
                        printf("integrated_x=%f, integrated_y=%f, distance=%f, quality=%d\n", integrated_x, integrated_y, distance, quality);
                    }
                    break;
                case MAVLINK_APM:
                    ret = apm_parse_char(ch);
                    if (!ret) {
                        float flow_comp_x = apm_flow_data.flow_comp_x;
                        float flow_comp_y = apm_flow_data.flow_comp_y;
                        float ground_distance = apm_flow_data.ground_distance;
                        uint8_t quality = apm_flow_data.quality;
                        printf("flow_comp_x=%f, flow_comp_y=%f, ground_distance=%f, quality=%d\n", flow_comp_x, flow_comp_y, ground_distance, quality);
                    }
                    break;
                case MSP_NO_TOF:
                    ret = mspnotof_parse_char(ch);
                    if (!ret) {
                        uint8_t flow_quality = msp_flow_data.flow_quality;
                        int32_t motionX = msp_flow_data.motionX;
                        int32_t motionY = msp_flow_data.motionY;
                        printf("flow_quality=%d, motionX=%d, motionY=%d\n", flow_quality, motionX, motionY);
                    }
                    break;
                case MSP:
                    ret = msp_parse_char(ch);
                    if (!ret) {
                        uint8_t flow_quality = msp_flow_data.flow_quality;
                        int32_t motionX = msp_flow_data.motionX;
                        int32_t motionY = msp_flow_data.motionY;
                        uint32_t distance = msp_dis_data.distance;
                        printf("flow_quality=%d, motionX=%d, motionY=%d, distance=%d\n", flow_quality, motionX, motionY, distance);
                    }
                    break;
                case UPIXELS_NO_TOF:
                    ret = upnotof_parse_char(ch);
                    if (!ret) {
                        int16_t flow_x_integral = up_flow_data.flow_x_integral;
                        int16_t flow_y_integral = up_flow_data.flow_y_integral;
                        uint8_t valid = up_flow_data.valid;
                        printf("flow_x_integral=%d, flow_y_integral=%d, valid=%d\n", flow_x_integral, flow_y_integral, valid);
                    }
                    break;
                case UPIXELS:
                    ret = up_parse_char(ch);
                    if (!ret) {
                        int16_t flow_x_integral = up_data.flow_x_integral;
                        int16_t flow_y_integral = up_data.flow_y_integral;
                        uint16_t ground_distance = up_data.ground_distance;
                        uint8_t valid = up_data.valid;
                        uint8_t tof_confidence = up_data.tof_confidence;
                        printf("flow_x_integral=%d, flow_y_integral=%d, ground_distance=%d, valid=%d, tof_confidence=%d\n", flow_x_integral, flow_y_integral, ground_distance, valid, tof_confidence);
                    }
                    break;
                default:
                    printf("Unknown protocol\n");
                    break;
            }
        }
    }

    // 关闭串口
    close(serial_fd);
    return 0;
}