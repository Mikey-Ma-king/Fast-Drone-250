#!/bin/bash

# 设置串口路径
PORT="/dev/rfcomm1"

# 第一次发送数据
echo -e '\x00\x5c\x00\x00\x70\x00' > "$PORT"
