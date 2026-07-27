#!/bin/bash
# 飞机端背板 USB 串口一键控制脚本（通过 hc-14.py 协议）
# 功能: 环境检查 → USB设备检查 → 串口配置 → 发送触发
# 用法: ./back_panel.sh

set -e

USB_HC14_SEND="/dev/USB_hc14_send"
USB_HC14_RECEIVE="/dev/USB_hc14_receive"
BAUD=115200
SUDO_PASS="123456"

echo "=========================================="
echo "  飞机端背板 USB 串口控制"
echo "  主发送设备: $USB_HC14_SEND"
echo "  备用设备:   $USB_HC14_RECEIVE"
echo "  波特率:     $BAUD"
echo "=========================================="

# ── 1. 环境检查 ────────────────────────────────────────────
echo "[1/4] 环境检查..."

if ! command -v stty &>/dev/null; then
    echo "  错误: stty 命令不可用"
    exit 1
fi
echo "  stty 可用 ✓"

if ! echo "$SUDO_PASS" | sudo -S true 2>/dev/null; then
    echo "  错误: sudo 密码不正确"
    exit 1
fi
echo "  sudo 就绪 ✓"

# ── 2. USB 设备检查 ────────────────────────────────────────
echo "[2/4] 检查 USB 串口设备..."

for dev in "$USB_HC14_SEND" "$USB_HC14_RECEIVE"; do
    if [ ! -e "$dev" ]; then
        echo "  错误: $dev 不存在"
        echo "  排查: (1) USB 收发模块是否已插入?"
        echo "        (2) udev 规则是否已配置?"
        echo "        (3) ls /dev/USB_hc14_* 检查设备列表"
        exit 1
    fi
    if [ ! -c "$dev" ]; then
        echo "  错误: $dev 不是字符设备"
        exit 1
    fi
    if [ ! -w "$dev" ]; then
        echo "  $dev 不可写，尝试修复权限..."
        echo "$SUDO_PASS" | sudo -S chmod 777 "$dev" 2>/dev/null || {
            echo "  错误: 无法修改 $dev 权限"
            exit 1
        }
    fi
    echo "  $dev 就绪 ✓"
done

# ── 3. 串口配置 ────────────────────────────────────────────
echo "[3/4] 配置串口参数..."

for dev in "$USB_HC14_SEND" "$USB_HC14_RECEIVE"; do
    if ! stty -F "$dev" $BAUD cs8 -cstopb -parenb 2>/dev/null; then
        echo "  错误: 无法配置 $dev 波特率 $BAUD"
        echo "  排查: (1) 设备是否被其他进程占用? (lsof $dev)"
        echo "        (2) 设备是否支持 ${BAUD} 波特率?"
        exit 1
    fi
    echo "  $dev 配置为 ${BAUD}/8N1 ✓"
done

# ── 4. 发送触发信号 ────────────────────────────────────────
echo "[4/4] 发送背板触发信号..."

send_ok=false
for dev in "$USB_HC14_SEND" "$USB_HC14_RECEIVE"; do
    echo "  尝试通过 $dev 发送..."
    if echo -n "1@" > "$dev" 2>/dev/null; then
        echo "  已通过 $dev 发送"
    else
        echo "  $dev 写入失败，尝试下一个..."
        continue
    fi

    read -p "  观察背板是否正常动作? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        echo "  背板动作确认 ✓，配对设备: $dev"
        send_ok=true
        break
    else
        echo "  $dev 无反应，尝试下一个..."
    fi
done

if ! $send_ok; then
    echo "  错误: 所有设备发送失败"
    echo "  排查: (1) 狗端 hc-14.py 是否已启动?"
    echo "        (2) USB 收发模块是否配对正常?"
    exit 1
fi

echo "  狗端 hc-14.py 收到后执行: PLC 0001 → 3s → PLC 0002"
echo "=========================================="
echo "  背板控制流程完成"
echo "=========================================="
