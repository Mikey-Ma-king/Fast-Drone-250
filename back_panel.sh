#!/bin/bash
# 飞机端背板蓝牙一键控制脚本（本地运行，蓝牙直连狗端舵机）
# 功能: 环境检查 → 蓝牙守护进程 → 检查rfcomm → 抬起背板 → 等待离地 → 放下背板
# 用法: ./back_panel.sh [等待秒数]
# 默认等待: 5 秒（飞机离地时间）

set -e

BLUETOOTH_MAC="39:93:17:13:48:B8"
RFCOMM_DEVICE="/dev/rfcomm1"
WAIT="${1:-5}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLUETEETH_PY="$SCRIPT_DIR/blueteeth.py"

echo "=========================================="
echo "  飞机端背板蓝牙一键控制"
echo "  狗端舵机蓝牙: $BLUETOOTH_MAC"
echo "  rfcomm:  $RFCOMM_DEVICE"
echo "  等待:    ${WAIT}s"
echo "=========================================="

# ── 1. 环境检查 ────────────────────────────────────────────
echo "[1/5] 环境检查..."

if [ ! -f "$BLUETEETH_PY" ]; then
    echo "  错误: 找不到 $BLUETEETH_PY"
    exit 1
fi
echo "  blueteeth.py 存在 ✓"

if ! bluetoothctl info "$BLUETOOTH_MAC" &>/dev/null; then
    echo "  错误: 蓝牙设备 $BLUETOOTH_MAC 未配对或不可达"
    exit 1
fi
echo "  蓝牙已配对 ✓"

if ! sudo -n true 2>/dev/null; then
    echo "  错误: sudo 需要密码，请先配置免密"
    exit 1
fi
echo "  sudo 免密 ✓"

# ── 2. 蓝牙守护进程 ────────────────────────────────────────
echo "[2/5] 检查蓝牙守护进程..."

if pgrep -f "blueteeth.py" > /dev/null; then
    echo "  blueteeth.py 已在运行 (PID: $(pgrep -f blueteeth.py))"
else
    echo "  启动 blueteeth.py..."
    python3 "$BLUETEETH_PY" &
    sleep 2
    echo "  blueteeth.py 已启动 (PID: $!)"
fi

# ── 3. 等待 rfcomm 设备就绪 ────────────────────────────────
echo "[3/5] 等待蓝牙 rfcomm 设备..."

MAX_WAIT=15
WAITED=0
while [ ! -e "$RFCOMM_DEVICE" ]; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "  错误: ${MAX_WAIT}s 超时, $RFCOMM_DEVICE 未出现"
        echo "  排查: (1) 狗端背板蓝牙模块上电? (2) sudo rfcomm -a"
        echo "        (3) bluetoothctl info $BLUETOOTH_MAC"
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
    echo "  ... ${WAITED}s / ${MAX_WAIT}s"
done

sudo chmod 777 "$RFCOMM_DEVICE" 2>/dev/null || true
echo "  $RFCOMM_DEVICE 就绪 ✓"

# ── 4. 抬起背板 ────────────────────────────────────────────
echo "[4/5] 抬起背板..."
printf '\x00\x5b\x00\x00\x70\x00' > "$RFCOMM_DEVICE"
echo "  已发送抬起指令 ✓"

# ── 5. 等待 & 放下背板 ─────────────────────────────────────
echo "[5/5] 等待 ${WAIT}s（飞机离地）..."
sleep "$WAIT"

echo "  放下背板..."
printf '\x00\x5c\x00\x00\x70\x00' > "$RFCOMM_DEVICE"
echo "  已发送放下指令 ✓"

echo "=========================================="
echo "  背板控制流程完成"
echo "=========================================="
