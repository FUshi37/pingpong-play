#!/bin/bash

# 停止 CAN 接口
sudo ip link set can0 down

# 配置 CAN 接口参数
sudo ip link set can0 type can bitrate 1000000 restart-ms 100

# 启动 CAN 接口
sudo ip link set can0 up

sudo ip link set can0 txqueuelen 1000
