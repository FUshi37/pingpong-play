#!/bin/bash  

# 等待 10 秒  
if [ $# -eq 0 ]; then
    # 没有参数，等待 10 秒
    sleep 10
fi
 
kill -9  $(pidof monitor_tool) || true
sleep 3

# 打印启动日志  
echo "[$(date +"%Y-%m-%d %H:%M:%S")] monitor_tool starting..."  

cd /home/spirit-ai/movax/env || {  
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] 错误: 无法进入目标目录！"  
  exit 1  
}  

python3 remove_angle_gap.py --dir  /home/spirit-ai/movax

# 切换到目标目录  
cd /home/spirit-ai/movax/tools || {  
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] 错误: 无法进入目标目录！"  
  exit 1  
}  

# 确保可执行权限  
chmod +x monitor_tool  

# 启动 monitor_tool  
echo "[$(date +"%Y-%m-%d %H:%M:%S")] monitor_tool starting..."  
./monitor_tool &  

# 打印启动完成日志  
echo "[$(date +"%Y-%m-%d %H:%M:%S")] monitor_tool started..."  

# 正常退出  
exit 0
