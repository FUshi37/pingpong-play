#!/bin/bash -e
#
# movax_startup.sh
#
# This script is used to start MovaX motion control system
#
exec > >(tee -a ~/movax/tools/log/start_movax.log) 2>&1

if [[ $# -gt 0 ]]; then  
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Sleeping for $1 seconds..."  
    sleep "$1"  
else  
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] No sleep parameter provided, continuing..."  
fi  

echo "[$(date +"%Y-%m-%d %H:%M:%S")] MovaX starting..."

source /opt/ros/foxy/setup.bash

if [[ $# -gt 1 ]]; then  
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] has ros_domain_id param "  
    export ROS_DOMAIN_ID="$2" 
else  
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] no ros_domain_id param "  
    source ~/.bashrc 
fi  

pids=$(pidof mc_core)

if [ -n "$pids" ]; then
    kill $pids
    sleep 3

    if pidof mc_core >/dev/null; then
        kill -9 $(pidof mc_core)
    fi
fi
  
echo ROS_DOMAIN_ID=$ROS_DOMAIN_ID
export LD_LIBRARY_PATH=/home/spirit-ai/movax/lib:$LD_LIBRARY_PATH
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/spirit-ai/movax/env/fastdds_config.xml

cd /home/spirit-ai/movax/controller
if [ -e "mc_core" ]; then
     chmod +x mc_core
     echo "[$(date +"%Y-%m-%d %H:%M:%S")] mc_core starting..."
     nohup ./mc_core --console_enable=false >/tmp/mc_core.log 2>&1 & disown
     echo "[$(date +"%Y-%m-%d %H:%M:%S")] mc_core started..."
else
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Error: mc_core not found or not executable!"
    exit 1
fi

echo "[$(date +"%Y-%m-%d %H:%M:%S")] MovaX started..."
exit 0
