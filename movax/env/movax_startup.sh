#!/bin/bash -e
#
# movax_startup.sh
#
# This script is used to start MovaX motion control system
#

sleep 10

echo "[$(date +"%Y-%m-%d %H:%M:%S")] MovaX starting..."

source /opt/ros/foxy/setup.bash
export LD_LIBRARY_PATH=/home/spirit-ai/movax/lib:$LD_LIBRARY_PATH
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/spirit-ai/movax/env/fastdds_config.xml

cd /home/spirit-ai/movax/controller
if [ -e "mc_core" ]; then
     chmod +x mc_core
     echo "[$(date +"%Y-%m-%d %H:%M:%S")] mc_core starting..."
     ./mc_core --console_enable=false &
     echo "[$(date +"%Y-%m-%d %H:%M:%S")] mc_core started..."
else
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] Error: mc_core not found or not executable!"
    exit 1
fi

echo "[$(date +"%Y-%m-%d %H:%M:%S")] MovaX started..."
exit 0
