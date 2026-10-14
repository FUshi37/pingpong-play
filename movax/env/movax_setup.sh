#!/bin/bash -e
#
# movax_setup.sh
#
# This script is used to set MovaX running environment
#

echo "[$(date +"%Y-%m-%d %H:%M:%S")] MovaX running environment setup..."

source /opt/ros/foxy/setup.bash
export LD_LIBRARY_PATH=/home/spirit-ai/movax/lib:$LD_LIBRARY_PATH
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/spirit-ai/movax/env/fastdds_config.xml

echo "[$(date +"%Y-%m-%d %H:%M:%S")] MovaX running environment set success."
