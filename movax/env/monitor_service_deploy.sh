#!/bin/bash  
sudo cp monitor_tool.service /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable monitor_tool.service
