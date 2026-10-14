#!/bin/bash

# 脚本名：monitor_tools_update_teleop.sh
# 功能：升级 moz_teleop 服务

exec > >(tee -a ~/movax/tools/log/update_moz_teleop.log) 2>&1

cd ~
TELEOP_DIR="moz_teleop"
ZIP_FILE="moz_teleop.zip"
TMP_DIR="teleop_tmp_$$"  # 用进程号防止重复
UPLOADTMP_DIR="moz_teleop_upload_temp.zip"

# 日志函数
log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1"
}

# 检查zip文件是否存在
if [[ ! -f "$ZIP_FILE" ]]; then
    log_error "找不到 $ZIP_FILE"
    exit 1
fi

log_info "开始 Teleop 升级流程"
log_info "升级包路径: ~/$ZIP_FILE"

# 解压到临时目录
log_info "解压升级包..."
unzip -oq "$ZIP_FILE" -d "$TMP_DIR"
if [[ $? -ne 0 ]]; then
    log_error "解压失败"
    rm -rf "$TMP_DIR"
    exit 1
fi

# 判断解压后是moz_teleop/还是直接内容
SRC_DIR="$TMP_DIR/moz_teleop"
if [ ! -d "$SRC_DIR" ]; then
    # 有些zip内部可能是直接内容，没有moz_teleop这一层
    SRC_DIR="$TMP_DIR"
fi

# 停止 moz_teleop 进程
log_info "停止 Teleop 服务..."
pid=$(pidof moz_teleop)
if [[ -n "$pid" ]]; then
    kill -9 $pid
    log_info "已杀死 moz_teleop 进程: $pid"
    sleep 1
else
    log_info "未找到运行中的 moz_teleop 进程"
fi

# 备份当前版本（可选）
BACKUP_DIR="$HOME/backup/moz_teleop_$(date +%Y%m%d_%H%M%S)"
if [ -d "$TELEOP_DIR" ]; then
    log_info "备份当前版本到: $BACKUP_DIR"
    mkdir -p "$HOME/backup"
    cp -r "$TELEOP_DIR" "$BACKUP_DIR"
fi

# 创建目标目录（如果不存在）
if [ ! -d "$TELEOP_DIR" ]; then
    mkdir -p "$TELEOP_DIR"
fi

# 用rsync递归覆盖
log_info "替换文件..."
rsync -a "$SRC_DIR/" "$TELEOP_DIR/"
if [[ $? -ne 0 ]]; then
    log_error "文件替换失败"
    rm -rf "$TMP_DIR"
    exit 1
fi

# 清理临时目录
rm -rf "$TMP_DIR"
log_info "清理临时文件完成"

# 设置执行权限
cd "$TELEOP_DIR/bin" 2>/dev/null || cd "$TELEOP_DIR"
if [ -f "moz_teleop" ]; then
    chmod +x moz_teleop
    log_info "设置 moz_teleop 执行权限"
fi

# 如果存在启动脚本，也设置权限
if [ -f "monitor_startup.sh" ]; then
    chmod +x monitor_startup.sh
fi

# 设置所有.sh脚本的执行权限
for script in *.sh; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        log_info "设置 $script 执行权限"
    fi
done

# 设置vr-tools目录下adb的执行权限
cd ~/"$TELEOP_DIR"
if [ -f "vr-tools/adb" ]; then
    chmod +x vr-tools/adb
    log_info "设置 vr-tools/adb 执行权限"
fi

# 设置vr-tools目录下其他可能的可执行文件权限
if [ -d "vr-tools" ]; then
    for tool in vr-tools/*; do
        if [ -f "$tool" ] && [[ ! "$tool" == *.txt ]] && [[ ! "$tool" == *.md ]] && [[ ! "$tool" == *.json ]]; then
            chmod +x "$tool"
            log_info "设置 $tool 执行权限"
        fi
    done
fi

log_info "开始升级 monitor_tool..."

# 切换到 movax 目录执行 monitor 升级
cd ~/movax/env
if [ -f "monitor_tool_startup.sh" ]; then
    # 杀掉 monitor_tool 进程
    monitor_pid=$(pidof monitor_tool)
    if [[ -n "$monitor_pid" ]]; then
        kill -9 $monitor_pid
        log_info "已杀死 monitor_tool 进程: $monitor_pid"
    fi
    
    # 执行 monitor 升级
    bash monitor_tool_startup.sh
    if [[ $? -eq 0 ]]; then
        log_info "Monitor 升级成功"
    else
        log_error "Monitor 升级失败"
    fi
else
    log_error "找不到 monitor 升级脚本"
fi

cd ~

if [[ -f "$UPLOADTMP_DIR" ]]; then
  echo "Removing old temp upload directory: $UPLOADTMP_DIR"
  rm -rf "$UPLOADTMP_DIR" || fail "Failed to remove old upload directory: $UPLOADTMP_DIR"
fi

log_info "moz_teleop 更新完成！"
exit 0

