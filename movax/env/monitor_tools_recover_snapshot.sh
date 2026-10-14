#!/bin/bash
# Script name: monitor_tools_recover_snapshot.sh
# Function: Restore Movax snapshot and restart service
# Log file: ~/movax/tools/recover_snapshot.log

# === Enable logging ===
exec > >(tee -a ~/movax/tools/log/recover_snapshot.log) 2>&1

echo "========== $(date '+%F %T') Start Movax Snapshot Recovery =========="

# === Error handler ===
fail() {
  echo "ERROR: $1" >&2
  exit 1
}

# === Go to home directory ===
cd ~ || fail "Cannot change to user home directory ~"

# === Define key paths ===
MOVAX_DIR="movax"
ZIP_FILE="movax_snapshot.zip"
TMP_DIR="movax_snapshot_tmp_$$"
UPLOADTMP_DIR="$HOME/movax_snapshot_upload_temp.zip"

BACKUP_PATTERN="$HOME/movax_bak_*"

for dir in $BACKUP_PATTERN; do
  if [[ -d "$dir" ]]; then
    echo "Removing old backup directory: $dir"
    rm -rf "$dir" || fail "Failed to remove old backup directory: $dir"
  fi
done

# === Step 2: Backup current Movax directory (no compression) ===
if [[ -d "$MOVAX_DIR" ]]; then
  TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
  BACKUP_DIR="movax_bak_${TIMESTAMP}"
  echo "Creating directory backup: ${BACKUP_DIR}"

  rsync -av --exclude 'controller/datatracer' --exclude 'controller/log' --exclude 'controller/userlogdb' --exclude 'tools' "$MOVAX_DIR/" "$BACKUP_DIR/" || fail "Failed to copy Movax directory"
else
  echo "No existing Movax directory found, skipping backup."
fi

# === Step 3: Check snapshot package ===
if [[ ! -f "$ZIP_FILE" ]]; then
  fail "Snapshot file not found: $ZIP_FILE"
fi

# === Step 4: Extract snapshot ===
mkdir -p "$TMP_DIR" || fail "Failed to create temp directory: $TMP_DIR"
echo "Extracting snapshot..."
unzip -oq "$ZIP_FILE" -d "$TMP_DIR"
unzip_status=$?

if [[ $unzip_status -ne 0 && $unzip_status -ne 1 ]]; then
  fail "Unzip failed with code $unzip_status: $ZIP_FILE"
elif [[ $unzip_status -eq 1 ]]; then
  echo "Warning: unzip returned warning code 1 (probably harmless, possible path separator issue)."
fi

# Fix permissions to avoid Permission Denied during subsequent steps
chmod -R u+rwX "$TMP_DIR" || echo "Warning: failed to fix permissions"

# === Step 5: Determine source directory inside the extracted content ===
SRC_DIR="$TMP_DIR/movax_snapshot"
if [[ ! -d "$SRC_DIR" ]]; then
  SRC_DIR="$TMP_DIR"
fi

cd ~/movax/env || fail "Cannot enter directory: ~/movax/env"
python3 movax_param_replace.py --src $HOME/$MOVAX_DIR/controller/configuration/model/user/ --dst $HOME/$SRC_DIR/controller/configuration/model/user/

cd ~ || fail "Cannot change to user home directory ~"
# === Step 6: Sync files into Movax directory ===
time rsync -a --no-perms --info=progress2 "$SRC_DIR/" "$MOVAX_DIR/" || fail "rsync synchronization failed"

# === Step 7: Clean up temporary directory ===
rm -rf "$TMP_DIR" || fail "Failed to clean temp directory: $TMP_DIR"

# === Step 8: Fix file permissions ===
cd ~/movax/tools || fail "Cannot enter directory: ~/movax/tools"
chmod +x monitor_tool || fail "Failed to set execute permission: monitor_tool"

cd ~/movax/controller || fail "Cannot enter directory: ~/movax/controller"
chmod +x mc_core || fail "Failed to set execute permission: mc_core"

cd ~/movax/env || fail "Cannot enter directory: ~/movax/env"
chmod +x * || fail "Failed to set execute permission: files in env directory"

if [[ $# -gt 0 ]]; then  
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] has ros_domain_id param "  
    export ROS_DOMAIN_ID="$1" 
fi  

if [[ -x ./monitor_tools_start_movax.sh ]]; then
  echo "Starting Movax..."
  ./monitor_tools_start_movax.sh  2  $1|| fail "Failed to start Movax: monitor_tools_start_movax.sh error"
else
  fail "Startup script is missing or not executable: monitor_tools_start_movax.sh"
fi

# === Remove old upload temp directory ===
if [[ -f "$UPLOADTMP_DIR" ]]; then
  echo "Removing old upload temp directory: $UPLOADTMP_DIR"
  rm -rf "$UPLOADTMP_DIR" || fail "Failed to remove old upload directory: $UPLOADTMP_DIR"
fi

cd ~
if [[ -n "$ZIP_FILE" ]]; then
  echo "Removing old upload temp directory: $ZIP_FILE"
  rm -rf "$ZIP_FILE" || fail "Failed to remove old upload directory: $ZIP_FILE"
fi

echo "Movax recovery completed successfully!"
echo "Finished at $(date '+%F %T')"
echo "==============================================================="
exit 0
