#!/bin/bash

# Script name: monitor_tools_recover_snapshot.sh
# Purpose: Upgrade the mc_core controller and update Movax

# === Enable log file ===
mkdir -p ~/movax/tools/log
exec > >(tee -a ~/movax/tools/log/update_movax.log) 2>&1

# === Error handling function ===
fail() {
  echo "Error: $1" >&2
  exit 1
}

echo "Starting Movax update..."

# === Move to home directory ===
cd ~ || fail "Unable to access user home directory (~)"

MOVAX_DIR="movax"
ZIP_FILE="movax.zip"
TMP_DIR="movax_tmp_$$"
UPLOADTMP_DIR="$HOME/movax_upload_temp.zip"

# === Remove old backup directories (if any remain from previous script versions) ===
BACKUP_PATTERN="$HOME/movax_bak_*"

for dir in $BACKUP_PATTERN; do
  if [[ -d "$dir" ]]; then
    echo "Removing old backup directory: $dir"
    rm -rf "$dir" || fail "Failed to remove old backup directory: $dir"
  fi
done

# === Create new backup ===
if [[ -d "$MOVAX_DIR" ]]; then  
  BACKUP_DIR="$HOME/movax_bak_$(date '+%Y%m%d_%H%M%S')"  
  echo "Creating backup directory: $BACKUP_DIR"  
  rsync -av --exclude 'controller/datatracer' --exclude 'controller/log' --exclude 'controller/userlogdb' --exclude 'tools' "$MOVAX_DIR" "$BACKUP_DIR" || fail "Failed to copy directory to backup: $BACKUP_DIR"  
  echo "Backup created: $BACKUP_DIR"  
else  
  echo "Movax directory not found, skipping backup."  
fi  

# === Check update package ===
if [[ ! -f "$ZIP_FILE" ]]; then
  fail "Update package not found: $ZIP_FILE"
fi

# === Extract package ===
unzip -oq "$ZIP_FILE" -d "$TMP_DIR" || fail "Failed to extract update package: $ZIP_FILE"

# === Verify extracted directory ===
SRC_DIR="$TMP_DIR/movax"
if [[ ! -d "$SRC_DIR" ]]; then
  SRC_DIR="$TMP_DIR"
fi
[[ -d "$SRC_DIR" ]] || fail "Movax directory not found in extracted content."

# === Synchronize updated files ===
rsync -a "$SRC_DIR/" "$MOVAX_DIR/" || fail "File synchronization failed: $SRC_DIR to $MOVAX_DIR"

# === Clean up temporary directories ===
rm -rf "$TMP_DIR" || fail "Failed to remove temporary directory: $TMP_DIR"

# === Set executable permissions ===
cd ~/movax/tools || fail "Cannot access directory: ~/movax/tools"
chmod +x monitor_tool || fail "Failed to set execute permissions: monitor_tool"

cd ~/movax/controller || fail "Cannot access directory: ~/movax/controller"
chmod +x mc_core || fail "Failed to set execute permissions: mc_core"

cd ~/movax/env || fail "Cannot access directory: ~/movax/env"
chmod +x * || fail "Failed to set execute permissions for files in ~/movax/env"


if [[ -x ./monitor_tool_startup.sh ]]; then
  ./monitor_tool_startup.sh 1 || fail "Failed to start monitor_tool_startup.sh"
else
  fail "Startup script monitor_tool_startup.sh not found or not executable."
fi

# === Remove old upload temp directory ===
if [[ -f "$UPLOADTMP_DIR" ]]; then
  echo "Removing old upload temp directory: $UPLOADTMP_DIR"
  rm -rf "$UPLOADTMP_DIR" || fail "Failed to remove old upload directory: $UPLOADTMP_DIR"
fi

echo "Movax update completed successfully."
exit 0
