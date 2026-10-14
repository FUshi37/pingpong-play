#!/bin/bash
# Script: monitor_tools_save_snapshot.sh
# Function: Save Movax snapshot package

# === Error handling function ===

exec > >(tee -a ~/movax/tools/log/save_snapshot.log) 2>&1

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

# === Argument check ===
ZIP_NAME="$1"
if [[ -z "$ZIP_NAME" ]]; then
  fail "Missing parameter: Please specify the ZIP file name, e.g. ./monitor_tools_save_snapshot.sh backup1.zip"
fi

# Ensure the file name ends with .zip
if [[ "$ZIP_NAME" != *.zip ]]; then
  ZIP_NAME="${ZIP_NAME}.zip"
fi

# === Directory definitions ===
MOVAX_DIR="$HOME/movax"
CONF_DIR="$MOVAX_DIR/controller/configuration"
SNAPSHOT_DIR="$HOME/snapshot"
TMP_DIR="$SNAPSHOT_DIR/movax_snapshot"

# === Ensure snapshot directory exists (needed before logging) ===
mkdir -p "$SNAPSHOT_DIR" || fail "Failed to create snapshot directory: $SNAPSHOT_DIR"

# === Check base directories ===
[[ -d "$MOVAX_DIR" ]] || fail "Directory not found: $MOVAX_DIR"
[[ -d "$CONF_DIR" ]] || fail "Configuration directory not found: $CONF_DIR"

# === Prepare temporary directory ===
if [[ -e "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
fi
mkdir -p "$TMP_DIR/controller" || fail "Failed to create temporary directory: $TMP_DIR/controller"

# === Copy configuration files ===
rsync -a --exclude '/basic/system.yaml' "$CONF_DIR/" "$TMP_DIR/controller/configuration" || fail "Failed to copy configuration directory"
rsync -a "$MOVAX_DIR/controller/mc_core" "$TMP_DIR/controller/mc_core" || fail "Failed to copy the executable"
rsync -a "$MOVAX_DIR/env" "$TMP_DIR/" || fail "Failed to copy env directory"
rsync -a "$MOVAX_DIR/lib" "$TMP_DIR/" || fail "Failed to copy lib directory"
rsync -a "$MOVAX_DIR/tools" "$TMP_DIR/" || fail "Failed to copy lib directory"

# === Compress to ZIP ===
cd "$SNAPSHOT_DIR" || fail "Cannot change to directory: $SNAPSHOT_DIR"
if [ -n "$ZIP_NAME" ] && [ -e "$ZIP_NAME" ]; then
  rm -f -- "$ZIP_NAME" || fail "Cannot remove file: $ZIP_NAME"
fi
zip -qr1 "$ZIP_NAME" "movax_snapshot" || fail "ZIP creation failed: $ZIP_NAME"

# === Cleanup ===
rm -rf "$TMP_DIR"

echo "Snapshot ZIP created successfully: $SNAPSHOT_DIR/$ZIP_NAME"
echo "Contents: configuration/ and empty controller/ directory"
echo "Completed at $(date '+%F %T')"
exit 0
