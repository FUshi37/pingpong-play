#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import argparse
import shutil

# === Global list of keys to replace (matches by key name only) ===
TARGET_KEYS = [
    "joint_ext_torque_offset",
    "zero_offset_ch1",
    "zero_offset_ch2",
    "ch1_positive_",
    "ch2_positive_",
    "ch1_negative_",
    "ch2_negative_",
    "linear_coefficient_ch1",
    "linear_coefficient_ch2",
    "calibration_point",
    "low_encoder_circle",
    "low_encoder_single_circle",
    "calibration_point",
]
# =================================================================


def load_yaml(file_path):
    """Load YAML safely."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[Warning] Failed to load {file_path}: {e}")
        return {}


def save_yaml(data, file_path):
    """Write YAML file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def replace_by_keys(src, dst):
    if not isinstance(src, dict):
        return dst

    if not isinstance(dst, dict):
        dst = {}

    for key, src_val in src.items():
        # 只对目标字段操作：若存在则替换，若不存在则补充
        if key in TARGET_KEYS:
            dst[key] = src_val
        # 若是嵌套字典，则递归进入（无论目标是否有此层）
        elif isinstance(src_val, dict):
            if key not in dst or not isinstance(dst[key], dict):
                dst[key] = {}
            dst[key] = replace_by_keys(src_val, dst[key])
        # 其他类型跳过
        else:
            continue

    return dst


def sync_yaml_directory(src_dir, dst_dir):
    """Walk through YAML files and update or copy them."""
    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith((".yml", ".yaml")):
                continue

            src_path = os.path.join(root, file)
            rel_path = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)

            # Load source YAML
            src_data = load_yaml(src_path)

            # If destination YAML does not exist -> copy directly
            if not os.path.exists(dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"[New] Copied new file: {dst_path}")
                continue

            # If exists -> merge updates
            dst_data = load_yaml(dst_path)
            updated = replace_by_keys(src_data, dst_data)
            save_yaml(updated, dst_path)

            print(f"[Updated] Replaced matching keys in: {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace or add YAML files from one directory to another."
    )
    parser.add_argument("--src", required=True, help="Source directory")
    parser.add_argument("--dst", required=True, help="Destination directory")
    args = parser.parse_args()

    print("Source directory:", args.src)
    print("Destination directory:", args.dst)
    print("--------------------------------")

    sync_yaml_directory(args.src, args.dst)


if __name__ == "__main__":
    main()