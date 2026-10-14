#!/usr/bin/env python3
"""
从指定目录下所有 YAML 文件中删除 angle_gap 配置项。
仅当 angle_gap 不是数组时删除；若为数组则保留。

用法:
    python remove_angle_gap.py --dir <movax根目录>
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("错误: 需要 PyYAML 库。请运行: pip install pyyaml")
    sys.exit(1)


def remove_angle_gap_recursive(obj):
    """递归遍历并删除 angle_gap 键（仅当 angle_gap 不是数组时删除）"""
    if isinstance(obj, dict):
        if "angle_gap" in obj:
            if not isinstance(obj["angle_gap"], list):
                del obj["angle_gap"]
        for value in obj.values():
            remove_angle_gap_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            remove_angle_gap_recursive(item)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="从目录下所有 YAML 文件中删除 angle_gap"
    )
    parser.add_argument(
        "--dir",
        "-d",
        dest="dir_path",
        type=Path,
        required=True,
        help="传入 movax 根目录（脚本会自动拼接 controller/configuration/model/user）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要修改的文件，不写入",
    )
    args = parser.parse_args()

    base_dir = args.dir_path.resolve()
    target = (base_dir / "controller" / "configuration" / "model" / "user").resolve()
    print(f"基准目录(--dir): {base_dir}")
    print(f"目标目录: {target}")

    if not target.exists():
        print(f"错误: 路径不存在: {target}")
        sys.exit(1)
    if not target.is_dir():
        print(f"错误: 路径不是目录: {target}")
        sys.exit(1)

    yaml_files = list(target.rglob("*.yaml")) + list(target.rglob("*.yml"))
    yaml_files = list(dict.fromkeys(yaml_files))  # 去重

    modified = 0
    for f in yaml_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)

        if data is None:
            continue

        original = yaml.dump(data, allow_unicode=True, sort_keys=False)
        remove_angle_gap_recursive(data)
        new_content = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)

        if original != new_content:
            modified += 1
            if not args.dry_run:
                with open(f, "w", encoding="utf-8") as fp:
                    fp.write(new_content)
                print(f"已更新: {f}")
            else:
                print(f"[dry-run] 将更新: {f}")

    if modified == 0:
        print("未找到包含 angle_gap 的文件")
    elif args.dry_run:
        print(f"\n(dry-run 模式，共 {modified} 个文件将修改，未写入)")

if __name__ == "__main__":
    main()
