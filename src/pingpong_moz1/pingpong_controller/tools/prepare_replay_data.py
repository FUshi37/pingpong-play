#!/usr/bin/env python3
"""
准备实录数据用于策略replay测试
从detection CSV提取前N秒的有效球状态
"""

import pandas as pd
import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='准备replay数据')
    parser.add_argument('detection_csv', help='检测CSV文件路径')
    parser.add_argument('--duration', type=float, default=12.0, help='提取时长(秒)')
    parser.add_argument('--output', '-o', help='输出CSV文件路径')
    args = parser.parse_args()

    # 加载检测数据
    df = pd.read_csv(args.detection_csv)

    # 提取前N秒
    t_start = df['t_sec'].min()
    df_subset = df[df['t_sec'] <= t_start + args.duration].copy()

    # 只保留有效检测
    df_valid = df_subset[df_subset['detected'] == 1].copy()

    if len(df_valid) == 0:
        print(f"错误: 前{args.duration}秒没有有效检测！", file=sys.stderr)
        return 1

    # 重新计算相对时间
    df_valid['t_rel'] = df_valid['t_sec'] - df_valid['t_sec'].min()

    # 保存
    if args.output:
        output_path = args.output
    else:
        output_path = args.detection_csv.replace('.csv', f'_replay_{args.duration:.0f}s.csv')

    # 只保留replay需要的列。pingpong_node replay模式需要 valid 列。
    replay_df = df_valid[['t_rel', 'x_mm', 'y_mm', 'z_mm', 'vx_mm_s', 'vy_mm_s', 'vz_mm_s']].copy()
    replay_df.insert(1, 'valid', 1)
    replay_df.to_csv(output_path, index=False)

    print(f"✅ 提取完成:")
    print(f"   输入: {args.detection_csv}")
    print(f"   输出: {output_path}")
    print(f"   时长: {args.duration:.1f} 秒")
    print(f"   有效帧: {len(replay_df)}")
    print(f"   频率: ~{len(replay_df)/args.duration:.1f} Hz")

    # 显示数据范围
    print(f"\n数据范围:")
    print(f"   x: [{replay_df['x_mm'].min():.1f}, {replay_df['x_mm'].max():.1f}] mm")
    print(f"   y: [{replay_df['y_mm'].min():.1f}, {replay_df['y_mm'].max():.1f}] mm")
    print(f"   z: [{replay_df['z_mm'].min():.1f}, {replay_df['z_mm'].max():.1f}] mm")

    return 0

if __name__ == '__main__':
    sys.exit(main())
