#!/usr/bin/env python3
"""
简化版本：直接分析现有的 policy trace 数据
不需要加载策略，基于数据的统计分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json


def analyze_obs_impact(real_trace_path: str, sim_trace_path: str, output_dir: str):
    """分析 observation 对 sim2real gap 的影响"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Observation Impact Analysis")
    print("=" * 70)
    print()
    
    # 加载数据
    print("Loading traces...")
    real_df = pd.read_csv(real_trace_path)
    sim_df = pd.read_csv(sim_trace_path)
    
    # 提取 obs 和 action
    obs_cols = sorted([c for c in real_df.columns if c.startswith('obs/')],
                     key=lambda x: int(x.split('/')[1]))

    # 尝试多种 action 列名格式
    action_cols_real = None
    action_cols_sim = None

    for prefix in ['action/', 'obs_prev_action/', 'arm_cmd/']:
        cols = sorted([c for c in real_df.columns if c.startswith(prefix) and 'RightArm' in c])
        if cols:
            action_cols_real = cols
            break

    for prefix in ['action/', 'obs_prev_action/', 'arm_cmd/']:
        cols = sorted([c for c in sim_df.columns if c.startswith(prefix) and 'RightArm' in c])
        if cols:
            action_cols_sim = cols
            break

    if action_cols_real is None:
        print("ERROR: Could not find action columns in real trace")
        print("Available columns:", [c for c in real_df.columns if 'action' in c.lower() or 'cmd' in c.lower()][:10])
        return

    if action_cols_sim is None:
        print("ERROR: Could not find action columns in sim trace")
        print("Available columns:", [c for c in sim_df.columns if 'action' in c.lower() or 'cmd' in c.lower()][:10])
        return

    n_steps = min(len(real_df), len(sim_df))
    obs_real = real_df[obs_cols].values[:n_steps]
    obs_sim = sim_df[obs_cols].values[:n_steps]
    action_real = real_df[action_cols_real].values[:n_steps]
    action_sim = sim_df[action_cols_sim].values[:n_steps]

    n_obs = len(obs_cols)
    n_actions = min(len(action_cols_real), len(action_cols_sim))
    
    print(f"  Real: {len(real_df)} steps, {n_obs} obs dims, {n_actions} actions")
    print(f"  Sim:  {len(sim_df)} steps")
    print(f"  Analyzing first {n_steps} steps")
    print()
    
    # 计算差异
    obs_diff = obs_real - obs_sim
    action_diff = action_real - action_sim
    
    # ===== 分析1: 观测误差统计 =====
    print("Analysis 1: Observation Error Statistics")
    print("-" * 70)
    
    obs_error_mean = np.abs(obs_diff).mean(axis=0)
    obs_error_std = np.abs(obs_diff).std(axis=0)
    obs_error_max = np.abs(obs_diff).max(axis=0)
    
    # 找出误差最大的维度
    top_k = 10
    top_error_indices = np.argsort(obs_error_mean)[-top_k:][::-1]
    
    print(f"Top {top_k} observations with largest errors:")
    for rank, idx in enumerate(top_error_indices, 1):
        print(f"  {rank}. {obs_cols[idx]:20s}: mean={obs_error_mean[idx]:.6f}, "
              f"std={obs_error_std[idx]:.6f}, max={obs_error_max[idx]:.6f}")
    print()
    
    # ===== 分析2: 相关性分析 =====
    print("Analysis 2: Correlation with Action Error")
    print("-" * 70)
    
    action_error_norm = np.linalg.norm(action_diff, axis=1)
    
    correlations = np.zeros(n_obs)
    for i in range(n_obs):
        obs_err_i = np.abs(obs_diff[:, i])
        if obs_err_i.std() > 1e-10 and action_error_norm.std() > 1e-10:
            correlations[i] = np.corrcoef(obs_err_i, action_error_norm)[0, 1]
    
    # 找出相关性最强的维度
    top_corr_indices = np.argsort(np.abs(correlations))[-top_k:][::-1]
    
    print(f"Top {top_k} observations most correlated with action error:")
    for rank, idx in enumerate(top_corr_indices, 1):
        print(f"  {rank}. {obs_cols[idx]:20s}: corr={correlations[idx]:+.4f}, "
              f"obs_err={obs_error_mean[idx]:.6f}")
    print()
    
    # ===== 分析3: 贡献度分析 =====
    print("Analysis 3: Contribution to Action Error")
    print("-" * 70)
    
    # 贡献度 = |相关系数| × 观测误差
    contribution = np.abs(correlations) * obs_error_mean
    contribution_norm = contribution / contribution.sum() if contribution.sum() > 0 else contribution
    
    top_contrib_indices = np.argsort(contribution_norm)[-top_k:][::-1]
    
    print(f"Top {top_k} observations contributing to action error:")
    cumulative = 0
    for rank, idx in enumerate(top_contrib_indices, 1):
        cumulative += contribution_norm[idx]
        print(f"  {rank}. {obs_cols[idx]:20s}: {contribution_norm[idx]*100:5.2f}% "
              f"(cumulative: {cumulative*100:5.1f}%)")
    print()
    
    # ===== 分析4: 时间演化分析 =====
    print("Analysis 4: Temporal Evolution")
    print("-" * 70)
    
    # 计算误差随时间的演化
    window_size = 50
    n_windows = n_steps // window_size
    
    temporal_contribution = np.zeros((n_windows, n_obs))
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size
        
        obs_err_w = np.abs(obs_diff[start:end]).mean(axis=0)
        action_err_w = np.linalg.norm(action_diff[start:end], axis=1)
        
        for i in range(n_obs):
            if obs_diff[start:end, i].std() > 1e-10 and action_err_w.std() > 1e-10:
                corr_w = np.corrcoef(np.abs(obs_diff[start:end, i]), action_err_w)[0, 1]
                temporal_contribution[w, i] = np.abs(corr_w) * obs_err_w[i]
    
    # 检查贡献度是否随时间变化
    contrib_variance = temporal_contribution.std(axis=0)
    stable_indices = np.where(contrib_variance < contribution.mean() * 0.1)[0]
    unstable_indices = np.where(contrib_variance >= contribution.mean() * 0.1)[0]
    
    print(f"Stable contributions (low variance over time): {len(stable_indices)}")
    print(f"Unstable contributions (high variance over time): {len(unstable_indices)}")
    print()
    
    # ===== 可视化 =====
    print("Generating visualizations...")
    
    # 图1: 误差和相关性
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 子图1: 观测误差
    ax = axes[0]
    bars = ax.bar(range(n_obs), obs_error_mean)
    for idx in top_error_indices[:5]:
        bars[idx].set_color('red')
    ax.set_xlabel('Observation Dimension', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('Observation Error (Real - Sim)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 子图2: 相关性
    ax = axes[1]
    colors = ['red' if c > 0 else 'blue' for c in correlations]
    bars = ax.bar(range(n_obs), correlations, color=colors, alpha=0.6)
    for idx in top_corr_indices[:5]:
        bars[idx].set_edgecolor('black')
        bars[idx].set_linewidth(2)
    ax.set_xlabel('Observation Dimension', fontsize=12)
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_title('Correlation with Action Error', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    
    # 子图3: 贡献度
    ax = axes[2]
    bars = ax.bar(range(n_obs), contribution_norm * 100)
    for idx in top_contrib_indices[:5]:
        bars[idx].set_color('orange')
    ax.set_xlabel('Observation Dimension', fontsize=12)
    ax.set_ylabel('Contribution to Action Error (%)', fontsize=12)
    ax.set_title('Contribution to Sim2Real Gap', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'obs_impact_analysis.png', dpi=150)
    plt.close()
    print(f"  Saved: obs_impact_analysis.png")
    
    # 图2: Top 观测维度详细分析
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(top_contrib_indices[:6]):
        ax = axes[i]
        
        # 绘制 obs 误差随时间的变化
        ax.plot(np.abs(obs_diff[:, idx]), label='Obs Error', linewidth=1, alpha=0.7)
        
        # 叠加 action 误差
        ax2 = ax.twinx()
        ax2.plot(action_error_norm, color='red', label='Action Error', 
                linewidth=1, alpha=0.5, linestyle='--')
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel(f'{obs_cols[idx]} Error', fontsize=10, color='blue')
        ax2.set_ylabel('Action Error', fontsize=10, color='red')
        ax.set_title(f'{obs_cols[idx]}\nContribution: {contribution_norm[idx]*100:.2f}%', 
                    fontsize=11, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_obs_details.png', dpi=150)
    plt.close()
    print(f"  Saved: top_obs_details.png")
    
    # 图3: 时间演化热图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 只显示贡献度最大的前20个维度
    top_20 = top_contrib_indices[:20]
    heatmap_data = temporal_contribution[:, top_20].T
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Time Window', fontsize=12)
    ax.set_ylabel('Observation Dimension', fontsize=12)
    ax.set_title('Temporal Evolution of Contribution (Top 20 Obs)', fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([obs_cols[i] for i in top_20], fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Contribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_evolution.png', dpi=150)
    plt.close()
    print(f"  Saved: temporal_evolution.png")
    
    # ===== 保存结果 =====
    results = {
        'obs_error': {
            'mean': obs_error_mean.tolist(),
            'std': obs_error_std.tolist(),
            'max': obs_error_max.tolist(),
        },
        'correlation': correlations.tolist(),
        'contribution': contribution_norm.tolist(),
        'top_contributors': {
            'indices': top_contrib_indices.tolist(),
            'names': [obs_cols[i] for i in top_contrib_indices],
            'values': [contribution_norm[i] for i in top_contrib_indices],
        },
        'action_error': {
            'mean': float(action_error_norm.mean()),
            'std': float(action_error_norm.std()),
            'max': float(action_error_norm.max()),
        }
    }
    
    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: analysis_results.json")
    
    # 保存报告
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Observation Impact Analysis Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total observations: {n_obs}\n")
        f.write(f"Total actions: {n_actions}\n")
        f.write(f"Time steps analyzed: {n_steps}\n\n")
        
        f.write(f"Action error: mean={action_error_norm.mean():.6f}, "
                f"std={action_error_norm.std():.6f}, max={action_error_norm.max():.6f}\n\n")
        
        f.write(f"Top {top_k} Observations Contributing to Sim2Real Gap:\n")
        f.write("-" * 70 + "\n")
        cumulative = 0
        for rank, idx in enumerate(top_contrib_indices, 1):
            cumulative += contribution_norm[idx]
            f.write(f"{rank:2d}. {obs_cols[idx]:25s}: {contribution_norm[idx]*100:6.2f}% "
                   f"(cumulative: {cumulative*100:6.1f}%)\n")
            f.write(f"     obs_error={obs_error_mean[idx]:.6f}, corr={correlations[idx]:+.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"  Saved: analysis_report.txt")
    print()
    print("=" * 70)
    print(f"Analysis complete! Results in: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_obs_impact_simple.py <real_trace.csv> <sim_trace.csv> [output_dir]")
        print()
        print("Example:")
        print("  python analyze_obs_impact_simple.py \\")
        print("    src/pingpong_moz1/pingpong_controller/outputs/rl_sim/real_policy_trace.csv \\")
        print("    src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_obs_trace.csv \\")
        print("    obs_impact_analysis")
        sys.exit(1)
    
    real_trace = sys.argv[1]
    sim_trace = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'obs_impact_analysis'
    
    analyze_obs_impact(real_trace, sim_trace, output_dir)
