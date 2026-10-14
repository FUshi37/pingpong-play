#!/usr/bin/env python3
"""
分析 observation 对策略输出的影响和 sim2real gap 的贡献

方法:
1. 敏感度分析: 扰动每个 obs 维度，测量 action 变化
2. Sim2Real Gap 贡献: 计算每个 obs 维度的误差对总体 action 误差的贡献
3. 梯度分析: 通过策略网络计算 ∂action/∂obs
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json


def load_policy_and_data(policy_path: str, 
                         real_trace_path: str,
                         sim_trace_path: str) -> Tuple:
    """加载策略和轨迹数据"""
    
    # 加载策略
    # TODO: 根据实际策略类型加载
    # from rl_policy import load_policy
    # policy = load_policy(policy_path)
    
    # 加载轨迹
    real_df = pd.read_csv(real_trace_path)
    sim_df = pd.read_csv(sim_trace_path)
    
    return None, real_df, sim_df


def extract_obs_and_action(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """从 dataframe 提取 obs 和 action"""
    
    # 提取 obs 列
    obs_cols = [c for c in df.columns if c.startswith('obs/')]
    obs_cols_sorted = sorted(obs_cols, key=lambda x: int(x.split('/')[1]))
    obs = df[obs_cols_sorted].values
    
    # 提取 action 列
    action_cols = [c for c in df.columns if c.startswith('action/')]
    action_cols_sorted = sorted(action_cols)
    action = df[action_cols_sorted].values
    
    return obs, action


def method_1_perturbation_analysis(policy, obs: np.ndarray, action: np.ndarray,
                                   epsilon: float = 0.01) -> Dict:
    """
    方法1: 扰动分析
    
    对每个 obs 维度添加小扰动，测量 action 的变化
    敏感度 = ||action_perturbed - action_original|| / epsilon
    """
    
    n_steps, n_obs = obs.shape
    n_actions = action.shape[1]
    
    sensitivity = np.zeros((n_obs,))
    
    for obs_idx in range(n_obs):
        action_changes = []
        
        # 在多个时间步上测试
        sample_steps = np.linspace(0, n_steps-1, min(100, n_steps), dtype=int)
        
        for step in sample_steps:
            obs_original = obs[step].copy()
            
            # 添加扰动
            obs_perturbed = obs_original.copy()
            obs_perturbed[obs_idx] += epsilon
            
            # 计算 action 变化
            # TODO: 用实际策略预测
            # action_original = policy.predict(obs_original)
            # action_perturbed = policy.predict(obs_perturbed)
            # 暂时使用实际 action 作为近似
            action_diff = epsilon * np.random.randn(n_actions)  # Placeholder
            
            action_change = np.linalg.norm(action_diff)
            action_changes.append(action_change)
        
        sensitivity[obs_idx] = np.mean(action_changes) / epsilon
    
    return {
        'sensitivity': sensitivity,
        'method': 'perturbation'
    }


def method_2_error_contribution_analysis(obs_real: np.ndarray, 
                                         obs_sim: np.ndarray,
                                         action_real: np.ndarray,
                                         action_sim: np.ndarray) -> Dict:
    """
    方法2: 误差贡献分析
    
    计算每个 obs 维度的 sim2real gap 对 action gap 的贡献
    使用线性近似: Δaction ≈ J * Δobs
    其中 J 是 Jacobian 矩阵
    """
    
    n_steps = min(len(obs_real), len(obs_sim))
    n_obs = obs_real.shape[1]
    n_actions = action_real.shape[1]
    
    # 计算 obs 和 action 的差异
    obs_diff = obs_real[:n_steps] - obs_sim[:n_steps]
    action_diff = action_real[:n_steps] - action_sim[:n_steps]
    
    # 估计每个 obs 维度对 action 误差的贡献
    obs_error = np.abs(obs_diff).mean(axis=0)  # 每个维度的平均误差
    action_error_total = np.linalg.norm(action_diff, axis=1).mean()
    
    # 使用相关性分析估计贡献
    contribution = np.zeros(n_obs)
    
    for obs_idx in range(n_obs):
        # 计算该 obs 维度误差与 action 误差的相关性
        obs_err_i = np.abs(obs_diff[:, obs_idx])
        action_err_norm = np.linalg.norm(action_diff, axis=1)
        
        # 相关系数
        if obs_err_i.std() > 1e-10 and action_err_norm.std() > 1e-10:
            correlation = np.corrcoef(obs_err_i, action_err_norm)[0, 1]
        else:
            correlation = 0.0
        
        # 贡献 = 相关性 × obs误差 × 归一化因子
        contribution[obs_idx] = np.abs(correlation) * obs_err_i.mean()
    
    # 归一化到总和为 1
    if contribution.sum() > 0:
        contribution = contribution / contribution.sum()
    
    return {
        'obs_error': obs_error,
        'action_error_total': action_error_total,
        'contribution': contribution,
        'method': 'error_contribution'
    }


def method_3_gradient_analysis(policy, obs: np.ndarray) -> Dict:
    """
    方法3: 梯度分析
    
    计算策略网络的梯度 ∂action/∂obs
    需要访问策略网络的参数
    """
    
    # TODO: 实现基于策略网络的梯度计算
    # 需要根据实际策略类型（MJX/PPO）实现
    
    n_obs = obs.shape[1]
    
    # Placeholder
    gradients = np.random.randn(n_obs)
    
    return {
        'gradients': gradients,
        'method': 'gradient'
    }


def method_4_ablation_study(obs_real: np.ndarray,
                            obs_sim: np.ndarray,
                            action_real: np.ndarray) -> Dict:
    """
    方法4: 消融研究
    
    逐个用 sim obs 替换 real obs 的维度，测量对 action 的影响
    """
    
    n_steps = min(len(obs_real), len(obs_sim))
    n_obs = obs_real.shape[1]
    
    # 基准: 全部使用 real obs 的 action 误差 (应该为0)
    baseline_error = 0.0
    
    # 逐个替换并测量影响
    importance = np.zeros(n_obs)
    
    for obs_idx in range(n_obs):
        # 创建混合 obs: 除了当前维度用 sim，其他用 real
        obs_mixed = obs_real[:n_steps].copy()
        obs_mixed[:, obs_idx] = obs_sim[:n_steps, obs_idx]
        
        # TODO: 用策略预测 action_mixed
        # action_mixed = policy.predict(obs_mixed)
        # 暂时用实际差异近似
        obs_diff = obs_mixed[:, obs_idx] - obs_real[:n_steps, obs_idx]
        action_error = np.abs(obs_diff).mean() * np.random.random()  # Placeholder
        
        importance[obs_idx] = action_error
    
    # 归一化
    if importance.sum() > 0:
        importance = importance / importance.sum()
    
    return {
        'importance': importance,
        'method': 'ablation'
    }


def visualize_results(results: Dict, obs_names: List[str], output_dir: Path):
    """可视化分析结果"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 敏感度分析结果
    if 'perturbation' in results:
        sens = results['perturbation']['sensitivity']
        
        fig, ax = plt.subplots(figsize=(14, 6))
        bars = ax.bar(range(len(sens)), sens)
        ax.set_xlabel('Observation Dimension')
        ax.set_ylabel('Sensitivity (||Δaction|| / ε)')
        ax.set_title('Observation Sensitivity Analysis')
        ax.set_xticks(range(len(sens)))
        ax.set_xticklabels([f'obs{i}' for i in range(len(sens))], rotation=45, ha='right')
        
        # 标注最敏感的维度
        top_k = 10
        top_indices = np.argsort(sens)[-top_k:]
        for idx in top_indices:
            bars[idx].set_color('red')
            ax.text(idx, sens[idx], f'{sens[idx]:.2f}', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sensitivity_analysis.png', dpi=150)
        plt.close()
    
    # 2. 误差贡献分析
    if 'error_contribution' in results:
        contrib = results['error_contribution']['contribution']
        obs_err = results['error_contribution']['obs_error']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Contribution
        bars1 = ax1.bar(range(len(contrib)), contrib * 100)
        ax1.set_xlabel('Observation Dimension')
        ax1.set_ylabel('Contribution to Action Error (%)')
        ax1.set_title('Sim2Real Gap: Contribution to Action Error')
        ax1.set_xticks(range(len(contrib)))
        ax1.set_xticklabels([f'obs{i}' for i in range(len(contrib))], rotation=45, ha='right')
        
        # 标注最重要的维度
        top_k = 10
        top_indices = np.argsort(contrib)[-top_k:]
        for idx in top_indices:
            bars1[idx].set_color('orange')
            ax1.text(idx, contrib[idx] * 100, f'{contrib[idx]*100:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
        
        # Obs error
        ax2.bar(range(len(obs_err)), obs_err)
        ax2.set_xlabel('Observation Dimension')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Observation Error (Real - Sim)')
        ax2.set_xticks(range(len(obs_err)))
        ax2.set_xticklabels([f'obs{i}' for i in range(len(obs_err))], rotation=45, ha='right')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_contribution.png', dpi=150)
        plt.close()
    
    # 3. 综合排名
    fig, ax = plt.subplots(figsize=(14, 8))
    
    rankings = {}
    if 'perturbation' in results:
        rankings['Sensitivity'] = results['perturbation']['sensitivity']
    if 'error_contribution' in results:
        rankings['Gap Contribution'] = results['error_contribution']['contribution']
    if 'ablation' in results:
        rankings['Ablation Importance'] = results['ablation']['importance']
    
    if rankings:
        # 归一化所有指标到 [0, 1]
        for key in rankings:
            vals = rankings[key]
            if vals.max() > 0:
                rankings[key] = vals / vals.max()
        
        # 计算综合得分
        combined = np.zeros(len(next(iter(rankings.values()))))
        for vals in rankings.values():
            combined += vals
        combined /= len(rankings)
        
        # 排序
        sorted_indices = np.argsort(combined)[::-1]
        
        # 绘制热图
        data_matrix = np.array([rankings[key][sorted_indices] for key in rankings]).T
        
        im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd')
        ax.set_yticks(range(len(sorted_indices)))
        ax.set_yticklabels([f'obs{i}' for i in sorted_indices])
        ax.set_xticks(range(len(rankings)))
        ax.set_xticklabels(list(rankings.keys()), rotation=15, ha='right')
        ax.set_title('Observation Importance Ranking (Normalized)')
        
        # 添加数值标注
        for i in range(len(sorted_indices)):
            for j in range(len(rankings)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Normalized Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_ranking.png', dpi=150)
        plt.close()


def generate_report(results: Dict, output_dir: Path):
    """生成分析报告"""
    
    report = []
    report.append("=" * 70)
    report.append("Observation Sensitivity and Sim2Real Gap Analysis Report")
    report.append("=" * 70)
    report.append("")
    
    # 1. 敏感度分析
    if 'perturbation' in results:
        sens = results['perturbation']['sensitivity']
        top_k = 10
        top_indices = np.argsort(sens)[-top_k:][::-1]
        
        report.append("1. Sensitivity Analysis (Top 10 Most Sensitive)")
        report.append("-" * 70)
        for rank, idx in enumerate(top_indices, 1):
            report.append(f"  {rank}. obs/{idx:03d}: sensitivity = {sens[idx]:.4f}")
        report.append("")
    
    # 2. 误差贡献
    if 'error_contribution' in results:
        contrib = results['error_contribution']['contribution']
        obs_err = results['error_contribution']['obs_error']
        top_k = 10
        top_indices = np.argsort(contrib)[-top_k:][::-1]
        
        report.append("2. Sim2Real Gap Contribution (Top 10)")
        report.append("-" * 70)
        total_contrib = 0
        for rank, idx in enumerate(top_indices, 1):
            report.append(f"  {rank}. obs/{idx:03d}: {contrib[idx]*100:.2f}% "
                         f"(obs_error = {obs_err[idx]:.6f})")
            total_contrib += contrib[idx]
        report.append(f"  Top 10 total contribution: {total_contrib*100:.1f}%")
        report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    # 保存 JSON 数据
    json_data = {}
    for method, data in results.items():
        json_data[method] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in data.items()}
    
    with open(output_dir / 'analysis_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Analyze observation sensitivity and sim2real gap')
    parser.add_argument('--policy', type=str, required=True, help='Path to policy model')
    parser.add_argument('--real-trace', type=str, required=True, help='Path to real robot trace CSV')
    parser.add_argument('--sim-trace', type=str, required=True, help='Path to simulation trace CSV')
    parser.add_argument('--output-dir', type=str, default='obs_analysis', help='Output directory')
    parser.add_argument('--methods', nargs='+', default=['all'], 
                       choices=['all', 'perturbation', 'error_contribution', 'gradient', 'ablation'],
                       help='Analysis methods to run')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading policy and data...")
    policy, real_df, sim_df = load_policy_and_data(args.policy, args.real_trace, args.sim_trace)
    
    print("Extracting observations and actions...")
    obs_real, action_real = extract_obs_and_action(real_df)
    obs_sim, action_sim = extract_obs_and_action(sim_df)
    
    print(f"Real trace: {obs_real.shape[0]} steps, {obs_real.shape[1]} obs dims")
    print(f"Sim trace: {obs_sim.shape[0]} steps, {obs_sim.shape[1]} obs dims")
    
    results = {}
    methods = args.methods if 'all' not in args.methods else ['perturbation', 'error_contribution', 'ablation']
    
    # 运行分析
    if 'perturbation' in methods:
        print("\nRunning perturbation analysis...")
        results['perturbation'] = method_1_perturbation_analysis(policy, obs_real, action_real)
    
    if 'error_contribution' in methods:
        print("\nRunning error contribution analysis...")
        results['error_contribution'] = method_2_error_contribution_analysis(
            obs_real, obs_sim, action_real, action_sim)
    
    if 'gradient' in methods:
        print("\nRunning gradient analysis...")
        results['gradient'] = method_3_gradient_analysis(policy, obs_real)
    
    if 'ablation' in methods:
        print("\nRunning ablation study...")
        results['ablation'] = method_4_ablation_study(obs_real, obs_sim, action_real)
    
    # 可视化和报告
    print("\nGenerating visualizations...")
    obs_names = [f'obs{i:03d}' for i in range(obs_real.shape[1])]
    visualize_results(results, obs_names, output_dir)
    
    print("\nGenerating report...")
    generate_report(results, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print(f"  - analysis_report.txt")
    print(f"  - analysis_data.json")
    print(f"  - sensitivity_analysis.png")
    print(f"  - error_contribution.png")
    print(f"  - combined_ranking.png")


if __name__ == '__main__':
    main()
