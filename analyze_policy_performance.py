#!/usr/bin/env python3
"""
Analyze policy performance by comparing recorded obs/actions with reference traces.
Generates obs_comparison.png and action_comparison.png.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_obs_comparison(recorded_trace, reference_obs, output_dir):
    """Generate observation comparison plots."""
    recorded = pd.read_csv(recorded_trace)
    reference = pd.read_csv(reference_obs)

    n = min(len(recorded), len(reference))
    time = np.arange(n) * 0.005  # 5ms timestep

    # Select key observations to plot
    obs_groups = {
        "Ball position": ([20, 21, 22], ["x", "y", "z"]),
        "Ball velocity": ([23, 24, 25], ["x", "y", "z"]),
        "Racket position": ([26, 27, 28], ["x", "y", "z"]),
        "Racket velocity": ([29, 30, 31], ["x", "y", "z"]),
        "Rel position": ([32, 33, 34], ["x", "y", "z"]),
        "Phase features": ([65, 66], ["t_contact", "t_margin"]),
    }

    fig, axes = plt.subplots(len(obs_groups), 1, figsize=(14, 12), sharex=True)

    for ax_idx, (group_name, (obs_indices, labels)) in enumerate(obs_groups.items()):
        ax = axes[ax_idx]

        for obs_idx, label in zip(obs_indices, labels):
            recorded_col = f"obs/{obs_idx:03d}"
            reference_col = f"obs/{obs_idx:03d}"

            if recorded_col in recorded.columns and reference_col in reference.columns:
                recorded_vals = recorded[recorded_col].iloc[:n].values
                reference_vals = reference[reference_col].iloc[:n].values

                # Plot recorded (solid) and reference (dashed)
                ax.plot(time, recorded_vals, label=f"{label} (real)", linewidth=1.5, alpha=0.8)
                ax.plot(time, reference_vals, '--', label=f"{label} (ref)", linewidth=1.5, alpha=0.6)

        ax.set_ylabel(group_name, fontsize=10)
        ax.legend(loc='upper right', fontsize=8, ncol=len(labels))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_title("Observation Comparison: Real-robot vs MJX Reference", fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = Path(output_dir) / "obs_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Generated: {output_path}")


def plot_action_comparison(recorded_trace, reference_action, output_dir):
    """Generate action comparison plots."""
    recorded = pd.read_csv(recorded_trace)
    reference = pd.read_csv(reference_action)

    n = min(len(recorded), len(reference))
    time = np.arange(n) * 0.005

    # Action dimensions (7 joints)
    action_names = ["RightArm-0", "RightArm-1", "RightArm-2", "RightArm-3",
                    "RightArm-4", "RightArm-5", "RightArm-6"]

    fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)

    for joint_idx, (ax, joint_name) in enumerate(zip(axes, action_names)):
        # Try to find action columns in recorded trace
        # Format might be action/000, action/001, etc.
        recorded_col = f"action/{joint_idx:03d}"

        # Reference trace uses different format
        reference_col = f"raw_action/{joint_name}"

        if recorded_col in recorded.columns and reference_col in reference.columns:
            recorded_vals = recorded[recorded_col].iloc[:n].values
            reference_vals = reference[reference_col].iloc[:n].values

            ax.plot(time, recorded_vals, 'b-', label='Real-robot', linewidth=1.5, alpha=0.8)
            ax.plot(time, reference_vals, 'r--', label='MJX reference', linewidth=1.5, alpha=0.6)

            # Calculate RMS difference
            rms_diff = np.sqrt(np.mean((recorded_vals - reference_vals)**2))
            ax.text(0.02, 0.95, f'RMS diff: {rms_diff:.4f}',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_ylabel(joint_name, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_title("Action Comparison: Real-robot vs MJX Reference", fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = Path(output_dir) / "action_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze policy performance")
    parser.add_argument("recorded_trace", type=str, help="Policy obs trace CSV")
    parser.add_argument("--reference-obs", type=str, required=True, help="Reference obs trace CSV")
    parser.add_argument("--reference-action", type=str, required=True, help="Reference action trace CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating observation comparison...")
    plot_obs_comparison(args.recorded_trace, args.reference_obs, args.output_dir)

    print("\nGenerating action comparison...")
    plot_action_comparison(args.recorded_trace, args.reference_action, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
