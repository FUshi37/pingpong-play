#!/usr/bin/env python3
"""
Analyze joint tracking: compare recorded joint positions, velocities, and actions
with MJX reference to identify actuator dynamics mismatch.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_joint_tracking(recorded_trace, reference_obs, output_path):
    """Generate joint tracking analysis plot."""
    recorded = pd.read_csv(recorded_trace)
    reference = pd.read_csv(reference_obs)

    n = min(len(recorded), len(reference))
    time = np.arange(n) * 0.005  # 5ms timestep

    # Select 3 representative joints
    joint_indices = [0, 3, 6]  # First, middle, last joints
    joint_names = [f"Joint {i}" for i in joint_indices]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    for row_idx, joint_idx in enumerate(joint_indices):
        # Joint position (obs[0-6])
        ax_pos = fig.add_subplot(gs[row_idx, 0])
        recorded_pos_col = f"obs/{joint_idx:03d}"
        reference_pos_col = f"obs/{joint_idx:03d}"

        if recorded_pos_col in recorded.columns and reference_pos_col in reference.columns:
            recorded_pos = recorded[recorded_pos_col].iloc[:n].values
            reference_pos = reference[reference_pos_col].iloc[:n].values

            ax_pos.plot(time, recorded_pos, 'b-', label='Real-robot', linewidth=1.5, alpha=0.8)
            ax_pos.plot(time, reference_pos, 'r--', label='MJX ref', linewidth=1.5, alpha=0.6)

            rms_diff = np.sqrt(np.mean((recorded_pos - reference_pos)**2))
            ax_pos.text(0.02, 0.98, f'RMS: {rms_diff:.4f} rad',
                       transform=ax_pos.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_pos.set_ylabel(f"{joint_names[row_idx]}\nPosition (rad)", fontsize=10)
        ax_pos.legend(loc='upper right', fontsize=8)
        ax_pos.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_pos.set_title("Joint Position", fontsize=12, fontweight='bold')

        # Joint velocity (obs[7-13])
        ax_vel = fig.add_subplot(gs[row_idx, 1])
        recorded_vel_col = f"obs/{joint_idx + 7:03d}"
        reference_vel_col = f"obs/{joint_idx + 7:03d}"

        if recorded_vel_col in recorded.columns and reference_vel_col in reference.columns:
            recorded_vel = recorded[recorded_vel_col].iloc[:n].values
            reference_vel = reference[reference_vel_col].iloc[:n].values

            ax_vel.plot(time, recorded_vel, 'b-', label='Real-robot', linewidth=1.5, alpha=0.8)
            ax_vel.plot(time, reference_vel, 'r--', label='MJX ref', linewidth=1.5, alpha=0.6)

            rms_diff = np.sqrt(np.mean((recorded_vel - reference_vel)**2))
            ax_vel.text(0.02, 0.98, f'RMS: {rms_diff:.4f} rad/s',
                       transform=ax_vel.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_vel.set_ylabel("Velocity (rad/s)", fontsize=10)
        ax_vel.legend(loc='upper right', fontsize=8)
        ax_vel.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_vel.set_title("Joint Velocity", fontsize=12, fontweight='bold')

        # Action (prev_action obs[35-41])
        ax_action = fig.add_subplot(gs[row_idx, 2])
        recorded_action_col = f"obs/{joint_idx + 35:03d}"
        reference_action_col = f"obs/{joint_idx + 35:03d}"

        if recorded_action_col in recorded.columns and reference_action_col in reference.columns:
            recorded_action = recorded[recorded_action_col].iloc[:n].values
            reference_action = reference[reference_action_col].iloc[:n].values

            ax_action.plot(time, recorded_action, 'b-', label='Real-robot', linewidth=1.5, alpha=0.8)
            ax_action.plot(time, reference_action, 'r--', label='MJX ref', linewidth=1.5, alpha=0.6)

            rms_diff = np.sqrt(np.mean((recorded_action - reference_action)**2))
            ax_action.text(0.02, 0.98, f'RMS: {rms_diff:.4f}',
                          transform=ax_action.transAxes, fontsize=9,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax_action.set_ylabel("Action", fontsize=10)
        ax_action.legend(loc='upper right', fontsize=8)
        ax_action.grid(True, alpha=0.3)
        if row_idx == 0:
            ax_action.set_title("Previous Action", fontsize=12, fontweight='bold')

        # Only set x-label on bottom row
        if row_idx == len(joint_indices) - 1:
            ax_pos.set_xlabel("Time (s)", fontsize=10)
            ax_vel.set_xlabel("Time (s)", fontsize=10)
            ax_action.set_xlabel("Time (s)", fontsize=10)

    fig.suptitle("Joint Tracking Analysis: Real-robot vs MJX Reference",
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze joint tracking")
    parser.add_argument("recorded_trace", type=str, help="Policy obs trace CSV")
    parser.add_argument("--reference", type=str, required=True, help="Reference obs trace CSV")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")

    args = parser.parse_args()

    print("Generating joint tracking analysis...")
    plot_joint_tracking(args.recorded_trace, args.reference, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
