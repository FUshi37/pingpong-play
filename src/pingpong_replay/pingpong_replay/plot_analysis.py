#!/usr/bin/env python3
"""Plot comparison of actual vs theoretical right-arm trajectory.

Reads the replay_analysis_*.csv file produced by replay_node with record_data=true
and generates comparison plots for position, velocity, and acceleration.
"""

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Right-arm joint names (must match replay_node.py)
RIGHT_ARM_JOINTS: List[str] = [
    "RightArm-0", "RightArm-1", "RightArm-2",
    "RightArm-3", "RightArm-4", "RightArm-5", "RightArm-6",
]

JOINT_DISPLAY_NAMES = [
    "Joint 0", "Joint 1", "Joint 2",
    "Joint 3", "Joint 4", "Joint 5", "Joint 6",
]


def plot_comparison(csv_path: str, output_path: Optional[str] = None,
                   joint_indices: Optional[List[int]] = None) -> None:
    """Generate comparison plots from recorded replay data.

    Args:
        csv_path: Path to the replay_analysis_*.csv file
        output_path: Path to save the plot (if None, displays interactively)
        joint_indices: List of joint indices to plot (if None, plots all joints)
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        print("Error: CSV file is empty")
        return

    # Check if theoretical data is available
    has_theoretical = f'theoretical_q/{RIGHT_ARM_JOINTS[0]}' in df.columns

    if not has_theoretical:
        print("Warning: No theoretical data found in CSV. Only actual data will be plotted.")

    # Determine which joints to plot
    if joint_indices is None:
        joint_indices = list(range(len(RIGHT_ARM_JOINTS)))

    n_joints = len(joint_indices)

    # Create figure with subplots: 3 rows (q, qvel, qdd) x n_joints columns
    fig, axes = plt.subplots(3, n_joints, figsize=(4*n_joints, 12))
    if n_joints == 1:
        axes = axes.reshape(-1, 1)

    # Extract time (use step index as x-axis)
    time = df['step'].values

    for col_idx, joint_idx in enumerate(joint_indices):
        joint_name = RIGHT_ARM_JOINTS[joint_idx]
        joint_display = JOINT_DISPLAY_NAMES[joint_idx]

        # --- Position (q) ---
        ax = axes[0, col_idx]
        actual_q = np.degrees(df[f'actual_q/{joint_name}'].values)
        cmd_q = np.degrees(df[f'cmd_q/{joint_name}'].values)

        ax.plot(time, actual_q, 'b-', linewidth=1.5, label='Actual', alpha=0.8)
        ax.plot(time, cmd_q, 'g--', linewidth=1.0, label='Command', alpha=0.7)

        if has_theoretical:
            theoretical_q = np.degrees(df[f'theoretical_q/{joint_name}'].values)
            ax.plot(time, theoretical_q, 'r:', linewidth=1.5, label='Theoretical', alpha=0.7)

        ax.set_ylabel('Angle (deg)')
        ax.set_title(f'{joint_display} - Position')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Velocity (qvel) ---
        ax = axes[1, col_idx]
        actual_qvel = np.degrees(df[f'actual_qvel/{joint_name}'].values)

        ax.plot(time, actual_qvel, 'b-', linewidth=1.5, label='Actual', alpha=0.8)

        if has_theoretical:
            theoretical_qvel = np.degrees(df[f'theoretical_qvel/{joint_name}'].values)
            ax.plot(time, theoretical_qvel, 'r:', linewidth=1.5, label='Theoretical', alpha=0.7)

        ax.set_ylabel('Velocity (deg/s)')
        ax.set_title(f'{joint_display} - Velocity')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Acceleration (qdd) ---
        ax = axes[2, col_idx]
        actual_qdd = np.degrees(df[f'actual_qdd/{joint_name}'].values)

        ax.plot(time, actual_qdd, 'b-', linewidth=1.5, label='Actual', alpha=0.8)

        if has_theoretical:
            theoretical_qdd = np.degrees(df[f'theoretical_qdd/{joint_name}'].values)
            ax.plot(time, theoretical_qdd, 'r:', linewidth=1.5, label='Theoretical', alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel('Acceleration (deg/s²)')
        ax.set_title(f'{joint_display} - Acceleration')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        print(f"Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("Plot saved successfully")
    else:
        print("Displaying plot (close window to continue)...")
        plt.show()


def plot_error_analysis(csv_path: str, output_path: Optional[str] = None,
                       joint_indices: Optional[List[int]] = None) -> None:
    """Generate error analysis plots.

    Args:
        csv_path: Path to the replay_analysis_*.csv file
        output_path: Path to save the plot (if None, displays interactively)
        joint_indices: List of joint indices to plot (if None, plots all joints)
    """
    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Check if theoretical data is available
    has_theoretical = f'theoretical_q/{RIGHT_ARM_JOINTS[0]}' in df.columns

    if not has_theoretical:
        print("Error: No theoretical data found. Cannot compute tracking errors.")
        return

    # Determine which joints to plot
    if joint_indices is None:
        joint_indices = list(range(len(RIGHT_ARM_JOINTS)))

    n_joints = len(joint_indices)

    # Create figure with subplots: 3 rows (q, qvel, qdd) x n_joints columns
    fig, axes = plt.subplots(3, n_joints, figsize=(4*n_joints, 12))
    if n_joints == 1:
        axes = axes.reshape(-1, 1)

    # Extract time
    time = df['step'].values

    for col_idx, joint_idx in enumerate(joint_indices):
        joint_name = RIGHT_ARM_JOINTS[joint_idx]
        joint_display = JOINT_DISPLAY_NAMES[joint_idx]

        # --- Position error ---
        ax = axes[0, col_idx]
        actual_q = df[f'actual_q/{joint_name}'].values
        theoretical_q = df[f'theoretical_q/{joint_name}'].values
        error_q = np.degrees(actual_q - theoretical_q)

        ax.plot(time, error_q, 'k-', linewidth=1.0, alpha=0.8)
        ax.axhline(0, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Position Error (deg)')
        ax.set_title(f'{joint_display} - Position Error')
        ax.grid(True, alpha=0.3)

        # Add statistics
        mean_err = np.mean(np.abs(error_q))
        max_err = np.max(np.abs(error_q))
        rms_err = np.sqrt(np.mean(error_q**2))
        ax.text(0.02, 0.98, f'Mean: {mean_err:.3f}°\nMax: {max_err:.3f}°\nRMS: {rms_err:.3f}°',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)

        # --- Velocity error ---
        ax = axes[1, col_idx]
        actual_qvel = df[f'actual_qvel/{joint_name}'].values
        theoretical_qvel = df[f'theoretical_qvel/{joint_name}'].values
        error_qvel = np.degrees(actual_qvel - theoretical_qvel)

        ax.plot(time, error_qvel, 'k-', linewidth=1.0, alpha=0.8)
        ax.axhline(0, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel('Velocity Error (deg/s)')
        ax.set_title(f'{joint_display} - Velocity Error')
        ax.grid(True, alpha=0.3)

        mean_err = np.mean(np.abs(error_qvel))
        max_err = np.max(np.abs(error_qvel))
        rms_err = np.sqrt(np.mean(error_qvel**2))
        ax.text(0.02, 0.98, f'Mean: {mean_err:.2f}°/s\nMax: {max_err:.2f}°/s\nRMS: {rms_err:.2f}°/s',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)

        # --- Acceleration error ---
        ax = axes[2, col_idx]
        actual_qdd = df[f'actual_qdd/{joint_name}'].values
        theoretical_qdd = df[f'theoretical_qdd/{joint_name}'].values
        error_qdd = np.degrees(actual_qdd - theoretical_qdd)

        ax.plot(time, error_qdd, 'k-', linewidth=1.0, alpha=0.8)
        ax.axhline(0, color='r', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Acceleration Error (deg/s²)')
        ax.set_title(f'{joint_display} - Acceleration Error')
        ax.grid(True, alpha=0.3)

        mean_err = np.mean(np.abs(error_qdd))
        max_err = np.max(np.abs(error_qdd))
        rms_err = np.sqrt(np.mean(error_qdd**2))
        ax.text(0.02, 0.98, f'Mean: {mean_err:.1f}°/s²\nMax: {max_err:.1f}°/s²\nRMS: {rms_err:.1f}°/s²',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=8)

    plt.tight_layout()

    if output_path:
        print(f"Saving error plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("Error plot saved successfully")
    else:
        print("Displaying error plot (close window to continue)...")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot comparison of actual vs theoretical right-arm trajectory')
    parser.add_argument('csv_path', help='Path to replay_analysis_*.csv file')
    parser.add_argument('-o', '--output', help='Output path for the plot (PNG/PDF)')
    parser.add_argument('--error-plot', help='Output path for error analysis plot')
    parser.add_argument('-j', '--joints', type=int, nargs='+',
                       help='Joint indices to plot (0-6), default: all joints')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively (default if no output path)')

    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        return 1

    # Validate joint indices
    if args.joints:
        for j in args.joints:
            if j < 0 or j >= len(RIGHT_ARM_JOINTS):
                print(f"Error: Invalid joint index {j}. Must be 0-{len(RIGHT_ARM_JOINTS)-1}")
                return 1

    # Generate comparison plot
    output_path = args.output if args.output and not args.show else None
    plot_comparison(args.csv_path, output_path, args.joints)

    # Generate error plot if requested
    if args.error_plot or args.show:
        error_output = args.error_plot if args.error_plot and not args.show else None
        plot_error_analysis(args.csv_path, error_output, args.joints)

    return 0


if __name__ == '__main__':
    sys.exit(main())
