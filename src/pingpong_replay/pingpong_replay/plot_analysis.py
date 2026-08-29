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

try:
    from scipy.spatial.transform import Rotation as _R
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


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
        plt.close(fig)
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
        plt.close(fig)
        print("Error plot saved successfully")
    else:
        print("Displaying error plot (close window to continue)...")
        plt.show()


def _ee_columns_present(df: pd.DataFrame, prefix: str) -> bool:
    """True if df has prefix_pos/{x,y,z} and prefix_quat/{x,y,z,w} columns."""
    pos_ok = all(f'{prefix}_pos/{a}' in df.columns for a in 'xyz')
    quat_ok = all(f'{prefix}_quat/{a}' in df.columns for a in 'xyzw')
    return pos_ok and quat_ok


def _extract_ee(df: pd.DataFrame, prefix: str):
    """Return (pos Nx3 [m], euler Nx3 [deg]) for an EE prefix, or (None, None).

    Quaternions are stored xyzw. Rows with NaN are kept as NaN so gaps show as
    breaks in the line rather than silently shifting the time axis.
    """
    if not _ee_columns_present(df, prefix):
        return None, None
    pos = np.column_stack([df[f'{prefix}_pos/{a}'].values for a in 'xyz'])
    quat = np.column_stack([df[f'{prefix}_quat/{a}'].values for a in 'xyzw'])
    euler = np.full((len(df), 3), np.nan)
    if _HAVE_SCIPY:
        ok = np.isfinite(quat).all(axis=1) & (
            np.linalg.norm(quat, axis=1) > 1e-6)
        if ok.any():
            euler[ok] = _R.from_quat(quat[ok]).as_euler('xyz', degrees=True)
    return pos, euler


def _auto_label(df: pd.DataFrame, fallback: str) -> str:
    """Guess a run's label from its columns: cartesian runs have cart_cmd_*."""
    if _ee_columns_present(df, 'cart_cmd'):
        return 'cartesian'
    if _ee_columns_present(df, 'cart_actual'):
        return 'joint'
    return fallback


def plot_ee_comparison(csv_path: str, output_path: Optional[str] = None,
                       compare_path: Optional[str] = None,
                       labels: Optional[List[str]] = None) -> None:
    """Plot end-effector position + orientation over time.

    Shows the controller-reported actual EE pose (``cart_actual_*``) and, when
    present, the commanded EE pose (``cart_cmd_*``). With ``compare_path`` a
    second run is overlaid so two replay methods (e.g. joint vs cartesian
    output) can be compared in task space. Both runs' actual poses come from the
    controller's own ``/cart_states`` so they share a frame and are directly
    comparable.

    Run labels: explicit ``labels`` win; otherwise each run is auto-named
    ``cartesian`` (has ``cart_cmd_*``) or ``joint`` from its columns.
    """
    paths = [csv_path] + ([compare_path] if compare_path else [])
    loaded = []
    for r_idx, path in enumerate(paths):
        print(f"Loading EE data from: {path}")
        df = pd.read_csv(path)
        if not _ee_columns_present(df, 'cart_actual'):
            print(f"Warning: {path} has no cart_actual_* columns; "
                  "re-run with record_data:=true to log the EE pose. Skipping.")
            continue
        if labels and r_idx < len(labels):
            lab = labels[r_idx]
        else:
            lab = _auto_label(df, fallback=f'run {r_idx + 1}')
        loaded.append((df, lab))
    if not loaded:
        print("Error: no run has end-effector pose columns; nothing to plot.")
        return
    if not _HAVE_SCIPY:
        print("Warning: scipy not available; orientation (euler) omitted.")

    pos_labels = ['X (m)', 'Y (m)', 'Z (m)']
    rot_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    # Distinct colours per run; actual solid, command dashed.
    run_colors = ['tab:blue', 'tab:red', 'tab:green']
    for r_idx, (df, lab) in enumerate(loaded):
        t = df['step'].values
        c = run_colors[r_idx % len(run_colors)]
        a_pos, a_eul = _extract_ee(df, 'cart_actual')
        c_pos, c_eul = _extract_ee(df, 'cart_cmd')
        for k in range(3):
            axes[0, k].plot(t, a_pos[:, k], color=c, linewidth=1.4,
                            label=f'{lab} actual', alpha=0.85)
            if c_pos is not None:
                axes[0, k].plot(t, c_pos[:, k], color=c, linewidth=1.0,
                                linestyle='--', alpha=0.7,
                                label=f'{lab} command')
            axes[1, k].plot(t, a_eul[:, k], color=c, linewidth=1.4,
                            label=f'{lab} actual', alpha=0.85)
            if c_eul is not None:
                axes[1, k].plot(t, c_eul[:, k], color=c, linewidth=1.0,
                                linestyle='--', alpha=0.7,
                                label=f'{lab} command')

    for k in range(3):
        axes[0, k].set_ylabel(pos_labels[k])
        axes[0, k].set_title(f'EE Position {pos_labels[k][0]}')
        axes[0, k].grid(True, alpha=0.3)
        axes[0, k].legend(loc='best', fontsize=8)
        axes[1, k].set_ylabel(rot_labels[k])
        axes[1, k].set_title(f'EE Orientation {rot_labels[k].split()[0]}')
        axes[1, k].set_xlabel('Step')
        axes[1, k].grid(True, alpha=0.3)
        axes[1, k].legend(loc='best', fontsize=8)

    title = 'End-effector pose'
    if len(loaded) > 1:
        title += f' comparison: {loaded[0][1]} vs {loaded[1][1]}'
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        print(f"Saving EE plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("EE plot saved successfully")
    else:
        print("Displaying EE plot (close window to continue)...")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot comparison of actual vs theoretical right-arm trajectory')
    parser.add_argument('csv_path', help='Path to replay_analysis_*.csv file')
    parser.add_argument('-o', '--output', help='Output path for the plot (PNG/PDF)')
    parser.add_argument('--error-plot', help='Output path for error analysis plot')
    parser.add_argument('--ee-plot',
                        help='Output path for the end-effector pose plot '
                             '(position + orientation over time)')
    parser.add_argument('--compare',
                        help='Second replay_analysis_*.csv to overlay on the '
                             'EE plot (e.g. a joint-mode run to compare against '
                             'a cartesian-mode run)')
    parser.add_argument('--labels', nargs='+',
                        help='Override legend labels for the EE plot runs, '
                             'e.g. --labels cartesian joint. If omitted, each '
                             'run is auto-named "cartesian" (has cart_cmd_*) '
                             'or "joint" from its columns.')
    parser.add_argument('-j', '--joints', type=int, nargs='+',
                       help='Joint indices to plot (0-6), default: all joints')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively (default if no output path)')

    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        return 1
    if args.compare and not os.path.isfile(args.compare):
        print(f"Error: --compare file not found: {args.compare}")
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

    # Generate end-effector pose plot if requested (or whenever a comparison
    # run is given, since that is its main purpose).
    if args.ee_plot or args.compare or args.show:
        ee_output = args.ee_plot if args.ee_plot and not args.show else None
        plot_ee_comparison(args.csv_path, ee_output, args.compare, args.labels)

    return 0


if __name__ == '__main__':
    sys.exit(main())
