#!/usr/bin/env python3
"""
Generate detailed comparison plots for all observation dimensions.
Compares policy obs trace (real-robot simulation) with validation obs trace (MJX reference).
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Observation dimension names (67 total for delay conditioning mode)
OBS_NAMES = [
    # Base obs (50 dims): 0-49
    # Joint positions (7)
    "arm_q[0]", "arm_q[1]", "arm_q[2]", "arm_q[3]", "arm_q[4]", "arm_q[5]", "arm_q[6]",
    # Joint velocities (7)
    "arm_dq[0]", "arm_dq[1]", "arm_dq[2]", "arm_dq[3]", "arm_dq[4]", "arm_dq[5]", "arm_dq[6]",
    # Base pose (3)
    "base_q_x", "base_q_y", "base_q_yaw",
    # Base velocity (3)
    "base_dq_x", "base_dq_y", "base_dq_yaw",
    # Ball position in base frame (3)
    "ball_pos_x", "ball_pos_y", "ball_pos_z",
    # Ball velocity in base frame (3)
    "ball_vel_x", "ball_vel_y", "ball_vel_z",
    # Racket position in base frame (3)
    "racket_pos_x", "racket_pos_y", "racket_pos_z",
    # Racket velocity in base frame (3)
    "racket_vel_x", "racket_vel_y", "racket_vel_z",
    # Relative position (ball - racket) in base frame (3)
    "rel_x", "rel_y", "rel_z",
    # Previous action (7)
    "prev_action[0]", "prev_action[1]", "prev_action[2]", "prev_action[3]",
    "prev_action[4]", "prev_action[5]", "prev_action[6]",
    # Arm command error (7)
    "arm_cmd_error[0]", "arm_cmd_error[1]", "arm_cmd_error[2]", "arm_cmd_error[3]",
    "arm_cmd_error[4]", "arm_cmd_error[5]", "arm_cmd_error[6]",
    # Ball obs age (1)
    "ball_age_norm",

    # Delay conditioning features (17 dims): 50-66
    # Delay features
    "tau_act_norm",
    # Arm command velocity (7)
    "arm_cmd_qvel[0]", "arm_cmd_qvel[1]", "arm_cmd_qvel[2]", "arm_cmd_qvel[3]",
    "arm_cmd_qvel[4]", "arm_cmd_qvel[5]", "arm_cmd_qvel[6]",
    # Active command error (7)
    "active_cmd_error[0]", "active_cmd_error[1]", "active_cmd_error[2]", "active_cmd_error[3]",
    "active_cmd_error[4]", "active_cmd_error[5]", "active_cmd_error[6]",
    # Phase features (2)
    "t_contact_est", "t_margin",
]


def plot_obs_dimension(recorded_data, reference_data, dim_idx, obs_name, output_path):
    """Plot comparison for a single observation dimension."""
    recorded_col = f"obs/{dim_idx:03d}"
    reference_col = f"obs/{dim_idx:03d}"

    if recorded_col not in recorded_data.columns or reference_col not in reference_data.columns:
        print(f"Warning: obs/{dim_idx:03d} not found in traces")
        return

    recorded_vals = recorded_data[recorded_col].values
    reference_vals = reference_data[reference_col].values

    n = min(len(recorded_vals), len(reference_vals))
    recorded_vals = recorded_vals[:n]
    reference_vals = reference_vals[:n]

    # Calculate statistics
    diff = recorded_vals - reference_vals
    rms_diff = np.sqrt(np.mean(diff**2))
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(diff)

    # Create figure
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top: overlay plot
    time = np.arange(n) * 0.005  # 5ms timestep
    axes[0].plot(time, recorded_vals, 'b-', label='Real-robot', linewidth=1.5, alpha=0.8)
    axes[0].plot(time, reference_vals, 'r--', label='MJX reference', linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'obs[{dim_idx}]: {obs_name}', fontsize=14, fontweight='bold')

    # Bottom: difference plot
    axes[1].plot(time, diff, 'g-', linewidth=1.0)
    axes[1].axhline(y=0, color='k', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Difference\n(Real - Ref)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = (
        f"RMS difference: {rms_diff:.6f}\n"
        f"Max |diff|: {max_diff:.6f}\n"
        f"Mean diff: {mean_diff:.6f}\n"
        f"\n"
        f"Real-robot:\n"
        f"  mean: {np.mean(recorded_vals):.6f}\n"
        f"  std: {np.std(recorded_vals):.6f}\n"
        f"  range: [{np.min(recorded_vals):.6f}, {np.max(recorded_vals):.6f}]\n"
        f"\n"
        f"MJX reference:\n"
        f"  mean: {np.mean(reference_vals):.6f}\n"
        f"  std: {np.std(reference_vals):.6f}\n"
        f"  range: [{np.min(reference_vals):.6f}, {np.max(reference_vals):.6f}]"
    )

    axes[1].text(
        0.98, 0.02, stats_text,
        transform=axes[1].transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace'
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def generate_index_html(output_dir, num_dims):
    """Generate HTML index for easy viewing of all plots."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Observation Comparison - All Dimensions</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .card img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            cursor: pointer;
        }
        .card h3 {
            margin: 10px 0 5px 0;
            font-size: 14px;
            color: #555;
        }
        .important {
            border: 3px solid #ff9800;
        }
    </style>
</head>
<body>
    <h1>Observation Comparison: Real-robot vs MJX Reference</h1>
    <p>Click on any image to view full size. Orange border indicates important dimensions.</p>
    <div class="grid">
"""

    # Important dimensions to highlight
    important_dims = [29, 30, 31, 65, 66]  # racket_vel and t_contact

    for i in range(num_dims):
        obs_name = OBS_NAMES[i] if i < len(OBS_NAMES) else f"obs_{i}"
        card_class = "card important" if i in important_dims else "card"
        html_content += f"""
        <div class="{card_class}">
            <h3>obs[{i:03d}]: {obs_name}</h3>
            <a href="obs_{i:03d}.png" target="_blank">
                <img src="obs_{i:03d}.png" alt="obs {i}">
            </a>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    index_path = output_dir / "index.html"
    with open(index_path, 'w') as f:
        f.write(html_content)

    print(f"Generated index: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate detailed obs comparison plots")
    parser.add_argument("recorded_trace", type=str, help="Policy obs trace CSV (real-robot)")
    parser.add_argument("reference_trace", type=str, help="Validation obs trace CSV (MJX reference)")
    parser.add_argument("--output-dir", type=str, default="obscomparison", help="Output directory")
    parser.add_argument("--num-dims", type=int, default=67, help="Number of obs dimensions")

    args = parser.parse_args()

    # Load data
    print(f"Loading recorded trace: {args.recorded_trace}")
    recorded_data = pd.read_csv(args.recorded_trace)

    print(f"Loading reference trace: {args.reference_trace}")
    reference_data = pd.read_csv(args.reference_trace)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {args.num_dims} comparison plots...")

    # Generate plots for each dimension
    for i in range(args.num_dims):
        obs_name = OBS_NAMES[i] if i < len(OBS_NAMES) else f"obs_{i}"
        output_path = output_dir / f"obs_{i:03d}.png"

        plot_obs_dimension(recorded_data, reference_data, i, obs_name, output_path)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{args.num_dims} plots")

    print(f"\nAll plots saved to: {output_dir}")

    # Generate HTML index
    generate_index_html(output_dir, args.num_dims)

    print("\nDone!")
    print(f"View results: firefox {output_dir}/index.html")


if __name__ == "__main__":
    main()
