#!/usr/bin/env python3
"""
Shift z coordinates in real_ball_replay_untransformed.csv by +0.3m (+300mm)
to test policy behavior with higher ball positions.
"""

import csv

input_file = "src/pingpong_moz1/pingpong_controller/outputs/rl_sim/real_ball_replay_untransformed.csv"
output_file = "src/pingpong_moz1/pingpong_controller/outputs/rl_sim/real_ball_replay_untransformed_z_plus_0.3.csv"

z_shift_mm = 300.0  # +0.3m in millimeters

with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # Read and write header
    header = next(reader)
    writer.writerow(header)

    # Find column indices
    z_mm_idx = header.index('z_mm')
    raw_z_mm_idx = header.index('raw_z_mm')

    print(f"Found z_mm at column {z_mm_idx}, raw_z_mm at column {raw_z_mm_idx}")
    print(f"Shifting z coordinates by +{z_shift_mm}mm (+0.3m)")

    # Process data rows
    row_count = 0
    for row in reader:
        # Shift z_mm
        row[z_mm_idx] = str(float(row[z_mm_idx]) + z_shift_mm)

        # Shift raw_z_mm
        row[raw_z_mm_idx] = str(float(row[raw_z_mm_idx]) + z_shift_mm)

        writer.writerow(row)
        row_count += 1

    print(f"Processed {row_count} rows")
    print(f"Output saved to: {output_file}")
