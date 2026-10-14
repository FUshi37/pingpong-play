#!/usr/bin/env python3
import os
import argparse
import tempfile

import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.lib.utils import ImageReader


def mm_to_pt(mm):
    return mm * 72.0 / 25.4


def get_apriltag_dict(family: str):
    family = family.lower()

    if family in ["tag36h11", "t36h11"]:
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    if family in ["tag25h9", "t25h9"]:
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)

    if family in ["tag16h5", "t16h5"]:
        return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

    raise ValueError(f"Unsupported AprilTag family: {family}")


def generate_marker(dictionary, tag_id, marker_size_px):
    marker = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)

    if hasattr(cv2.aruco, "generateImageMarker"):
        cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size_px, marker, 1)
    else:
        marker = cv2.aruco.drawMarker(dictionary, tag_id, marker_size_px)

    # OpenCV marker is already black/white uint8.
    return marker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6, help="Number of tag rows")
    parser.add_argument("--cols", type=int, default=6, help="Number of tag cols")
    parser.add_argument("--tag-size-mm", type=float, default=40.0, help="Black tag size in mm")
    parser.add_argument("--spacing", type=float, default=0.3, help="White gap / tag size")
    parser.add_argument("--family", type=str, default="tag36h11", help="AprilTag family")
    parser.add_argument("--paper", type=str, default="A3", choices=["A3", "A4"], help="Paper size")
    parser.add_argument("--output", type=str, default="aprilgrid_6x6_40mm.pdf")
    args = parser.parse_args()

    dictionary = get_apriltag_dict(args.family)

    if args.paper == "A3":
        page_size = landscape(A3)
    else:
        page_size = landscape(A4)

    page_w_pt, page_h_pt = page_size

    tag_size_pt = mm_to_pt(args.tag_size_mm)
    gap_size_pt = tag_size_pt * args.spacing

    grid_w_pt = args.cols * tag_size_pt + (args.cols - 1) * gap_size_pt
    grid_h_pt = args.rows * tag_size_pt + (args.rows - 1) * gap_size_pt

    if grid_w_pt > page_w_pt or grid_h_pt > page_h_pt:
        print("WARNING: Grid is larger than selected paper.")
        print(f"Grid size: {grid_w_pt / 72.0 * 25.4:.1f} mm × {grid_h_pt / 72.0 * 25.4:.1f} mm")
        print(f"Paper size: {page_w_pt / 72.0 * 25.4:.1f} mm × {page_h_pt / 72.0 * 25.4:.1f} mm")
        print("Use A3 or reduce --tag-size-mm.")

    margin_x = (page_w_pt - grid_w_pt) / 2.0
    margin_y = (page_h_pt - grid_h_pt) / 2.0

    output_path = os.path.abspath(args.output)

    c = canvas.Canvas(output_path, pagesize=page_size)

    # Optional title
    title = (
        f"AprilGrid {args.cols}x{args.rows}, "
        f"{args.family}, tagSize={args.tag_size_mm:.1f}mm, spacing={args.spacing}"
    )
    c.setFont("Helvetica", 10)
    c.drawString(mm_to_pt(10), page_h_pt - mm_to_pt(10), title)

    marker_px = 800

    with tempfile.TemporaryDirectory() as tmpdir:
        for r in range(args.rows):
            for col in range(args.cols):
                tag_id = r * args.cols + col

                marker = generate_marker(dictionary, tag_id, marker_px)
                img = Image.fromarray(marker)

                img_path = os.path.join(tmpdir, f"tag_{tag_id}.png")
                img.save(img_path)

                x = margin_x + col * (tag_size_pt + gap_size_pt)
                # PDF origin is bottom-left, so invert row order visually.
                y = margin_y + (args.rows - 1 - r) * (tag_size_pt + gap_size_pt)

                c.drawImage(
                    ImageReader(img_path),
                    x,
                    y,
                    width=tag_size_pt,
                    height=tag_size_pt,
                    mask="auto",
                )

                # Tiny ID label, outside tag area, optional.
                c.setFont("Helvetica", 5)
                c.drawString(x, y - mm_to_pt(2.5), str(tag_id))

    c.showPage()
    c.save()

    print(f"Saved PDF: {output_path}")
    print("")
    print("Corresponding Kalibr-style YAML:")
    print("target_type: aprilgrid")
    print(f"tagCols: {args.cols}")
    print(f"tagRows: {args.rows}")
    print(f"tagSize: {args.tag_size_mm / 1000.0:.6f}")
    print(f"tagSpacing: {args.spacing}")
    print(f"tagFamily: {args.family}")


if __name__ == "__main__":
    main()