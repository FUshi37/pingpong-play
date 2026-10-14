#!/usr/bin/env python3
"""
测试脚本：验证 CalibrationData 加载和坐标变换的正确性
"""

import sys
import argparse
from pathlib import Path
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.calibration_loader import load_calibration


def test_calibration_loading(json_path):
    """测试 JSON 文件加载"""
    print("=" * 60)
    print("测试 1: CalibrationData 加载")
    print("=" * 60)

    json_path = Path(json_path).expanduser().resolve()
    if not json_path.exists():
        print(f"❌ JSON 文件不存在: {json_path}")
        return False

    try:
        calib = load_calibration(str(json_path))
        print(f"✓ JSON 文件加载成功")
        print(f"\n{calib}")

        # 验证维度
        assert calib.camera_matrix.shape == (3, 3), "camera_matrix 维度错误"
        assert calib.original_camera_matrix.shape == (3, 3), "original_camera_matrix 维度错误"
        assert calib.T_base_camera.shape == (4, 4), "T_base_camera 维度错误"
        assert len(calib.dist_coeffs) >= 5, "dist_coeffs 长度错误"
        assert len(calib.original_dist_coeffs) >= 5, "original_dist_coeffs 长度错误"
        print("✓ 所有矩阵维度正确")

        # 验证 T_base_camera 是有效的变换矩阵
        bottom_row = calib.T_base_camera[3, :]
        assert np.allclose(bottom_row, [0, 0, 0, 1]), "T_base_camera 底行应为 [0,0,0,1]"
        print("✓ T_base_camera 是有效的齐次变换矩阵")

        return True, calib
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_coordinate_transformation(calib):
    """测试坐标变换方向和单位转换"""
    print("\n" + "=" * 60)
    print("测试 2: 坐标变换方向和单位转换验证")
    print("=" * 60)

    # 测试点：camera 坐标系原点 (0, 0, 0) mm
    point_camera = np.array([0.0, 0.0, 0.0])
    point_base = calib.transform_camera_to_base(point_camera)

    print(f"\n相机原点在 camera 坐标系: {point_camera} mm")
    print(f"相机原点在 base_link 坐标系: {point_base} mm")
    print(f"  (应该接近 T_base_camera 的平移部分 * 1000)")

    expected_translation_mm = calib.T_base_camera[:3, 3] * 1000.0
    print(f"T_base_camera 平移向量 (转换为 mm): {expected_translation_mm}")

    if np.allclose(point_base, expected_translation_mm, atol=1e-3):
        print("✓ 坐标变换方向和单位转换正确")
    else:
        print("❌ 坐标变换方向或单位转换可能错误")
        print(f"  差异: {point_base - expected_translation_mm}")
        return False

    # 测试点：camera 坐标系下的一个点 (0, 0, 1000) mm
    point_camera_2 = np.array([0.0, 0.0, 1000.0])
    point_base_2 = calib.transform_camera_to_base(point_camera_2)

    print(f"\n测试点在 camera 坐标系: {point_camera_2} mm")
    print(f"测试点在 base_link 坐标系: {point_base_2} mm")

    # 手动验证（正确处理单位）
    point_camera_m = point_camera_2 / 1000.0  # mm -> m
    point_homo = np.append(point_camera_m, 1.0)
    point_base_homo = calib.T_base_camera @ point_homo
    point_base_manual = point_base_homo[:3] * 1000.0  # m -> mm

    print(f"手动计算 (mm -> m -> 变换 -> mm): {point_base_manual}")

    if np.allclose(point_base_2, point_base_manual, atol=1e-3):
        print("✓ 变换结果与手动计算一致（单位转换正确）")
    else:
        print("❌ 变换结果与手动计算不一致")
        print(f"  差异: {point_base_2 - point_base_manual}")
        return False

    # 验证 T_base_camera 平移单位确实是 m
    translation_z = calib.T_base_camera[2, 3]
    print(f"\nT_base_camera[2,3] (z 平移): {translation_z:.6f}")
    if 1.0 < translation_z < 2.0:
        print("✓ T_base_camera 平移单位确认为 m（z 约 1.4m 符合预期）")
    else:
        print("⚠ T_base_camera 平移单位可能不是 m")

    return True


def test_intrinsics_usage(calib):
    """测试内参使用"""
    print("\n" + "=" * 60)
    print("测试 3: 内参使用验证")
    print("=" * 60)

    fx_orig, fy_orig, cx_orig, cy_orig = calib.get_original_intrinsics()
    fx_undist, fy_undist, cx_undist, cy_undist = calib.get_undistorted_intrinsics()

    print(f"\n原始内参 (用于 undistort 输入):")
    print(f"  fx={fx_orig:.2f}, fy={fy_orig:.2f}, cx={cx_orig:.2f}, cy={cy_orig:.2f}")

    print(f"\n去畸变内参 (用于 3D 计算):")
    print(f"  fx={fx_undist:.2f}, fy={fy_undist:.2f}, cx={cx_undist:.2f}, cy={cy_undist:.2f}")

    print(f"\n畸变系数 (原始): {calib.original_dist_coeffs[:5]}")
    print(f"畸变系数 (去畸变后): {calib.dist_coeffs[:5]}")

    # 验证去畸变后的畸变系数应该接近零
    if np.allclose(calib.dist_coeffs, 0, atol=1e-6):
        print("✓ 去畸变后的畸变系数为零（符合预期）")
    else:
        print("⚠ 去畸变后的畸变系数不为零（可能不符合预期）")

    return True


def test_backprojection_example(calib):
    """测试反投影示例"""
    print("\n" + "=" * 60)
    print("测试 4: 反投影示例")
    print("=" * 60)

    # 假设球在图像中心，半径 20 像素
    fx, fy, cx, cy = calib.get_undistorted_intrinsics()

    # 图像中心像素
    pixel_x = cx
    pixel_y = cy

    # 球直径 40mm，半径 20 像素 -> 深度估计
    ball_diameter_mm = 40.0
    radius_px = 20.0
    f_avg = (fx + fy) / 2.0
    depth_mm = (f_avg * ball_diameter_mm) / (2.0 * radius_px)

    print(f"\n假设球在图像中心:")
    print(f"  像素位置: ({pixel_x:.1f}, {pixel_y:.1f})")
    print(f"  半径: {radius_px:.1f} px")
    print(f"  估计深度: {depth_mm:.1f} mm")

    # 反投影到 camera 坐标系
    Z = depth_mm
    X = (pixel_x - cx) * Z / fx
    Y = (pixel_y - cy) * Z / fy
    pos_camera = np.array([X, Y, Z])

    print(f"\nCamera 坐标系位置: ({X:.1f}, {Y:.1f}, {Z:.1f}) mm")

    # 转换到 base_link
    pos_base = calib.transform_camera_to_base(pos_camera)
    print(f"Base_link 坐标系位置: ({pos_base[0]:.1f}, {pos_base[1]:.1f}, {pos_base[2]:.1f}) mm")

    print("\n✓ 反投影流程演示完成")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证 eye-to-hand 标定结果 JSON 的内参和坐标变换。"
    )
    parser.add_argument(
        "calibration_json",
        help="Path to eye_hand_result_*.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("CalibrationData 自检测试")
    print("=" * 60)
    print(f"标定文件: {Path(args.calibration_json).expanduser().resolve()}")

    success, calib = test_calibration_loading(args.calibration_json)
    if not success:
        print("\n❌ 测试失败：无法加载 CalibrationData")
        return 1

    if not test_coordinate_transformation(calib):
        print("\n❌ 测试失败：坐标变换验证失败")
        return 1

    if not test_intrinsics_usage(calib):
        print("\n❌ 测试失败：内参验证失败")
        return 1

    if not test_backprojection_example(calib):
        print("\n❌ 测试失败：反投影示例失败")
        return 1

    print("\n" + "=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
