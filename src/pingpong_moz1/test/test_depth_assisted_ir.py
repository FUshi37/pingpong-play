"""Independent stereo support must recover faint balls and reject distractors."""

import cv2
import numpy as np
from pingpong_controller.depth_assisted_ir import (
    depth_supported_candidates,
    validate_ir_depth,
    range_quality,
)


def scene():
    ir = np.zeros((120, 160), np.uint8)
    depth = np.full(ir.shape, 2000, np.uint16)
    cv2.circle(ir, (80, 60), 10, 120, -1)
    cv2.circle(depth, (80, 60), 10, 500, -1)
    return ir, depth


OPTIONS = dict(depth_scale=0.001, fx=250.0, fy=250.0)


def test_faint_compact_stereo_surface_has_correct_ir_center():
    ir, depth = scene()
    found = depth_supported_candidates(ir, depth, **OPTIONS)
    assert len(found) == 1
    np.testing.assert_allclose([found[0]["cx"], found[0]["cy"]], [80, 60], atol=0.5)
    assert not found[0]["geometry_quality_valid"]
    # Extracted stereo radius must not be treated as independent IR evidence.
    assert found[0]["depth_support_pixels"] > 200


def test_bright_far_background_and_near_dark_surface_are_rejected():
    ir, depth = scene()
    assert not depth_supported_candidates(ir, np.full_like(depth, 2000), **OPTIONS)
    assert not depth_supported_candidates(np.zeros_like(ir), depth, **OPTIONS)


def test_sparse_or_disagreeing_depth_cannot_validate_a_loose_ir_candidate():
    _, depth = scene()
    candidate = dict(cx=80.0, cy=60.0, radius=10.0)
    assert validate_ir_depth(candidate, depth, **OPTIONS) is not None
    sparse = np.zeros_like(depth)
    sparse[59:62, 79:82] = 500
    assert validate_ir_depth(candidate, sparse, **OPTIONS) is None
    assert validate_ir_depth(dict(candidate, radius=30.0), depth, **OPTIONS) is None
    depth[60:70, 70:91] = 700
    assert validate_ir_depth(candidate, depth, **OPTIONS) is None


def test_clipped_surface_and_merged_large_object_are_rejected():
    ir, depth = scene()
    assert not depth_supported_candidates(ir, depth, roi=(75, 40, 100, 80), **OPTIONS)
    cv2.rectangle(depth, (20, 25), (140, 95), 500, -1)
    assert not depth_supported_candidates(ir, depth, **OPTIONS)


def test_primary_range_quality_rejects_stable_but_sparse_depth():
    _, depth = scene()
    candidate = dict(cx=80., cy=60., radius=10.)
    good = range_quality(candidate, depth, depth_scale=.001, selected_depth_mm=500.)
    assert good['range_valid'] and good['range_quality_strong']
    sparse = np.zeros_like(depth)
    sparse[58:63, 77:84] = 380
    bad = range_quality(candidate, sparse, depth_scale=.001, selected_depth_mm=380.)
    assert not bad['range_valid']
    assert bad['range_quality_reason'] == 'sparse_stereo'


def test_range_quality_rejects_background_selected_over_a_dense_ball():
    _, depth = scene()
    result = range_quality(dict(cx=80., cy=60., radius=10.), depth,
                           depth_scale=.001, selected_depth_mm=800.)
    assert not result['range_valid']
    assert result['range_quality_reason'] == 'sample_outside_surface'
