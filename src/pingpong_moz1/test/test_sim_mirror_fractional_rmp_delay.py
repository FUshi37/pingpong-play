"""Regression tests for mirror-only fractional recovered-RMP delays."""

from __future__ import annotations

import numpy as np

from pingpong_controller.sim_mirror import RealBallSimMirror


def _delay_once(delays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mirror = RealBallSimMirror.__new__(RealBallSimMirror)
    mirror.rmp_output_delay_steps = np.asarray(delays, dtype=np.float64)
    buffer = np.arange(21, dtype=np.float64).reshape(3, 7)
    value = np.arange(100, 107, dtype=np.float64)
    return mirror._push_rmp_output_delay(buffer, value)


def test_integer_delay_keeps_legacy_sample_selection() -> None:
    delays = np.asarray([0, 1, 2, 3, 0, 1, 2], dtype=np.float64)
    shifted, delayed = _delay_once(delays)
    combined = np.concatenate((
        np.arange(21, dtype=np.float64).reshape(3, 7),
        np.arange(100, 107, dtype=np.float64)[None, :]), axis=0)
    legacy = combined[
        combined.shape[0] - 1 - delays.astype(np.int64), np.arange(7)]
    np.testing.assert_array_equal(shifted, combined[1:])
    np.testing.assert_array_equal(delayed, legacy)


def test_fractional_delay_interpolates_adjacent_one_ms_outputs() -> None:
    delays = np.asarray([0.5, 1.5, 2.5, 0.25, 1.25, 2.25, 0.75])
    _shifted, delayed = _delay_once(delays)
    combined = np.concatenate((
        np.arange(21, dtype=np.float64).reshape(3, 7),
        np.arange(100, 107, dtype=np.float64)[None, :]), axis=0)
    floor = np.floor(delays).astype(np.int64)
    fraction = delays - floor
    newer = combined.shape[0] - 1 - floor
    older = newer - 1
    expected = (
        (1.0 - fraction) * combined[newer, np.arange(7)]
        + fraction * combined[older, np.arange(7)])
    np.testing.assert_allclose(delayed, expected, rtol=0.0, atol=0.0)
