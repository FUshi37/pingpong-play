#!/usr/bin/env python3
"""Unit tests for vectorized RMP/MJX replay helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np

try:
    import jax.numpy as jnp
    _JAX_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - fixture-gated optional dep
    jnp = None
    _JAX_IMPORT_ERROR = exc

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
RL_SIM_DIR = REPO_ROOT / "simcode5" / "pingpong_moz1-main" / "pingpong_controller" / "tools" / "rl_sim"
CORE_PATH = RL_SIM_DIR / "rmp_mjx_replay_batch_core.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))


def _load_core() -> ModuleType:
    spec = importlib.util.spec_from_file_location("rmp_mjx_replay_batch_core", CORE_PATH)
    if spec is None or spec.loader is None:
        raise unittest.SkipTest("cannot load rmp_mjx_replay_batch_core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = None if _JAX_IMPORT_ERROR is not None else _load_core()
if _JAX_IMPORT_ERROR is None:
    from recovered_rmp.replay import RecordingTrace  # noqa: E402
else:  # pragma: no cover
    RecordingTrace = None


class RmpMjxReplayBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _JAX_IMPORT_ERROR is not None:
            raise unittest.SkipTest("JAX is required for RMP/MJX replay batch helper tests")
    def _record(self, name: str, length: int, offset: float = 0.0):
        target = np.arange(length * 7, dtype=np.float32).reshape(length, 7) + offset
        trace = RecordingTrace(
            source=Path(f"/tmp/{name}/control_samples.jsonl"),
            time=np.arange(length, dtype=float) * 0.005,
            source_sequence=np.arange(length, dtype=np.int64),
            joint_time=np.arange(length, dtype=float) * 0.005,
            target_q=target,
            actual_q=target + 0.1,
            actual_qd=np.ones((length, 7), dtype=np.float32) * (offset + 1.0),
            actual_qdd=np.ones((length, 7), dtype=np.float32) * (offset + 2.0),
            policy_mask=np.ones(length, dtype=bool),
            median_dt=0.005,
        )
        return core.PreparedReplay(
            group="record_new3",
            recording_path=Path(f"/tmp/{name}"),
            recording_name=name,
            trace=trace,
            phase_info={"valid_phase_only": True},
            sample_count=length,
        )

    def test_trace_packing_handles_unequal_lengths(self):
        records = [self._record("a", 3, 0.0), self._record("b", 5, 100.0)]
        batch = core.build_replay_batch(records)
        self.assertEqual(tuple(batch.targets.shape), (5, 2, 7))
        np.testing.assert_array_equal(batch.lengths, np.asarray([3, 5], dtype=np.int32))
        active = np.asarray(batch.active_mask)
        np.testing.assert_array_equal(active[:, 0], [True, True, True, False, False])
        np.testing.assert_array_equal(active[:, 1], [True, True, True, True, True])
        targets = np.asarray(batch.targets)
        np.testing.assert_allclose(targets[:3, 0], records[0].trace.target_q)
        np.testing.assert_allclose(targets[3:, 0], np.tile(records[0].trace.target_q[-1], (2, 1)))
        np.testing.assert_allclose(np.asarray(batch.initial_actual_q)[1], records[1].trace.actual_q[0])

    def test_select_active_lanes_freezes_inactive_batched_leaves(self):
        old = {
            "a": jnp.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
            "b": jnp.asarray([10.0, 20.0, 30.0]),
            "c": "metadata",
        }
        new = {
            "a": jnp.asarray([[11.0, 11.0], [22.0, 22.0], [33.0, 33.0]]),
            "b": jnp.asarray([110.0, 220.0, 330.0]),
            "c": "new_metadata",
        }
        selected = core.select_active_lanes(old, new, jnp.asarray([True, False, True]), 3)
        np.testing.assert_allclose(np.asarray(selected["a"]), [[11.0, 11.0], [2.0, 2.0], [33.0, 33.0]])
        np.testing.assert_allclose(np.asarray(selected["b"]), [110.0, 20.0, 330.0])
        self.assertEqual(selected["c"], "new_metadata")

    def test_markdown_renderer_includes_core_sections(self):
        summary = core.summarize_records(
            model=Path("moz1_pd.xml"),
            record_root=Path("records"),
            include_groups=["record_new3"],
            calibration_profile="recovered_yaml_pd_fit",
            output_root=Path("out"),
            records=[{
                "group": "record_new3",
                "recording_name": "probe_recording_a",
                "status": "pass",
                "batch_index": 0,
                "environment_index": 0,
                "active_samples": 3,
                "position_rms_rad": 0.1,
                "velocity_rms_rad_s": 0.2,
                "acceleration_rms_rad_s2": 0.3,
                "maximum_position_error_rad": 0.4,
                "racket_metrics": {"available": True},
            }],
            discovered_count=1,
            loaded_count=1,
            batch_size_requested=1,
            vectorized_batch_count=1,
            max_padded_samples=3,
            total_active_samples=3,
            device_seconds=0.01,
            wall_seconds=0.02,
            env_dt=0.001,
        )
        markdown = core.render_markdown_report(summary)
        self.assertIn("# Vectorized MJX RMP Replay Validation", markdown)
        self.assertIn("## Group metrics", markdown)
        self.assertIn("probe_recording_a", markdown)
        self.assertIn("All passing reports include racket center", markdown)


if __name__ == "__main__":
    unittest.main()
