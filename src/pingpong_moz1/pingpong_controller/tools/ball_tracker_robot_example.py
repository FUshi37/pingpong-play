#!/usr/bin/env python3
"""Real-robot integration example for the IR Ball State Estimator.

The bridge is ROS-independent so it can be called from the existing camera
callback and 200 Hz controller without introducing another vision module.
YOLO remains only an ROI gate; only the IR-derived 3-D position is fused.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from pingpong_controller.ball_tracker import (
    BallEstimate,
    BallObservationConfig,
    BallStateEstimator,
    build_rl_observation,
)
from pingpong_controller.ball_tracker.experiment_logger import (
    BallExperimentLogger,
)


class RealtimeBallEstimatorBridge:
    """Convert detector millimetres to the estimator's SI-unit API."""

    def __init__(
            self,
            config_path: Optional[str | Path] = None,
            *,
            experiment_log_path: Optional[str | Path] = None,
    ) -> None:
        self.estimator = BallStateEstimator(config_path)
        self.logger = (
            None if experiment_log_path is None
            else BallExperimentLogger(experiment_log_path))

    def on_ir_measurement(
            self,
            position_mm: Any,
            image_timestamp_s: float,
            racket_pose_mm: Any,
            *,
            measurement_covariance_mm2: Optional[np.ndarray] = None,
    ) -> BallEstimate:
        """Fuse one genuine IR 3-D measurement at its image timestamp.

        Call this only when the IR blob reconstruction succeeds. A YOLO box
        must never be substituted as a ball-state measurement.
        """
        position_m = np.asarray(position_mm, dtype=float).reshape(3) * 1e-3
        covariance_m2 = None
        if measurement_covariance_mm2 is not None:
            covariance_m2 = (
                np.asarray(
                    measurement_covariance_mm2, dtype=float).reshape(3, 3)
                * 1e-6)
        estimate = self.estimator.update(
            position_m,
            image_timestamp_s,
            racket_pose=self._racket_pose_to_metres(racket_pose_mm),
            measurement_covariance=covariance_m2,
        )
        self._log(estimate, measurement=position_m)
        return estimate

    def on_control_tick(
            self,
            control_timestamp_s: float,
            racket_pose_mm: Any,
            robot_q: Any,
            robot_dq: Any,
            *,
            include_uncertainty: bool = False,
    ) -> dict[str, Any]:
        """Predict at 200 Hz and construct the enlarged RL observation."""
        estimate = self.estimator.predict_to(
            control_timestamp_s,
            racket_pose=self._racket_pose_to_metres(racket_pose_mm),
        )
        observation = build_rl_observation(
            robot_q,
            robot_dq,
            estimate,
            BallObservationConfig(
                include_uncertainty=include_uncertainty,
                include_validity=True,
            ),
        )
        self._log(estimate)
        result = estimate.as_dict()
        result['rl_observation'] = observation
        # These convenience fields help an existing mm-based ROS node migrate
        # without accidentally double-converting the SI estimator state.
        result['position_mm'] = (estimate.position * 1e3).tolist()
        result['velocity_mm_s'] = (estimate.velocity * 1e3).tolist()
        result['acceleration_mm_s2'] = (
            estimate.acceleration * 1e3).tolist()
        return result

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()

    def _log(
            self, estimate: BallEstimate, *, measurement: Any = None) -> None:
        if self.logger is not None:
            self.logger.log(estimate, measurement=measurement)

    @staticmethod
    def _racket_pose_to_metres(racket_pose_mm: Any) -> Any:
        """Convert pose translation/velocity while preserving orientation."""
        if isinstance(racket_pose_mm, Mapping):
            pose = dict(racket_pose_mm)
            if 'position' in pose:
                pose['position'] = (
                    np.asarray(pose['position'], dtype=float) * 1e-3).tolist()
            elif 'z_racket' in pose:
                pose['z_racket'] = float(pose['z_racket']) * 1e-3
            elif 'z' in pose:
                pose['z'] = float(pose['z']) * 1e-3
            if 'linear_velocity' in pose:
                pose['linear_velocity'] = (
                    np.asarray(pose['linear_velocity'], dtype=float)
                    * 1e-3).tolist()
            elif 'velocity' in pose:
                pose['velocity'] = (
                    np.asarray(pose['velocity'], dtype=float) * 1e-3).tolist()
            return pose
        pose = np.asarray(racket_pose_mm, dtype=float)
        if pose.shape == (4, 4):
            pose = pose.copy()
            pose[:3, 3] *= 1e-3
            return pose
        if pose.shape == (3,):
            return pose * 1e-3
        raise ValueError('Unsupported racket pose representation')


def parse_args() -> argparse.Namespace:
    default_config = (
        Path(__file__).resolve().parents[1] / 'ball_tracker' / 'config.yaml')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_config)
    return parser.parse_args()


def main() -> None:
    """Run a dependency-free example of camera and control callbacks."""
    args = parse_args()
    bridge = RealtimeBallEstimatorBridge(args.config)
    racket_pose_mm = {
        'position': [0.0, 0.0, 650.0],
        'normal': [0.0, 0.0, 1.0],
    }
    try:
        # Representative IR positions. In the ROS node, t_img must come from
        # the RealSense frame metadata/header rather than time.time() here.
        bridge.on_ir_measurement(
            [20.0, -10.0, 900.0], 10.000, racket_pose_mm)
        bridge.on_ir_measurement(
            [24.0, -12.0, 875.0], 10.0167, racket_pose_mm)
        output = bridge.on_control_tick(
            10.0217,
            racket_pose_mm,
            robot_q=np.zeros(7),
            robot_dq=np.zeros(7),
            include_uncertainty=True,
        )
        print('filtered_position_mm:', output['position_mm'])
        print('filtered_velocity_mm_s:', output['velocity_mm_s'])
        print('impact_prediction:', output['impact_prediction'])
        print('rl_observation_shape:', output['rl_observation'].shape)
    finally:
        bridge.close()


if __name__ == '__main__':
    main()
