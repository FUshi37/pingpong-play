#!/usr/bin/env python3
"""Validate the hybrid ball estimator with noise and measurement dropouts.

The simulation runs ground truth at 600 Hz, estimator outputs at 200 Hz, and
IR-like position measurements at 60 Hz.  It logs per-tick CSV files plus JSON
summaries for 5/10/20/30 percent dropout scenarios by default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from pingpong_controller.ball_tracker import (
    BallStateEstimator,
    BallTrackerConfig,
    predict_contact,
)
from pingpong_controller.ball_tracker.ball_model import ball_dynamics
from pingpong_controller.ball_tracker.experiment_logger import (
    BallExperimentLogger,
)


SIMULATION_HZ = 600
CONTROL_HZ = 200
MEASUREMENT_HZ = 60
RACKET_HEIGHT_M = 0.65
RACKET_POSE = {
    'position': [0.0, 0.0, RACKET_HEIGHT_M],
    'normal': [0.0, 0.0, 1.0],
}
PREDICTION_HORIZONS_S = (0.050, 0.100, 0.150, 0.200)
BLACKOUT_DURATIONS_S = (0.050, 0.100, 0.200)


def _rmse(errors: list[np.ndarray]) -> float | None:
    if not errors:
        return None
    values = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _impact_metrics(
        records: list[dict[str, Any]],
        actual_time_s: float,
        actual_position: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    if not records:
        return metrics
    for horizon in PREDICTION_HORIZONS_S:
        candidates = [
            row for row in records
            if 0.0 <= actual_time_s - row['timestamp_s'] <= 0.30
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda row: abs(
                actual_time_s - row['timestamp_s'] - horizon),
        )
        position_error = float(np.linalg.norm(
            selected['impact_position'] - actual_position))
        time_error = abs(
            selected['timestamp_s'] + selected['time_to_contact_s']
            - actual_time_s)
        metrics[f'{int(round(horizon * 1000.0))}ms'] = {
            'position_error_m': position_error,
            'time_error_s': float(time_error),
            'actual_horizon_s': float(
                actual_time_s - selected['timestamp_s']),
        }
    return metrics


def run_dropout_scenario(
        dropout_rate: float,
        *,
        duration_s: float,
        measurement_std_m: Iterable[float],
        seed: int,
        output_dir: Path,
) -> dict[str, Any]:
    """Run one dropout scenario and return JSON-safe aggregate metrics."""
    rng = np.random.default_rng(seed)
    noise_std = np.asarray(measurement_std_m, dtype=float).reshape(3)
    config = BallTrackerConfig(
        measurement_std_m=tuple(noise_std),
        innovation_gate_chi2=25.0,
    )
    estimator = BallStateEstimator(config)
    truth = np.array([
        0.0, 0.0, RACKET_HEIGHT_M,
        0.32, -0.14, 4.10,
        0.0, 0.0, -9.81,
    ], dtype=float)

    position_errors: list[np.ndarray] = []
    velocity_errors: list[np.ndarray] = []
    acceleration_errors: list[np.ndarray] = []
    flight_position_errors: list[np.ndarray] = []
    flight_velocity_errors: list[np.ndarray] = []
    measurement_position_errors: list[np.ndarray] = []
    filtered_measurement_position_errors: list[np.ndarray] = []
    raw_velocity_errors: list[np.ndarray] = []
    age_samples: list[float] = []
    invalid_outputs = 0
    control_outputs = 0
    accepted_measurements = 0
    dropped_measurements = 0
    impacts = 0
    impact_cycle_records: list[dict[str, Any]] = []
    impact_results: list[dict[str, dict[str, float]]] = []
    last_raw_measurement = None
    last_raw_measurement_time = None
    last_actual_impact_time = -float('inf')

    scenario_name = f'dropout_{int(round(dropout_rate * 100)):02d}pct'
    log_path = output_dir / f'{scenario_name}.csv'
    dt = 1.0 / SIMULATION_HZ
    steps = int(round(duration_s * SIMULATION_HZ))
    control_stride = SIMULATION_HZ // CONTROL_HZ
    measurement_stride = SIMULATION_HZ // MEASUREMENT_HZ

    logger = BallExperimentLogger(log_path)
    try:
        for simulation_step in range(steps + 1):
            timestamp_s = simulation_step * dt
            actual_impact = None
            if simulation_step > 0:
                previous = truth.copy()
                predicted = ball_dynamics(truth, dt)
                if (previous[2] > RACKET_HEIGHT_M
                        and predicted[2] <= RACKET_HEIGHT_M
                        and predicted[5] < 0.0):
                    crossing = predict_contact(
                        previous, RACKET_POSE,
                        max_time_s=dt * 1.01,
                        min_approach_speed_m_s=0.0,
                    )
                    contact_dt = float(crossing['time_to_contact'])
                    contact_position = np.asarray(
                        crossing['impact_position'], dtype=float)
                    contact_velocity = np.asarray(
                        crossing['impact_velocity'], dtype=float)
                    impact_time = timestamp_s - dt + contact_dt
                    actual_impact = (impact_time, contact_position.copy())
                    last_actual_impact_time = impact_time
                    impacts += 1

                    # Racket impulse: preserve most lateral speed and launch a
                    # repeatable upward juggling arc.
                    truth[:3] = contact_position
                    truth[3:6] = np.array([
                        contact_velocity[0] * 0.98,
                        contact_velocity[1] * 0.98,
                        4.05 + 0.08 * np.sin(impacts),
                    ])
                    truth[6:9] = (0.0, 0.0, -9.81)
                    truth = ball_dynamics(truth, dt - contact_dt)
                else:
                    truth = predicted

            measurement = None
            estimate = None
            if simulation_step % measurement_stride == 0:
                if rng.random() < dropout_rate:
                    dropped_measurements += 1
                else:
                    measurement = truth[:3] + rng.normal(0.0, noise_std)
                    estimate = estimator.update(
                        measurement, timestamp_s, racket_pose=RACKET_POSE)
                    if estimate.measurement_accepted:
                        accepted_measurements += 1
                    measurement_position_errors.append(
                        measurement - truth[:3])
                    if estimate.valid and estimate.measurement_accepted:
                        filtered_measurement_position_errors.append(
                            estimate.position - truth[:3])
                    if (last_raw_measurement is not None
                            and last_raw_measurement_time is not None):
                        raw_dt = timestamp_s - last_raw_measurement_time
                        if raw_dt > 0.0:
                            raw_velocity = (
                                measurement - last_raw_measurement) / raw_dt
                            raw_velocity_errors.append(
                                raw_velocity - truth[3:6])
                    last_raw_measurement = measurement.copy()
                    last_raw_measurement_time = timestamp_s

            if simulation_step % control_stride == 0:
                if estimate is None:
                    estimate = estimator.predict_to(
                        timestamp_s, racket_pose=RACKET_POSE)
                control_outputs += 1
                if estimate.valid:
                    # Skip the brief initialization transient in aggregate
                    # state accuracy, but retain it in the raw CSV.
                    if timestamp_s >= 0.10:
                        position_errors.append(estimate.position - truth[:3])
                        velocity_errors.append(estimate.velocity - truth[3:6])
                        acceleration_errors.append(
                            estimate.acceleration - truth[6:9])
                        if timestamp_s - last_actual_impact_time >= 0.050:
                            flight_position_errors.append(
                                estimate.position - truth[:3])
                            flight_velocity_errors.append(
                                estimate.velocity - truth[3:6])
                    age_samples.append(estimate.time_since_measurement_s)
                    impact = estimate.impact_prediction or {}
                    if impact.get('valid', False):
                        impact_cycle_records.append({
                            'timestamp_s': timestamp_s,
                            'time_to_contact_s': float(
                                impact['time_to_contact']),
                            'impact_position': np.asarray(
                                impact['impact_position'], dtype=float),
                        })
                else:
                    invalid_outputs += 1
                logger.log(
                    estimate,
                    measurement=measurement,
                    ground_truth_state=truth,
                )

            if actual_impact is not None:
                impact_results.append(_impact_metrics(
                    impact_cycle_records,
                    actual_impact[0],
                    actual_impact[1],
                ))
                impact_cycle_records = []

        horizon_summary: dict[str, dict[str, Any]] = {}
        for horizon in PREDICTION_HORIZONS_S:
            key = f'{int(round(horizon * 1000.0))}ms'
            position_values = [
                cycle[key]['position_error_m']
                for cycle in impact_results if key in cycle
            ]
            time_values = [
                cycle[key]['time_error_s']
                for cycle in impact_results if key in cycle
            ]
            horizon_summary[key] = {
                'samples': len(position_values),
                'mean_position_error_m': _mean(position_values),
                'mean_time_error_s': _mean(time_values),
            }

        summary: dict[str, Any] = {
            'dropout_rate': float(dropout_rate),
            'duration_s': float(duration_s),
            'seed': int(seed),
            'measurement_std_m': noise_std.tolist(),
            'control_rate_hz': CONTROL_HZ,
            'measurement_rate_hz': MEASUREMENT_HZ,
            'impacts': impacts,
            'accepted_measurements': accepted_measurements,
            'dropped_measurements': dropped_measurements,
            'invalid_output_fraction': (
                float(invalid_outputs / control_outputs)
                if control_outputs else 0.0),
            'mean_prediction_age_s': _mean(age_samples),
            'raw_position_rmse_m': _rmse(measurement_position_errors),
            'ekf_position_rmse_at_measurement_m': _rmse(
                filtered_measurement_position_errors),
            'ekf_position_rmse_control_m': _rmse(position_errors),
            'ekf_flight_position_rmse_m': _rmse(flight_position_errors),
            'raw_derived_velocity_rmse_m_s': _rmse(raw_velocity_errors),
            'ekf_velocity_rmse_m_s': _rmse(velocity_errors),
            'ekf_flight_velocity_rmse_m_s': _rmse(flight_velocity_errors),
            'ekf_acceleration_rmse_m_s2': _rmse(acceleration_errors),
            'impact_prediction': horizon_summary,
            'tracker_diagnostics': estimator.diagnostics(),
        }
        logger.write_summary(summary)
        return summary
    finally:
        logger.close()


def benchmark_prediction_rate(iterations: int = 5000) -> float:
    """Measure prediction throughput for the >200 Hz requirement."""
    estimator = BallStateEstimator()
    estimator.update([0.0, 0.0, 1.0], 0.0, racket_pose=RACKET_POSE)
    start = time.perf_counter()
    for _ in range(iterations):
        estimator.predict_to(0.005, racket_pose=RACKET_POSE)
    elapsed = max(time.perf_counter() - start, 1e-12)
    return float(iterations / elapsed)


def run_blackout_experiment(
        measurement_std_m: Iterable[float],
        *,
        trials: int = 50,
        seed: int = 101,
) -> dict[str, dict[str, Any]]:
    """Measure drift after explicit 50/100/200 ms observation blackouts."""
    noise_std = np.asarray(measurement_std_m, dtype=float).reshape(3)
    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, Any]] = {}
    for blackout_s in BLACKOUT_DURATIONS_S:
        position_errors: list[np.ndarray] = []
        velocity_errors: list[np.ndarray] = []
        normalized_position_errors: list[float] = []
        invalid = 0
        for _ in range(trials):
            estimator = BallStateEstimator(BallTrackerConfig(
                measurement_std_m=tuple(noise_std),
                innovation_gate_chi2=25.0,
                max_prediction_duration_s=0.200,
            ))
            truth = np.array([
                0.0, 0.0, 1.0,
                0.35, -0.12, 3.8,
                0.0, 0.0, -9.81,
            ], dtype=float)
            timestamp_s = 0.0
            # Thirty camera frames establish velocity and acceleration before
            # the controlled blackout begins.
            for frame in range(30):
                if frame:
                    truth = ball_dynamics(truth, 1.0 / MEASUREMENT_HZ)
                    timestamp_s += 1.0 / MEASUREMENT_HZ
                measurement = truth[:3] + rng.normal(0.0, noise_std)
                estimator.update(measurement, timestamp_s)

            truth_after_gap = ball_dynamics(truth, blackout_s)
            estimate = estimator.predict_to(timestamp_s + blackout_s)
            if not estimate.valid:
                invalid += 1
                continue
            position_error = estimate.position - truth_after_gap[:3]
            velocity_error = estimate.velocity - truth_after_gap[3:6]
            position_errors.append(position_error)
            velocity_errors.append(velocity_error)
            normalized_position_errors.append(float(np.linalg.norm(
                position_error / np.maximum(estimate.position_std, 1e-9))))

        key = f'{int(round(blackout_s * 1000.0))}ms'
        results[key] = {
            'trials': int(trials),
            'invalid_fraction': float(invalid / max(trials, 1)),
            'position_drift_rmse_m': _rmse(position_errors),
            'velocity_drift_rmse_m_s': _rmse(velocity_errors),
            'mean_normalized_position_error': _mean(
                normalized_position_errors),
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dropout-rates', nargs='+', type=float,
        default=[0.05, 0.10, 0.20, 0.30],
        help='Independent measurement dropout probabilities.')
    parser.add_argument('--duration-s', type=float, default=6.0)
    parser.add_argument(
        '--noise-mm', nargs=3, type=float, default=[3.0, 3.0, 5.0],
        metavar=('X', 'Y', 'Z'),
        help='IR position noise standard deviations in millimetres.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path('outputs/ball_tracker_validation'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0.0:
        raise SystemExit('--duration-s must be positive')
    if any(rate < 0.0 or rate >= 1.0 for rate in args.dropout_rates):
        raise SystemExit('dropout rates must be in [0, 1)')
    measurement_std_m = np.asarray(args.noise_mm, dtype=float) * 1e-3
    if np.any(measurement_std_m <= 0.0):
        raise SystemExit('--noise-mm values must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for index, dropout_rate in enumerate(args.dropout_rates):
        key = f'{int(round(dropout_rate * 100)):02d}pct'
        results[key] = run_dropout_scenario(
            dropout_rate,
            duration_s=args.duration_s,
            measurement_std_m=measurement_std_m,
            seed=args.seed + index,
            output_dir=args.output_dir,
        )
    prediction_rate_hz = benchmark_prediction_rate()
    combined = {
        'prediction_benchmark_hz': prediction_rate_hz,
        'meets_200hz_requirement': prediction_rate_hz > 200.0,
        'missing_observation_experiment': run_blackout_experiment(
            measurement_std_m, seed=args.seed + 1000),
        'scenarios': results,
    }
    output_path = args.output_dir / 'summary.json'
    with output_path.open('w', encoding='utf-8') as stream:
        json.dump(combined, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(combined, indent=2, sort_keys=True, allow_nan=False))
    print(f'Wrote validation logs to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
