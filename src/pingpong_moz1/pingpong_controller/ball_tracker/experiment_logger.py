"""Streaming CSV/JSON experiment logging for tracker validation."""

from __future__ import annotations

import csv
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .estimator import BallEstimate


AXES = ('x', 'y', 'z')


class BallExperimentLogger:
    """Write one estimator row at a time without retaining a run in memory."""

    def __init__(
            self, csv_path: str | Path, *, flush_every: int = 200,
    ) -> None:
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(
            self._stream, fieldnames=self.field_names(), extrasaction='ignore')
        self._writer.writeheader()
        self._flush_every = max(1, int(flush_every))
        self._rows = 0
        self._lock = threading.Lock()
        self._closed = False

    @staticmethod
    def field_names() -> list[str]:
        names = [
            'timestamp_s', 'estimate_valid', 'mode',
            'measurement_available', 'measurement_accepted',
            'time_since_measurement_s',
        ]
        for prefix in ('measurement_p', 'truth_p', 'estimate_p'):
            names.extend(f'{prefix}{axis}' for axis in AXES)
        for prefix in ('truth_v', 'estimate_v', 'truth_a', 'estimate_a'):
            names.extend(f'{prefix}{axis}' for axis in AXES)
        for prefix in ('position_std_', 'velocity_std_', 'acceleration_std_'):
            names.extend(f'{prefix}{axis}' for axis in AXES)
        names += [
            'position_error_m', 'velocity_error_m_s',
            'impact_valid', 'time_to_contact_s', 'time_to_contact_std_s',
        ]
        for prefix in ('impact_p', 'impact_v'):
            names.extend(f'{prefix}{axis}' for axis in AXES)
        return names

    def log(
            self,
            estimate: BallEstimate,
            *,
            measurement: Optional[Any] = None,
            ground_truth_state: Optional[Any] = None,
            extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append one control-tick record."""
        row: dict[str, Any] = {
            'timestamp_s': float(estimate.timestamp_s),
            'estimate_valid': int(estimate.valid),
            'mode': estimate.mode,
            'measurement_available': int(measurement is not None),
            'measurement_accepted': (
                '' if estimate.measurement_accepted is None
                else int(estimate.measurement_accepted)),
            'time_since_measurement_s': float(
                estimate.time_since_measurement_s),
        }
        self._put_vector(row, 'measurement_p', measurement)

        truth = None
        if ground_truth_state is not None:
            truth = np.asarray(ground_truth_state, dtype=float).reshape(9)
        self._put_vector(row, 'truth_p', None if truth is None else truth[:3])
        self._put_vector(row, 'truth_v', None if truth is None else truth[3:6])
        self._put_vector(row, 'truth_a', None if truth is None else truth[6:9])

        if estimate.valid:
            self._put_vector(row, 'estimate_p', estimate.position)
            self._put_vector(row, 'estimate_v', estimate.velocity)
            self._put_vector(row, 'estimate_a', estimate.acceleration)
            self._put_vector(row, 'position_std_', estimate.position_std)
            self._put_vector(row, 'velocity_std_', estimate.velocity_std)
            self._put_vector(
                row, 'acceleration_std_', estimate.acceleration_std)
            if truth is not None:
                row['position_error_m'] = float(np.linalg.norm(
                    estimate.position - truth[:3]))
                row['velocity_error_m_s'] = float(np.linalg.norm(
                    estimate.velocity - truth[3:6]))
        else:
            for prefix in (
                    'estimate_p', 'estimate_v', 'estimate_a', 'position_std_',
                    'velocity_std_', 'acceleration_std_'):
                self._put_vector(row, prefix, None)

        impact = estimate.impact_prediction or {}
        impact_valid = bool(impact.get('valid', False))
        row['impact_valid'] = int(impact_valid)
        row['time_to_contact_s'] = (
            impact.get('time_to_contact') if impact_valid else '')
        row['time_to_contact_std_s'] = (
            impact.get('time_std_s')
            if impact_valid and impact.get('time_std_s') is not None else '')
        self._put_vector(
            row, 'impact_p',
            impact.get('impact_position') if impact_valid else None)
        self._put_vector(
            row, 'impact_v',
            impact.get('impact_velocity') if impact_valid else None)
        if extra:
            row.update(extra)

        with self._lock:
            if self._closed:
                raise RuntimeError('Cannot log to a closed experiment logger')
            self._writer.writerow(row)
            self._rows += 1
            if self._rows % self._flush_every == 0:
                self._stream.flush()

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        """Write a JSON summary next to the CSV and return its path."""
        summary_path = self.path.with_suffix('.summary.json')
        payload = dict(summary)
        payload.setdefault('csv_path', str(self.path))
        payload.setdefault('rows', self._rows)
        with summary_path.open('w', encoding='utf-8') as stream:
            json.dump(
                payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        return summary_path

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stream.flush()
                self._stream.close()
                self._closed = True

    def __enter__(self) -> 'BallExperimentLogger':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    @staticmethod
    def _put_vector(row: dict[str, Any], prefix: str, value: Any) -> None:
        if value is None:
            vector = [('', '', '')]
            values = vector[0]
        else:
            array = np.asarray(value, dtype=float).reshape(3)
            values = tuple(float(item) for item in array)
        for axis, item in zip(AXES, values):
            row[f'{prefix}{axis}'] = item


class RLPerformanceLogger:
    """Log baseline/EKF/EKF+impact policy episodes and aggregate them."""

    FIELD_NAMES = (
        'variant', 'episode', 'hits', 'continuous_juggling_time_s',
        'impact_position_error_m', 'failure_case',
    )

    def __init__(self, csv_path: str | Path) -> None:
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(
            self._stream, fieldnames=self.FIELD_NAMES)
        self._writer.writeheader()
        self._records: list[dict[str, Any]] = []
        self._closed = False

    def log_episode(
            self,
            variant: str,
            episode: int,
            *,
            hits: int,
            continuous_juggling_time_s: float,
            impact_position_error_m: Optional[float] = None,
            failure_case: str = '',
    ) -> None:
        """Append metrics for one policy evaluation episode."""
        if self._closed:
            raise RuntimeError('Cannot log to a closed RL performance logger')
        if hits < 0 or continuous_juggling_time_s < 0.0:
            raise ValueError('hits and juggling time must be non-negative')
        row = {
            'variant': str(variant),
            'episode': int(episode),
            'hits': int(hits),
            'continuous_juggling_time_s': float(
                continuous_juggling_time_s),
            'impact_position_error_m': (
                '' if impact_position_error_m is None
                else float(impact_position_error_m)),
            'failure_case': str(failure_case),
        }
        self._writer.writerow(row)
        self._stream.flush()
        self._records.append(row)

    def write_summary(self) -> Path:
        """Aggregate each policy variant into a JSON comparison."""
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self._records:
            grouped[record['variant']].append(record)
        summary: dict[str, Any] = {}
        for variant, records in grouped.items():
            impact_errors = [
                float(record['impact_position_error_m'])
                for record in records
                if record['impact_position_error_m'] != ''
            ]
            failures = Counter(
                str(record['failure_case']) for record in records
                if record['failure_case'])
            summary[variant] = {
                'episodes': len(records),
                'average_hits': float(np.mean([
                    record['hits'] for record in records])),
                'maximum_continuous_juggling_time_s': float(max(
                    record['continuous_juggling_time_s']
                    for record in records)),
                'mean_impact_position_error_m': (
                    None if not impact_errors
                    else float(np.mean(impact_errors))),
                'failure_cases': dict(failures),
            }
        summary_path = self.path.with_suffix('.summary.json')
        with summary_path.open('w', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write('\n')
        return summary_path

    def close(self) -> None:
        if not self._closed:
            self._stream.flush()
            self._stream.close()
            self._closed = True

    def __enter__(self) -> 'RLPerformanceLogger':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()
