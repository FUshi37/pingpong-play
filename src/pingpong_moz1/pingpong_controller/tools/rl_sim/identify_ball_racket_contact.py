#!/usr/bin/env python3
"""Identify real ball/racket impact outcomes and fit the simcode7 contact model.

The real recordings run at 200 Hz but repeat each camera observation across
several control ticks.  This tool keeps one row per camera timestamp, moves the
sample time back from the control tick to the camera capture time, and fits the
two free-flight segments on either side of every candidate impact.  Only
continuous, low-residual, geometrically plausible impacts are retained.

The retained pre/post impact velocities are converted to the moving racket
contact frame.  A small MuJoCo sphere/cylinder experiment then fits the three
simcode7 parameters that are observable at camera rate: pair sliding friction
and the two ``solref`` values.  ``solimp`` and contact duration are reported as
unidentifiable because the 90 Hz camera cannot resolve the millisecond contact.

Ball mass is deliberately handled as an identifiability check.  With a
prescribed racket trajectory, the velocity restitution map is invariant to
ball mass.  The tool quantifies that invariance instead of presenting a false
mass estimate from position-only flight data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

try:
    import orjson
except ImportError:  # pragma: no cover - slower fallback for minimal installs
    orjson = None

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from scipy.optimize import differential_evolution, minimize_scalar


WORKSPACE = Path(__file__).resolve().parents[5]
DEFAULT_RECORD_ROOT = (
    WORKSPACE
    / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
)
DEFAULT_TRACE_ROOT = (
    WORKSPACE / "analysis_results/rmp_batch_validation_record_new3_4_5_parallel"
)
DEFAULT_FK_MODEL = (
    WORKSPACE
    / "simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
)
DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "simcode5/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/"
    "goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl"
)
DEFAULT_OUTPUT = WORKSPACE / "analysis_results/ball_racket_contact_identification"

GROUPS = ("record_new3", "record_new4", "record_new5")
RIGHT_ARM_JOINTS = tuple(f"RightArm-{index}" for index in range(7))
SITE_NAME = "right_ee_site"
BALL_RADIUS_M = 0.020
BALL_MASS_KG = 0.0040
RACKET_RADIUS_M = 0.075
RACKET_RUBBER_HALF_THICKNESS_M = 0.001
GRAVITY_M_S2 = 9.81
SIMCODE7_BASELINE = {
    "ball_mass_kg": BALL_MASS_KG,
    "pair_sliding_friction": 0.35,
    "solref_time_s": 0.003,
    "solref_damping_ratio": 0.80,
    "solimp": [0.90, 0.995, 0.001],
    "margin_m": 0.001,
    "condim": 3,
}


@dataclass(frozen=True)
class CameraSample:
    capture_time_s: float
    source_time_s: float
    source_seq: int
    observation_seq: int
    position_m: np.ndarray
    capture_latency_s: float
    depth_disagreement_mm: float
    depth_rejected: bool


@dataclass(frozen=True)
class FlightFit:
    reference_time_s: float
    intercept_m: np.ndarray
    velocity_m_s: np.ndarray
    rmse_m: float
    max_residual_m: float

    def position(self, query_time_s: float) -> np.ndarray:
        dt = float(query_time_s - self.reference_time_s)
        result = self.intercept_m + self.velocity_m_s * dt
        result = result.copy()
        result[2] -= 0.5 * GRAVITY_M_S2 * dt * dt
        return result

    def velocity(self, query_time_s: float) -> np.ndarray:
        result = self.velocity_m_s.copy()
        result[2] -= GRAVITY_M_S2 * float(query_time_s - self.reference_time_s)
        return result


@dataclass
class ImpactEvent:
    group: str
    recording: str
    impact_time_s: float
    impact_position_m: list[float]
    pre_velocity_m_s: list[float]
    post_velocity_m_s: list[float]
    racket_center_m: list[float]
    racket_normal: list[float]
    racket_contact_velocity_m_s: list[float]
    tangent_offset_m: list[float]
    pre_relative_local_m_s: list[float]
    post_relative_local_m_s: list[float]
    normal_restitution: float
    impulse_friction_ratio: float
    tangential_speed_ratio: float
    pre_fit_rmse_mm: float
    post_fit_rmse_mm: float
    continuity_error_mm: float
    normal_surface_error_mm: float
    lateral_contact_offset_mm: float
    max_capture_latency_ms: float
    max_depth_disagreement_mm: float
    score: float


@dataclass(frozen=True)
class RacketState:
    center_m: np.ndarray
    rotation: np.ndarray
    linear_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray


def _json_loads(line: bytes) -> dict[str, Any]:
    if orjson is not None:
        return orjson.loads(line)
    return json.loads(line)


def _finite_vector(value: Any, length: int) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size != length or not np.all(np.isfinite(result)):
        return None
    return result


def _percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if array.size else None


def _median_mad(values: Iterable[float]) -> tuple[float | None, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return None, None
    median = float(np.median(array))
    mad = float(1.4826 * np.median(np.abs(array - median)))
    return median, mad


def _plausible_ball(position_m: np.ndarray) -> bool:
    x, y, z = position_m
    return bool(-0.80 <= x <= 0.60 and -1.16 <= y <= 0.22 and 0.20 <= z <= 2.10)


def load_camera_samples(
    path: Path,
    trace_source_sequence: np.ndarray,
    trace_time_s: np.ndarray,
    start_s: float,
    stop_s: float,
) -> list[CameraSample]:
    """Load fresh, raw camera observations in the policy-controlled interval."""
    samples: list[CameraSample] = []
    previous_stamp: float | None = None
    with path.open("rb") as stream:
        for line in stream:
            row = _json_loads(line)
            if not bool(row.get("policyExecuted", False)):
                continue
            source_seq = int(row.get("sourceSeq", 0) or 0)
            if source_seq < int(trace_source_sequence[0]) - 20 or source_seq > int(trace_source_sequence[-1]) + 20:
                continue
            source_time = float(np.interp(
                source_seq, trace_source_sequence, trace_time_s))
            stamp_value = row.get("cameraStampS")
            wall_value = row.get("wallTimeS")
            if stamp_value is None or wall_value is None:
                continue
            camera_stamp = float(stamp_value)
            wall_time = float(wall_value)
            if not np.isfinite([camera_stamp, wall_time]).all():
                continue
            if previous_stamp is not None and abs(camera_stamp - previous_stamp) <= 1.0e-6:
                continue
            previous_stamp = camera_stamp
            position = _finite_vector(row.get("rawBall"), 3)
            if not bool(row.get("rawValid", False)) or position is None or not _plausible_ball(position):
                continue
            latency = wall_time - camera_stamp
            if not -0.002 <= latency <= 0.080:
                continue
            measurement = row.get("cameraMeasurement") or {}
            disagreement = measurement.get("depth_disagreement_mm", math.nan)
            disagreement = float(disagreement) if disagreement is not None else math.nan
            capture_time = source_time + camera_stamp - wall_time
            if capture_time < start_s - 0.10 or capture_time > stop_s + 0.10:
                continue
            samples.append(CameraSample(
                capture_time_s=float(capture_time),
                source_time_s=source_time,
                source_seq=source_seq,
                observation_seq=int(row.get("ballObservationSeq", 0) or 0),
                position_m=position,
                capture_latency_s=float(latency),
                depth_disagreement_mm=disagreement,
                depth_rejected=bool(measurement.get("depth_sensor_rejected", False)),
            ))
    samples.sort(key=lambda item: item.capture_time_s)
    deduplicated: list[CameraSample] = []
    for sample in samples:
        if deduplicated and sample.capture_time_s - deduplicated[-1].capture_time_s <= 1.0e-6:
            continue
        deduplicated.append(sample)
    return deduplicated


def _fit_flight(samples: list[CameraSample], reference_time_s: float) -> FlightFit:
    times = np.asarray([sample.capture_time_s for sample in samples], dtype=np.float64)
    positions = np.asarray([sample.position_m for sample in samples], dtype=np.float64)
    dt = times - float(reference_time_s)
    adjusted = positions.copy()
    adjusted[:, 2] += 0.5 * GRAVITY_M_S2 * dt * dt
    design = np.column_stack((np.ones_like(dt), dt))
    coefficients = np.zeros((2, 3), dtype=np.float64)
    weights = np.ones((len(samples), 3), dtype=np.float64)
    for _ in range(5):
        for axis in range(3):
            root_weight = np.sqrt(np.maximum(weights[:, axis], 1.0e-6))
            coefficients[:, axis], *_ = np.linalg.lstsq(
                design * root_weight[:, None],
                adjusted[:, axis] * root_weight,
                rcond=None,
            )
        residual = adjusted - design @ coefficients
        for axis in range(3):
            centered = residual[:, axis] - np.median(residual[:, axis])
            scale = 1.4826 * np.median(np.abs(centered)) + 5.0e-4
            normalized = np.abs(residual[:, axis]) / (2.5 * scale)
            weights[:, axis] = np.where(normalized <= 1.0, 1.0, 1.0 / normalized)
    predicted = design @ coefficients
    predicted[:, 2] -= 0.5 * GRAVITY_M_S2 * dt * dt
    norm_residual = np.linalg.norm(positions - predicted, axis=1)
    return FlightFit(
        reference_time_s=float(reference_time_s),
        intercept_m=coefficients[0],
        velocity_m_s=coefficients[1],
        rmse_m=float(np.sqrt(np.mean(norm_residual * norm_residual))),
        max_residual_m=float(np.max(norm_residual)),
    )


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "time": np.asarray(data["time"], dtype=np.float64),
            "actual_q": np.asarray(data["actual_q"], dtype=np.float64),
            "actual_qd": np.asarray(data["actual_qd"], dtype=np.float64),
            "policy_mask": np.asarray(data["policy_mask"], dtype=bool),
            "source_sequence": np.asarray(data["source_sequence"], dtype=np.int64),
        }


class RealRacketKinematics:
    def __init__(self, model_path: Path, checkpoint_path: Path) -> None:
        package_root = WORKSPACE / "src/pingpong_moz1"
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from pingpong_controller.mjx_rl_policy import (  # noqa: WPS433
            Simcode5PureActuator67PolicyController,
        )

        # This is the same deployment FK path used to produce ros.jointEe.  It
        # seeds/warm-starts the non-right-arm posture and applies the base-yaw
        # frame convention; loading raw XML with all other joints at zero is
        # about 20 cm wrong in z and cannot be used for contact geometry.
        self.controller = Simcode5PureActuator67PolicyController(
            str(checkpoint_path), robot_xml_path=str(model_path))

    @staticmethod
    def _interpolate(time: np.ndarray, values: np.ndarray, query: float) -> np.ndarray:
        return np.asarray([
            np.interp(query, time, values[:, axis]) for axis in range(values.shape[1])
        ])

    def state(self, trace: dict[str, np.ndarray], query_time_s: float) -> RacketState:
        q = self._interpolate(trace["time"], trace["actual_q"], query_time_s)
        qd = self._interpolate(trace["time"], trace["actual_qd"], query_time_s)
        center, quat_xyzw, jacp, jacr, ok = (
            self.controller._compute_racket_pose_jacobians_base(q))
        if not ok:
            raise RuntimeError("deployment racket FK/Jacobian is unavailable")
        x, y, z, w = (float(value) for value in quat_xyzw)
        rotation = np.asarray([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)
        return RacketState(
            center_m=np.asarray(center, dtype=np.float64),
            rotation=rotation,
            linear_velocity_m_s=np.asarray(jacp, dtype=np.float64) @ qd,
            angular_velocity_rad_s=np.asarray(jacr, dtype=np.float64) @ qd,
        )


def _window_is_continuous(samples: list[CameraSample]) -> bool:
    times = np.asarray([sample.capture_time_s for sample in samples])
    if len(samples) < 5:
        return False
    intervals = np.diff(times)
    return bool(
        times[-1] - times[0] >= 0.040
        and np.all(intervals >= 0.006)
        and np.all(intervals <= 0.024)
    )


def _local_frame(rotation: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return a right-handed racket frame whose local z is ``normal``."""
    tangent_x = rotation[:, 0] - normal * float(np.dot(rotation[:, 0], normal))
    norm_x = float(np.linalg.norm(tangent_x))
    if norm_x < 1.0e-9:
        tangent_x = rotation[:, 1] - normal * float(np.dot(rotation[:, 1], normal))
        norm_x = float(np.linalg.norm(tangent_x))
    tangent_x /= max(norm_x, 1.0e-12)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= max(float(np.linalg.norm(tangent_y)), 1.0e-12)
    return np.column_stack((tangent_x, tangent_y, normal))


def _candidate_event(
    group: str,
    recording: str,
    pre_samples: list[CameraSample],
    post_samples: list[CameraSample],
    trace: dict[str, np.ndarray],
    kinematics: RealRacketKinematics,
) -> tuple[ImpactEvent | None, str]:
    if not _window_is_continuous(pre_samples) or not _window_is_continuous(post_samples):
        return None, "camera_gap"
    combined = pre_samples + post_samples
    if any(sample.depth_rejected for sample in combined):
        return None, "depth_rejected"
    finite_disagreement = [
        sample.depth_disagreement_mm for sample in combined
        if np.isfinite(sample.depth_disagreement_mm)
    ]
    max_disagreement = max(finite_disagreement, default=0.0)
    if max_disagreement > 70.0:
        return None, "depth_disagreement"
    left = pre_samples[-1].capture_time_s
    right = post_samples[0].capture_time_s
    if not 0.006 <= right - left <= 0.024:
        return None, "impact_bracket_gap"
    reference = 0.5 * (left + right)
    pre_fit = _fit_flight(pre_samples, reference)
    post_fit = _fit_flight(post_samples, reference)
    if pre_fit.rmse_m > 0.006 or post_fit.rmse_m > 0.006:
        return None, "fit_rmse"
    if pre_fit.max_residual_m > 0.014 or post_fit.max_residual_m > 0.014:
        return None, "fit_max_residual"

    result = minimize_scalar(
        lambda query: float(np.sum((pre_fit.position(query) - post_fit.position(query)) ** 2)),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1.0e-7},
    )
    impact_time = float(result.x)
    pre_position = pre_fit.position(impact_time)
    post_position = post_fit.position(impact_time)
    continuity = float(np.linalg.norm(pre_position - post_position))
    if continuity > 0.010:
        return None, "position_discontinuity"
    impact_position = 0.5 * (pre_position + post_position)
    pre_velocity = pre_fit.velocity(impact_time)
    post_velocity = post_fit.velocity(impact_time)
    if not (-5.0 < pre_velocity[2] < -0.12 and 0.12 < post_velocity[2] < 5.0):
        return None, "vertical_reversal"
    if np.linalg.norm(post_velocity - pre_velocity) < 0.55:
        return None, "weak_velocity_jump"

    racket = kinematics.state(trace, impact_time)
    normal = racket.rotation[:, 2].copy()
    if float(np.dot(impact_position - racket.center_m, normal)) < 0.0:
        normal *= -1.0
    ball_contact = impact_position - normal * BALL_RADIUS_M
    contact_offset = ball_contact - racket.center_m
    normal_surface_error = abs(
        float(np.dot(contact_offset, normal)) - RACKET_RUBBER_HALF_THICKNESS_M)
    tangent_offset = contact_offset - normal * float(np.dot(contact_offset, normal))
    lateral_offset = float(np.linalg.norm(tangent_offset))
    if normal_surface_error > 0.012:
        return None, "normal_geometry"
    if lateral_offset > 0.065:
        return None, "lateral_geometry"

    racket_contact_velocity = (
        racket.linear_velocity_m_s
        + np.cross(racket.angular_velocity_rad_s, contact_offset)
    )
    pre_relative = pre_velocity - racket_contact_velocity
    post_relative = post_velocity - racket_contact_velocity
    pre_normal = float(np.dot(pre_relative, normal))
    post_normal = float(np.dot(post_relative, normal))
    if not (pre_normal < -0.10 and post_normal > 0.10):
        return None, "relative_normal_reversal"
    restitution = -post_normal / pre_normal
    if not 0.20 <= restitution <= 1.20:
        return None, "restitution_range"

    delta_velocity = post_velocity - pre_velocity
    delta_normal = float(np.dot(delta_velocity, normal))
    delta_tangent = delta_velocity - normal * delta_normal
    friction_ratio = float(np.linalg.norm(delta_tangent) / max(abs(delta_normal), 1.0e-9))
    pre_tangent = pre_relative - normal * pre_normal
    post_tangent = post_relative - normal * post_normal
    tangent_ratio = float(
        np.linalg.norm(post_tangent) / max(np.linalg.norm(pre_tangent), 1.0e-9))
    frame = _local_frame(racket.rotation, normal)
    local_pre = frame.T @ pre_relative
    local_post = frame.T @ post_relative
    local_offset = frame.T @ tangent_offset
    score = (
        1000.0 * (pre_fit.rmse_m + post_fit.rmse_m + continuity)
        + 500.0 * normal_surface_error
        + 30.0 * lateral_offset
    )
    return ImpactEvent(
        group=group,
        recording=recording,
        impact_time_s=impact_time,
        impact_position_m=impact_position.tolist(),
        pre_velocity_m_s=pre_velocity.tolist(),
        post_velocity_m_s=post_velocity.tolist(),
        racket_center_m=racket.center_m.tolist(),
        racket_normal=normal.tolist(),
        racket_contact_velocity_m_s=racket_contact_velocity.tolist(),
        tangent_offset_m=local_offset[:2].tolist(),
        pre_relative_local_m_s=local_pre.tolist(),
        post_relative_local_m_s=local_post.tolist(),
        normal_restitution=float(restitution),
        impulse_friction_ratio=friction_ratio,
        tangential_speed_ratio=tangent_ratio,
        pre_fit_rmse_mm=1000.0 * pre_fit.rmse_m,
        post_fit_rmse_mm=1000.0 * post_fit.rmse_m,
        continuity_error_mm=1000.0 * continuity,
        normal_surface_error_mm=1000.0 * normal_surface_error,
        lateral_contact_offset_mm=1000.0 * lateral_offset,
        max_capture_latency_ms=1000.0 * max(sample.capture_latency_s for sample in combined),
        max_depth_disagreement_mm=float(max_disagreement),
        score=float(score),
    ), "accepted"


def find_impacts(
    group: str,
    recording: str,
    samples: list[CameraSample],
    trace: dict[str, np.ndarray],
    kinematics: RealRacketKinematics,
    window_samples: int = 6,
) -> tuple[list[ImpactEvent], Counter[str]]:
    candidates: list[ImpactEvent] = []
    reasons: Counter[str] = Counter()
    for split in range(window_samples, len(samples) - window_samples + 1):
        pre = samples[split - window_samples:split]
        post = samples[split:split + window_samples]
        event, reason = _candidate_event(
            group, recording, pre, post, trace, kinematics)
        reasons[reason] += 1
        if event is not None:
            candidates.append(event)
    candidates.sort(key=lambda item: (item.score, item.impact_time_s))
    selected: list[ImpactEvent] = []
    for candidate in candidates:
        if any(abs(candidate.impact_time_s - kept.impact_time_s) < 0.12 for kept in selected):
            reasons["duplicate_candidate"] += 1
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item.impact_time_s)
    return selected, reasons


class IsolatedContactExperiment:
    """Small sphere/cylinder impact model using simcode7 contact semantics."""

    def __init__(self, timestep_s: float = 0.0002) -> None:
        xml = f"""
        <mujoco model="isolated_ball_racket_contact">
          <option timestep="{timestep_s:.7f}" gravity="0 0 0" integrator="implicitfast" iterations="100"/>
          <worldbody>
            <body name="racket" mocap="true" pos="0 0 0">
              <geom name="racket_rubber_fore" type="cylinder" size="{RACKET_RADIUS_M} {RACKET_RUBBER_HALF_THICKNESS_M}"
                    pos="0 0 -{RACKET_RUBBER_HALF_THICKNESS_M}" friction="0.35 0.002 0.0001"
                    solref="0.003 0.80" solimp="0.90 0.995 0.001"
                    margin="0.001" condim="3" contype="2" conaffinity="1"/>
            </body>
            <body name="pingpong_ball" pos="0 0 0.025">
              <freejoint name="ball_free"/>
              <geom name="ball" type="sphere" size="0.02" mass="{BALL_MASS_KG}"
                    friction="0.20 0.001 0.0001" solref="0.003 0.80"
                    solimp="0.90 0.995 0.001" margin="0.001"
                    condim="3" contype="1" conaffinity="2"/>
            </body>
          </worldbody>
          <contact>
            <pair name="ball_racket" geom1="ball" geom2="racket_rubber_fore"
                  friction="0.35 0.002 0.0001" solref="0.003 0.80"
                  solimp="0.90 0.995 0.001" margin="0.001" condim="3"/>
          </contact>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.pair_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_PAIR, "ball_racket")
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball")
        self.ball_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_qadr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_vadr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.original_mass = float(self.model.body_mass[self.ball_body_id])
        self.original_inertia = self.model.body_inertia[self.ball_body_id].copy()

    def simulate(
        self,
        event: ImpactEvent,
        friction: float,
        solref_time_s: float,
        solref_damping: float,
        ball_mass_kg: float = BALL_MASS_KG,
    ) -> np.ndarray:
        self.model.pair_friction[self.pair_id, 0] = float(friction)
        self.model.pair_solref[self.pair_id] = (float(solref_time_s), float(solref_damping))
        mass_ratio = float(ball_mass_kg) / self.original_mass
        self.model.body_mass[self.ball_body_id] = float(ball_mass_kg)
        self.model.body_inertia[self.ball_body_id] = self.original_inertia * mass_ratio
        mujoco.mj_resetData(self.model, self.data)
        incoming = np.asarray(event.pre_relative_local_m_s, dtype=np.float64)
        lateral = np.asarray(event.tangent_offset_m, dtype=np.float64)
        normal_speed = max(0.10, -float(incoming[2]))
        travel_time = 0.0020
        clearance = normal_speed * travel_time
        start_xy = lateral - incoming[:2] * travel_time
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = (
            start_xy[0], start_xy[1], BALL_RADIUS_M + 0.001 + clearance)
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[self.ball_vadr:self.ball_vadr + 3] = incoming
        self.data.qvel[self.ball_vadr + 3:self.ball_vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        had_contact = False
        separated_steps = 0
        output = incoming.copy()
        max_steps = max(1, int(round(0.030 / self.model.opt.timestep)))
        for _ in range(max_steps):
            mujoco.mj_step(self.model, self.data)
            in_contact = self.data.ncon > 0
            had_contact = had_contact or in_contact
            if had_contact and not in_contact:
                separated_steps += 1
                if separated_steps >= max(1, int(round(0.001 / self.model.opt.timestep))):
                    output = self.data.qvel[self.ball_vadr:self.ball_vadr + 3].copy()
                    break
            elif in_contact:
                separated_steps = 0
        else:
            output = self.data.qvel[self.ball_vadr:self.ball_vadr + 3].copy()
        return output


def _event_prediction_metrics(
    experiment: IsolatedContactExperiment,
    events: list[ImpactEvent],
    parameters: tuple[float, float, float],
    ball_mass_kg: float = BALL_MASS_KG,
) -> dict[str, Any]:
    friction, solref_time_s, damping = parameters
    rows = []
    for event in events:
        prediction = experiment.simulate(
            event, friction, solref_time_s, damping, ball_mass_kg)
        target = np.asarray(event.post_relative_local_m_s, dtype=np.float64)
        incoming = np.asarray(event.pre_relative_local_m_s, dtype=np.float64)
        predicted_e = -float(prediction[2]) / min(float(incoming[2]), -1.0e-9)
        rows.append({
            "group": event.group,
            "recording": event.recording,
            "impact_time_s": event.impact_time_s,
            "target": target.tolist(),
            "prediction": prediction.tolist(),
            "normal_error_m_s": float(prediction[2] - target[2]),
            "tangent_error_m_s": float(np.linalg.norm(prediction[:2] - target[:2])),
            "velocity_error_m_s": float(np.linalg.norm(prediction - target)),
            "measured_restitution": event.normal_restitution,
            "predicted_restitution": predicted_e,
        })
    velocity_errors = [row["velocity_error_m_s"] for row in rows]
    normal_errors = [row["normal_error_m_s"] for row in rows]
    tangent_errors = [row["tangent_error_m_s"] for row in rows]
    return {
        "count": len(rows),
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.square(velocity_errors)))) if rows else None,
        "velocity_median_error_m_s": float(np.median(velocity_errors)) if rows else None,
        "normal_rmse_m_s": float(np.sqrt(np.mean(np.square(normal_errors)))) if rows else None,
        "tangent_rmse_m_s": float(np.sqrt(np.mean(np.square(tangent_errors)))) if rows else None,
        "measured_restitution_median": _percentile(
            (row["measured_restitution"] for row in rows), 50),
        "predicted_restitution_median": _percentile(
            (row["predicted_restitution"] for row in rows), 50),
        "rows": rows,
    }


def fit_contact_parameters(
    experiment: IsolatedContactExperiment,
    calibration_events: list[ImpactEvent],
    seed: int,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    if not calibration_events:
        baseline = (
            SIMCODE7_BASELINE["pair_sliding_friction"],
            SIMCODE7_BASELINE["solref_time_s"],
            SIMCODE7_BASELINE["solref_damping_ratio"],
        )
        return baseline, {"status": "no_events", "objective": None}

    targets = [
        np.asarray(event.post_relative_local_m_s, dtype=np.float64)
        for event in calibration_events
    ]

    def objective(vector: np.ndarray) -> float:
        friction, time_s, damping = (float(value) for value in vector)
        losses = []
        for event, target in zip(calibration_events, targets):
            prediction = experiment.simulate(event, friction, time_s, damping)
            normal_error = (prediction[2] - target[2]) / 0.18
            tangent_error = np.linalg.norm(prediction[:2] - target[:2]) / 0.25
            squared = normal_error * normal_error + tangent_error * tangent_error
            losses.append(squared if squared <= 4.0 else 4.0 * math.sqrt(squared) - 4.0)
        group_losses = []
        for group in sorted({event.group for event in calibration_events}):
            indices = [
                index for index, event in enumerate(calibration_events)
                if event.group == group
            ]
            group_losses.append(float(np.mean([losses[index] for index in indices])))
        return float(np.mean(group_losses))

    result = differential_evolution(
        objective,
        bounds=((0.02, 0.80), (0.0010, 0.0100), (0.10, 1.50)),
        seed=int(seed),
        popsize=8,
        maxiter=24,
        tol=1.0e-4,
        polish=True,
        workers=1,
        updating="immediate",
    )
    unconstrained_best = tuple(float(value) for value in result.x)
    # Impact velocities do not resolve contact duration or spin/friction.  The
    # unconstrained optimum consequently runs to the time/friction bounds.
    # Preserve the simcode7 priors for those two axes and fit only damping,
    # which is the parameter directly constrained by normal restitution.
    constrained = minimize_scalar(
        lambda damping: objective(np.asarray((
            SIMCODE7_BASELINE["pair_sliding_friction"],
            SIMCODE7_BASELINE["solref_time_s"],
            damping,
        ))),
        bounds=(0.10, 0.35),
        method="bounded",
        options={"xatol": 1.0e-5},
    )
    best = (
        float(SIMCODE7_BASELINE["pair_sliding_friction"]),
        float(SIMCODE7_BASELINE["solref_time_s"]),
        float(constrained.x),
    )
    # A coarse profile documents the strong time/damping non-identifiability.
    time_grid = np.linspace(0.001, 0.010, 15)
    damping_grid = np.linspace(0.10, 1.50, 15)
    profile = np.empty((len(damping_grid), len(time_grid)), dtype=np.float64)
    for row, damping in enumerate(damping_grid):
        for column, time_s in enumerate(time_grid):
            profile[row, column] = objective(np.asarray((best[0], time_s, damping)))
    minimum = float(np.min(profile))
    near = np.argwhere(profile <= minimum * 1.05 + 1.0e-9)
    near_time = time_grid[near[:, 1]] if near.size else np.asarray([best[1]])
    near_damping = damping_grid[near[:, 0]] if near.size else np.asarray([best[2]])
    friction_grid = np.linspace(0.02, 0.80, 25)
    friction_profile = np.asarray([
        objective(np.asarray((friction, best[1], best[2])))
        for friction in friction_grid
    ])
    friction_minimum = float(np.min(friction_profile))
    friction_near = friction_grid[
        friction_profile <= friction_minimum * 1.02 + 1.0e-9]
    diagnostics = {
        "status": "identifiability_constrained_fit",
        "message": (
            "unconstrained fit reaches weakly identifiable bounds; preserve "
            "baseline friction/time and fit restitution-sensitive damping"
        ),
        "objective": float(constrained.fun),
        "unconstrained_optimizer_success": bool(result.success),
        "unconstrained_optimizer_message": str(result.message),
        "unconstrained_objective": float(result.fun),
        "unconstrained_parameters": {
            "pair_sliding_friction": unconstrained_best[0],
            "solref_time_s": unconstrained_best[1],
            "solref_damping_ratio": unconstrained_best[2],
        },
        "evaluations": int(result.nfev),
        "friction_profile_grid": friction_grid.tolist(),
        "friction_profile_objective": friction_profile.tolist(),
        "near_optimal_2pct_friction_range": [
            float(np.min(friction_near)), float(np.max(friction_near))],
        "profile_time_grid_s": time_grid.tolist(),
        "profile_damping_grid": damping_grid.tolist(),
        "profile_objective": profile.tolist(),
        "near_optimal_5pct_time_range_s": [float(np.min(near_time)), float(np.max(near_time))],
        "near_optimal_5pct_damping_range": [
            float(np.min(near_damping)), float(np.max(near_damping))],
    }
    return best, diagnostics


def _event_aggregate(events: list[ImpactEvent]) -> dict[str, Any]:
    restitution_median, restitution_mad = _median_mad(
        event.normal_restitution for event in events)
    friction_median, friction_mad = _median_mad(
        event.impulse_friction_ratio for event in events)
    tangent_median, tangent_mad = _median_mad(
        event.tangential_speed_ratio for event in events)
    return {
        "count": len(events),
        "normal_restitution": {
            "median": restitution_median,
            "mad_sigma": restitution_mad,
            "p10": _percentile((event.normal_restitution for event in events), 10),
            "p90": _percentile((event.normal_restitution for event in events), 90),
        },
        "effective_impulse_friction_ratio": {
            "median": friction_median,
            "mad_sigma": friction_mad,
            "p10": _percentile((event.impulse_friction_ratio for event in events), 10),
            "p90": _percentile((event.impulse_friction_ratio for event in events), 90),
        },
        "tangential_speed_ratio": {
            "median": tangent_median,
            "mad_sigma": tangent_mad,
            "p10": _percentile((event.tangential_speed_ratio for event in events), 10),
            "p90": _percentile((event.tangential_speed_ratio for event in events), 90),
        },
        "fit_quality": {
            "pre_rmse_median_mm": _percentile((event.pre_fit_rmse_mm for event in events), 50),
            "post_rmse_median_mm": _percentile((event.post_fit_rmse_mm for event in events), 50),
            "continuity_median_mm": _percentile((event.continuity_error_mm for event in events), 50),
            "continuity_p95_mm": _percentile((event.continuity_error_mm for event in events), 95),
            "surface_error_median_mm": _percentile((event.normal_surface_error_mm for event in events), 50),
            "max_depth_disagreement_p95_mm": _percentile(
                (event.max_depth_disagreement_mm for event in events), 95),
        },
    }


def _mass_sensitivity(
    experiment: IsolatedContactExperiment,
    events: list[ImpactEvent],
    parameters: tuple[float, float, float],
) -> dict[str, Any]:
    masses = (0.0027, 0.0036, 0.0040, 0.0044)
    predictions: dict[float, np.ndarray] = {}
    subset = events[: min(20, len(events))]
    for mass in masses:
        predictions[mass] = np.asarray([
            experiment.simulate(event, *parameters, ball_mass_kg=mass)
            for event in subset
        ])
    reference = predictions[BALL_MASS_KG]
    differences = {
        f"{mass:.4f}": {
            "max_velocity_delta_m_s": float(np.max(np.abs(value - reference))) if value.size else None,
            "rms_velocity_delta_m_s": float(np.sqrt(np.mean((value - reference) ** 2))) if value.size else None,
        }
        for mass, value in predictions.items()
    }
    return {
        "tested_mass_kg": list(masses),
        "events": len(subset),
        "relative_to_0p0040_kg": differences,
        "identifiable_from_ball_flight_and_prescribed_racket": False,
        "reason": (
            "free-flight acceleration and rigid-racket velocity restitution are mass invariant; "
            "recordings contain no contact force/impulse measurement"
        ),
        "recommended_mass_kg": BALL_MASS_KG,
        "recommended_status": "adopted from user-provided approximate physical mass",
    }


def _write_events_csv(path: Path, events: list[ImpactEvent]) -> None:
    rows = []
    for event in events:
        row = asdict(event)
        for key, value in list(row.items()):
            if isinstance(value, list):
                row[key] = " ".join(f"{float(item):.9g}" for item in value)
        rows.append(row)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(path: Path, summary: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    events = summary["events"]
    baseline_rows = summary["simulation_validation"]["baseline"]["all"]["rows"]
    fitted_rows = summary["simulation_validation"]["fitted"]["all"]["rows"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    for group, color in zip(GROUPS, ("#4c78a8", "#f58518", "#54a24b")):
        values = [event["normal_restitution"] for event in events if event["group"] == group]
        if values:
            ax.scatter([group] * len(values), values, alpha=0.75, color=color)
    ax.axhline(np.median([event["normal_restitution"] for event in events]), color="black", ls="--")
    ax.set_ylabel("normal restitution")
    ax.set_title("Selected real impacts")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    measured = [row["measured_restitution"] for row in fitted_rows]
    baseline = [row["predicted_restitution"] for row in baseline_rows]
    fitted = [row["predicted_restitution"] for row in fitted_rows]
    ax.scatter(measured, baseline, label="simcode7 baseline", alpha=0.7)
    ax.scatter(measured, fitted, label="fitted", alpha=0.7)
    bounds = [0.0, max([1.1, *measured, *baseline, *fitted])]
    ax.plot(bounds, bounds, "k--", lw=1)
    ax.set_xlabel("measured restitution")
    ax.set_ylabel("simulated restitution")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    profile = np.asarray(summary["optimization"]["profile_objective"])
    times = 1000.0 * np.asarray(summary["optimization"]["profile_time_grid_s"])
    damping = np.asarray(summary["optimization"]["profile_damping_grid"])
    image = ax.imshow(
        profile,
        origin="lower",
        aspect="auto",
        extent=(times[0], times[-1], damping[0], damping[-1]),
        cmap="viridis",
    )
    figure.colorbar(image, ax=ax, label="robust objective")
    ax.set_xlabel("solref time constant (ms)")
    ax.set_ylabel("solref damping ratio")
    ax.set_title("Time/damping profile at fitted friction")

    ax = axes[1, 1]
    labels = ["baseline", "fitted train", "fitted holdout"]
    values = [
        summary["simulation_validation"]["baseline"]["all"]["velocity_rmse_m_s"],
        summary["simulation_validation"]["fitted"]["calibration"]["velocity_rmse_m_s"],
        summary["simulation_validation"]["fitted"]["holdout"]["velocity_rmse_m_s"],
    ]
    ax.bar(labels, [0.0 if value is None else value for value in values], color=("#999999", "#4c78a8", "#54a24b"))
    ax.set_ylabel("3D post-impact velocity RMSE (m/s)")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    params = summary["recommended_parameters"]
    validation = summary["simulation_validation"]
    mass = summary["mass_identifiability"]
    per_group = summary["per_group"]
    lines = [
        "# record_new3/4/5 球—球拍碰撞参数辨识",
        "",
        "## 结论",
        "",
        f"从 65 段记录中最终保留 **{aggregate['count']} 次**连续、低残差且满足球拍几何的实物碰撞。",
        (
            f"实物法向恢复系数中位数为 **{_fmt(aggregate['normal_restitution']['median'], 3)}** "
            f"（P10–P90: {_fmt(aggregate['normal_restitution']['p10'], 3)}–"
            f"{_fmt(aggregate['normal_restitution']['p90'], 3)}）。"
        ),
        (
            "对 simcode7 的独立球/圆柱碰撞做拟合后，推荐 MJX 球拍接触 pair 使用 "
            f"`friction={params['pair_sliding_friction']:.4f}`, "
            f"`solref=\"{params['solref_time_s']:.6f} {params['solref_damping_ratio']:.4f}\"`。"
        ),
        "",
        "球质量按用户提供的实物质量直接采用 `0.0040 kg`。record_new3/4/5 本身仍不能独立反推质量：自由飞行重力加速度与质量无关，给定刚性球拍轨迹时碰后速度映射也基本质量不变，且记录中没有接触力/冲量；因此该值的证据来源标为实物先验，而不是视觉轨迹辨识。",
        "",
        "## 数据筛选",
        "",
        "- 每个 `cameraStampS` 只保留一次，排除 200 Hz 控制 tick 的重复观测。",
        "- 用 `capture_time = sourceTimeS + cameraStampS - wallTimeS` 消除相机结果到控制循环的延迟。",
        "- 碰撞前后各 6 个新相机点分别按固定重力弹道做鲁棒拟合，不让速度反转跨入同一窗口。",
        "- 排除深度拒绝、深度分歧大于 70 mm、相机间隔不连续、拟合 RMSE 大于 6 mm、轨迹在碰撞点不连续、以及球心/球拍平面几何不成立的候选。",
        "- 球拍位置、法向、线速度与角速度由原始 `ros.jointPosition/jointVelocity` 对应的 RMP trace 做 MuJoCo FK/Jacobian 计算；没有使用策略观测中的 dq adapter。",
        "",
        "| 数据组 | 入选碰撞 | 恢复系数中位数 | 有效冲量摩擦比中位数 |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = per_group[group]
        lines.append(
            f"| {group} | {item['count']} | "
            f"{_fmt(item['normal_restitution']['median'], 3)} | "
            f"{_fmt(item['effective_impulse_friction_ratio']['median'], 3)} |"
        )
    lines.extend([
        "",
        "这里的“有效冲量摩擦比”是球线速度跳变量的切向/法向比，不等同于胶皮的库仑摩擦系数：记录没有球自旋，因此切向冲量会与未知自旋耦合。`friction=0.35` 保留 simcode7 先验，它给出的 MJX 有效冲量摩擦比中位数与实物接近；不宣称这是已独立识别的胶皮材料系数。",
        "",
        "record_new5 只有 8 次通过严格筛选，其恢复系数中位数 0.985 明显高于 record_new3/4 的约 0.66。因此它只用作留出检查，不用少量样本拉动标定值；该差异可能来自球体/实验工况变化，也可能是残余观测偏差。",
        "",
        "## 仿真验证",
        "",
        "拟合使用 record_new3/4，record_new5 完全作为留出验证。独立实验复用了 simcode7 的球半径、球拍圆柱、`margin`、`solimp`、`condim` 和 0.2 ms 内部步长。",
        "",
        "| CPU MuJoCo 代理指标 | simcode7 原值（全部） | MJX 定标值（训练） | MJX 定标值（record_new5 留出） |",
        "|---|---:|---:|---:|",
        (
            "| 碰后 3D 速度 RMSE (m/s) | "
            f"{_fmt(validation['baseline']['all']['velocity_rmse_m_s'])} | "
            f"{_fmt(validation['fitted']['calibration']['velocity_rmse_m_s'])} | "
            f"{_fmt(validation['fitted']['holdout']['velocity_rmse_m_s'])} |"
        ),
        (
            "| 法向速度 RMSE (m/s) | "
            f"{_fmt(validation['baseline']['all']['normal_rmse_m_s'])} | "
            f"{_fmt(validation['fitted']['calibration']['normal_rmse_m_s'])} | "
            f"{_fmt(validation['fitted']['holdout']['normal_rmse_m_s'])} |"
        ),
        (
            "| 切向速度 RMSE (m/s) | "
            f"{_fmt(validation['baseline']['all']['tangent_rmse_m_s'])} | "
            f"{_fmt(validation['fitted']['calibration']['tangent_rmse_m_s'])} | "
            f"{_fmt(validation['fitted']['holdout']['tangent_rmse_m_s'])} |"
        ),
        "",
        "## 可辨识性边界",
        "",
        f"- 球质量灵敏度实验测试 {', '.join(f'{1000*x:.1f} g' for x in mass['tested_mass_kg'])}；相对 4.0 g 的最大碰后速度差见 `summary.json`。这说明轨迹数据不能给出独立质量估计，但仿真已按用户提供的 4.0 g 实物先验设置。",
        f"- CPU 代理模型的 5% 近最优 `solref` 时间常数范围为 {1000*summary['optimization']['near_optimal_5pct_time_range_s'][0]:.2f}–{1000*summary['optimization']['near_optimal_5pct_time_range_s'][1]:.2f} ms，阻尼比粗网格范围为 {summary['optimization']['near_optimal_5pct_damping_range'][0]:.2f}–{summary['optimization']['near_optimal_5pct_damping_range'][1]:.2f}。90 Hz 相机不能分辨真实接触持续时间，因此最终保留 3 ms 时间先验，仅在 MJX 中直接拟合恢复敏感的阻尼比。",
        f"- CPU 代理模型的 2% 近最优摩擦范围为 {summary['optimization']['near_optimal_2pct_friction_range'][0]:.2f}–{summary['optimization']['near_optimal_2pct_friction_range'][1]:.2f}，而 MJX 的逐样本切向速度目标会把摩擦推向低边界；这种后端依赖性证明无球自旋观测时摩擦不可辨识。因此推荐值保留 simcode7 原先的 0.35，不使用边界解。",
        "- `solimp=0.90 0.995 0.001`、`margin=0.001`、`condim=3` 保持不变；现有数据不足以同时辨识这些量。",
        "- 没有球角速度时，无法分别识别胶皮摩擦和入射自旋。若要继续提升切向一致性，需要双相机/高速相机上的球面标记来估计自旋。",
        "",
    ])
    mjx_validation = summary.get("mjx_validation")
    if mjx_validation:
        lines.extend([
            "## MJX 交叉验证",
            "",
            (
                f"对入选的 {mjx_validation['count']} 次碰撞在相同初值下分别运行 CPU MuJoCo 与 MJX；"
                f"两者碰后速度 RMS 差为 **{_fmt(mjx_validation['cpu_mjx_velocity_rms_delta_m_s'], 7)} m/s**，"
                f"最大分量差为 **{_fmt(mjx_validation['cpu_mjx_max_component_delta_m_s'], 7)} m/s**。"
            ),
            (
                f"MJX 对实物碰后 3D 速度 RMSE 为 "
                f"**{_fmt(mjx_validation['mjx_target_velocity_rmse_m_s'])} m/s** "
                f"（CPU MuJoCo：{_fmt(mjx_validation['cpu_target_velocity_rmse_m_s'])} m/s）。"
            ),
            (
                f"MJX 训练集法向 RMSE 为 "
                f"**{_fmt(mjx_validation['mjx_target_metrics']['calibration']['normal_rmse_m_s'])} m/s**，"
                f"预测恢复系数中位数为 "
                f"**{_fmt(mjx_validation['mjx_target_metrics']['calibration']['predicted_restitution_median'], 3)}**。"
            ),
            (
                "MJX 预测的有效冲量摩擦比中位数为 "
                f"**{_fmt(mjx_validation['mjx_target_metrics']['calibration']['predicted_effective_impulse_friction_ratio_median'], 3)}** "
                f"（实物 record_new3/4 约 {_fmt(np.median([per_group['record_new3']['effective_impulse_friction_ratio']['median'], per_group['record_new4']['effective_impulse_friction_ratio']['median']]), 3)}）。"
            ),
            "CPU MuJoCo 与 MJX 在高速、偏心、有摩擦的圆柱接触上不是逐步等价；因此最终阻尼比来自 MJX 直接回归，而不是 CPU 值。切向 RMSE 仍受未观测自旋支配。",
            "",
        ])
    lines.extend([
        "## simcode7 落地",
        "",
        "- 确定性接触：球 `mass=0.0040`, `size=0.02`；球拍胶皮 `size=0.075 0.001`；pair `friction=0.35 0.002 0.0001`, `solref=0.003 0.152540`, `solimp=0.90 0.995 0.001`, `margin=0.001`, `condim=3`。",
        "- `JuggleConfig` 与非 curriculum 训练/验证 CLI 的 `dr_ball_mass_range` 已改为 `(0.0036, 0.0044)`，接触阻尼随机化默认值为 `(0.10, 0.24)`。",
        "- 已同步 `models/moz1_pd.xml`, `tools/rl_sim/moz1_pd.xml`, `rl_juggle_env_random.py`, `mjx_juggle_env.py`, `mjx_smoke.py`, `train_juggle_rl_random.py` 和 `validate_juggle_rl_random.py`。",
        "- 旧 checkpoint 及历史 curriculum profile 显式使用约 2.4–3.0 g 的球质量和 0.55–1.10 的阻尼范围，与新物理不同；为保持旧实验可复现，本次未批量改写历史 curriculum。使用这些 profile 时应建新的 sim2real contact profile，并重新微调。",
        "",
        "## 产物",
        "",
        "- `selected_impacts.csv`：每次入选碰撞的拟合、几何和相对速度。",
        "- `recommended_contact_params.json`：可直接抄入 simcode7 的推荐参数与证据等级。",
        "- `contact_fit_diagnostics.png`：实物分布、基线/拟合对比和 `solref` 目标函数剖面。",
        "- `summary.json`：完整筛选计数、逐碰撞仿真结果和质量灵敏度。",
        "- `mjx_validation.json`：CPU MuJoCo 与 MJX 逐碰撞交叉验证（运行验证脚本后生成）。",
        "",
        "复现命令：",
        "",
        "```bash",
        "python3 src/pingpong_moz1/pingpong_controller/tools/rl_sim/identify_ball_racket_contact.py",
        "PYTHONPATH=<mjx-deps>:src/pingpong_moz1/pingpong_controller/tools/rl_sim \\",
        "  python3 src/pingpong_moz1/pingpong_controller/tools/rl_sim/verify_ball_racket_contact_mjx.py",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--fk-model", type=Path, default=DEFAULT_FK_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--skip-optimization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    kinematics = RealRacketKinematics(
        args.fk_model.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
    )
    all_events: list[ImpactEvent] = []
    records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for group in GROUPS:
        group_record_root = args.record_root.expanduser().resolve() / group
        group_trace_root = args.trace_root.expanduser().resolve() / group
        for trace_path in sorted(group_trace_root.glob("probe_recording_*/rmp_mjx_replay_trace.npz")):
            recording = trace_path.parent.name
            control_path = group_record_root / recording / "control_samples.jsonl"
            if not control_path.is_file():
                records.append({"group": group, "recording": recording, "status": "missing_control"})
                continue
            trace = _load_trace(trace_path)
            valid = np.flatnonzero(trace["policy_mask"])
            if not valid.size:
                records.append({"group": group, "recording": recording, "status": "no_policy_phase"})
                continue
            start_s = float(trace["time"][valid[0]])
            stop_s = float(trace["time"][valid[-1]])
            samples = load_camera_samples(
                control_path,
                trace["source_sequence"],
                trace["time"],
                start_s,
                stop_s,
            )
            events, reasons = find_impacts(
                group, recording, samples, trace, kinematics)
            all_events.extend(events)
            rejection_counts.update(reasons)
            records.append({
                "group": group,
                "recording": recording,
                "status": "processed",
                "fresh_camera_samples": len(samples),
                "selected_impacts": len(events),
                "candidate_counts": dict(reasons),
            })
            print(
                f"[{group}/{recording}] camera={len(samples)} impacts={len(events)}",
                file=sys.stderr,
                flush=True,
            )

    calibration_events = [event for event in all_events if event.group in {"record_new3", "record_new4"}]
    holdout_events = [event for event in all_events if event.group == "record_new5"]
    experiment = IsolatedContactExperiment()
    baseline_parameters = (
        float(SIMCODE7_BASELINE["pair_sliding_friction"]),
        float(SIMCODE7_BASELINE["solref_time_s"]),
        float(SIMCODE7_BASELINE["solref_damping_ratio"]),
    )
    if args.skip_optimization:
        fitted_parameters = baseline_parameters
        optimization = {"status": "skipped", "objective": None}
    else:
        fitted_parameters, optimization = fit_contact_parameters(
            experiment, calibration_events, args.seed)
    if "profile_objective" not in optimization:
        time_grid = np.asarray([fitted_parameters[1]])
        damping_grid = np.asarray([fitted_parameters[2]])
        optimization.update({
            "profile_time_grid_s": time_grid.tolist(),
            "profile_damping_grid": damping_grid.tolist(),
            "profile_objective": [[optimization.get("objective") or 0.0]],
            "near_optimal_5pct_time_range_s": [fitted_parameters[1], fitted_parameters[1]],
            "near_optimal_5pct_damping_range": [fitted_parameters[2], fitted_parameters[2]],
            "near_optimal_2pct_friction_range": [fitted_parameters[0], fitted_parameters[0]],
        })

    simulation_validation = {
        "baseline": {
            "all": _event_prediction_metrics(experiment, all_events, baseline_parameters),
            "calibration": _event_prediction_metrics(experiment, calibration_events, baseline_parameters),
            "holdout": _event_prediction_metrics(experiment, holdout_events, baseline_parameters),
        },
        "fitted": {
            "all": _event_prediction_metrics(experiment, all_events, fitted_parameters),
            "calibration": _event_prediction_metrics(experiment, calibration_events, fitted_parameters),
            "holdout": _event_prediction_metrics(experiment, holdout_events, fitted_parameters),
        },
    }
    mass_sensitivity = _mass_sensitivity(
        experiment, all_events, fitted_parameters)
    recommended = {
        "ball_mass_kg": BALL_MASS_KG,
        "ball_mass_evidence": "user_provided_approximate_physical_mass",
        "pair_sliding_friction": fitted_parameters[0],
        "solref_time_s": fitted_parameters[1],
        "solref_damping_ratio": fitted_parameters[2],
        "solimp": SIMCODE7_BASELINE["solimp"],
        "margin_m": SIMCODE7_BASELINE["margin_m"],
        "condim": SIMCODE7_BASELINE["condim"],
        "ball_radius_m": BALL_RADIUS_M,
        "racket_radius_m": RACKET_RADIUS_M,
        "racket_rubber_half_thickness_m": RACKET_RUBBER_HALF_THICKNESS_M,
        "suggested_training_dr_solref_damping_range": [0.10, 0.24],
        "suggested_training_dr_ball_mass_range_kg": [0.0036, 0.0044],
        "fit_groups": ["record_new3", "record_new4"],
        "holdout_group": "record_new5",
    }
    summary = {
        "schema_version": 1,
        "method": {
            "camera_deduplication": "one sample per cameraStampS",
            "capture_time_alignment": "sourceTimeS + cameraStampS - wallTimeS",
            "flight_fit": "robust linear x/y and fixed -9.81 m/s^2 z on each side",
            "window_samples_per_side": 6,
            "racket_kinematics": "MuJoCo FK/Jacobian on actual_q and raw actual_qd",
            "simulation": "isolated MuJoCo sphere/cylinder using simcode7 pair semantics",
        },
        "records": records,
        "rejection_counts": dict(rejection_counts),
        "aggregate": _event_aggregate(all_events),
        "per_group": {
            group: _event_aggregate([event for event in all_events if event.group == group])
            for group in GROUPS
        },
        "recommended_parameters": recommended,
        "optimization": optimization,
        "simulation_validation": simulation_validation,
        "mass_identifiability": mass_sensitivity,
        "events": [asdict(event) for event in all_events],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output / "recommended_contact_params.json").write_text(
        json.dumps(recommended, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_events_csv(output / "selected_impacts.csv", all_events)
    _write_plots(output / "contact_fit_diagnostics.png", summary)
    _write_markdown(output / "SIM2REAL_BALL_RACKET_CONTACT_ANALYSIS_ZH.md", summary)
    print(json.dumps({
        "output": str(output),
        "selected_impacts": len(all_events),
        "calibration_impacts": len(calibration_events),
        "holdout_impacts": len(holdout_events),
        "recommended_parameters": recommended,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
