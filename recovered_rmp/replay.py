"""Replay real position targets through recovered RMP and MuJoCo control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .mujoco_adapter import single_dof_joint_indices
from .params import RMPParams, load_movax_rmp_config
from .trajectory import JointRMPPlanner

FloatArray = NDArray[np.float64]

RIGHT_ARM_JOINTS = tuple(f"RightArm-{index}" for index in range(7))
REST_POSTURE_DEG = {
    "LegWaist-0": 0.0,
    "LegWaist-1": 60.0,
    "LegWaist-2": -90.0,
    "LegWaist-3": 30.0,
    "LegWaist-4": 0.0,
    "LegWaist-5": 0.0,
    "LeftArm-0": -9.0,
    "LeftArm-1": -50.0,
    "LeftArm-2": -20.0,
    "LeftArm-3": -90.0,
    "LeftArm-4": -35.0,
    "LeftArm-5": 8.0,
    "LeftArm-6": -45.0,
    "Head-0": 0.0,
    "Head-1": 40.0,
}


@dataclass(frozen=True)
class RecordingTrace:
    source: Path
    time: FloatArray
    source_sequence: NDArray[np.int64]
    joint_time: FloatArray
    target_q: FloatArray
    actual_q: FloatArray
    actual_qd: FloatArray
    actual_qdd: FloatArray
    policy_mask: NDArray[np.bool_]
    median_dt: float


@dataclass(frozen=True)
class IdealRMPControllerConfig:
    """Ideal torque-controller parameters; there is no actuator dynamics block."""

    kp_scale: float = 1.0
    kd_scale: float = 1.0
    kp_scale_per_joint: tuple[float, ...] = ()
    kd_scale_per_joint: tuple[float, ...] = ()
    qdd_feedforward: float = 1.0
    inverse_dynamics_feedforward: float = 1.0
    torque_limit_nm: float = 0.0
    rmp_period_s: float = 0.001
    use_rmp_safety_bounds: bool = True
    use_rmp_jerk_limit: bool = False


@dataclass(frozen=True)
class ReplayMetrics:
    samples: int
    policy_samples: int
    acceleration_samples: int
    position_rms_rad: float
    position_rms_deg: float
    position_rms_per_joint_rad: tuple[float, ...]
    velocity_rms_rad_s: float
    velocity_rms_per_joint_rad_s: tuple[float, ...]
    acceleration_rms_rad_s2: float
    acceleration_rms_per_joint_rad_s2: tuple[float, ...]
    rmp_position_rms_rad: float
    maximum_position_error_rad: float


@dataclass(frozen=True)
class ReplayResult:
    trace: RecordingTrace
    rmp_q: FloatArray
    rmp_qd: FloatArray
    rmp_qdd: FloatArray
    sim_q: FloatArray
    sim_qd: FloatArray
    sim_qacc: FloatArray
    sim_qdd: FloatArray
    actual_qdd_observed: FloatArray
    acceleration_valid_mask: NDArray[np.bool_]
    applied_tau: FloatArray
    metrics: ReplayMetrics
    controller: IdealRMPControllerConfig
    rmp_params: RMPParams
    kp: FloatArray
    kd: FloatArray


def _as_joint_vector(value: object) -> FloatArray | None:
    if not isinstance(value, list) or len(value) != 7:
        return None
    result = np.asarray(value, dtype=float)
    return result if np.all(np.isfinite(result)) else None


def _row_sample(
    row: dict[str, object],
) -> tuple[
    float,
    int,
    float,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    bool,
] | None:
    command = row.get("command")
    ros = row.get("ros")
    if not isinstance(command, dict) or not isinstance(ros, dict):
        return None
    if not bool(command.get("published")):
        return None
    target = _as_joint_vector(command.get("position"))
    actual = _as_joint_vector(ros.get("jointPosition"))
    velocity = _as_joint_vector(ros.get("jointVelocity"))
    acceleration = _as_joint_vector(ros.get("jointAcceleration"))
    timestamp = row.get("sourceTimeS", row.get("time"))
    sequence = row.get("sourceSeq", row.get("tick", -1))
    if target is None or actual is None or not isinstance(timestamp, (int, float)):
        return None
    if velocity is None:
        velocity = np.zeros(7, dtype=float)
    if acceleration is None:
        acceleration = np.zeros(7, dtype=float)
    joint_timestamp = ros.get("jointSampleStampS", timestamp)
    if not isinstance(joint_timestamp, (int, float)) or not np.isfinite(joint_timestamp):
        joint_timestamp = timestamp
    return (
        float(timestamp),
        int(sequence) if isinstance(sequence, (int, float)) else -1,
        float(joint_timestamp),
        target,
        actual,
        velocity,
        acceleration,
        bool(row.get("policyExecuted")),
    )


def load_control_recording(path: str | Path, *, segment: str = "longest") -> RecordingTrace:
    """Load a JSONL log and select one monotonic, contiguous command segment.

    Several ``record_new3/4`` files contain a recorder restart appended to the
    same file.  Replaying across that discontinuity would invent a target jump,
    so the default selects the longest monotonic segment.
    """

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "control_samples.jsonl"
    if not source.exists():
        raise FileNotFoundError(source)

    Sample = tuple[
        float,
        int,
        float,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        bool,
    ]
    groups: list[list[Sample]] = []
    current: list[Sample] = []
    previous_time: float | None = None
    previous_sequence: int | None = None
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            sample = _row_sample(row)
            if sample is None:
                continue
            timestamp, sequence = sample[0], sample[1]
            discontinuity = previous_time is not None and (
                timestamp <= previous_time - 1.0e-9
                or timestamp - previous_time > 0.100
                or (sequence >= 0 and previous_sequence is not None and sequence <= previous_sequence)
            )
            if discontinuity and current:
                groups.append(current)
                current = []
            current.append(sample)
            previous_time = timestamp
            previous_sequence = sequence
    if current:
        groups.append(current)
    if not groups:
        raise ValueError(f"no published seven-joint samples in {source}")
    if segment == "longest":
        samples = max(groups, key=len)
    elif segment == "first":
        samples = groups[0]
    elif segment == "last":
        samples = groups[-1]
    else:
        raise ValueError("segment must be 'longest', 'first', or 'last'")
    if len(samples) < 2:
        raise ValueError("recording segment must contain at least two samples")

    time = np.asarray([sample[0] for sample in samples], dtype=float)
    positive_dt = np.diff(time)
    positive_dt = positive_dt[positive_dt > 0.0]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else 0.005
    return RecordingTrace(
        source=source,
        time=time - time[0],
        source_sequence=np.asarray([sample[1] for sample in samples], dtype=np.int64),
        joint_time=np.asarray([sample[2] for sample in samples], dtype=float),
        target_q=np.stack([sample[3] for sample in samples]),
        actual_q=np.stack([sample[4] for sample in samples]),
        actual_qd=np.stack([sample[5] for sample in samples]),
        actual_qdd=np.stack([sample[6] for sample in samples]),
        policy_mask=np.asarray([sample[7] for sample in samples], dtype=bool),
        median_dt=median_dt,
    )


def forward_velocity_acceleration(
    velocity: FloatArray,
    time: FloatArray,
    source_sequence: NDArray[np.int64],
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Match the recorder's aligned joint-acceleration measurement operator.

    Acceleration belongs to the current tick and is the forward difference to
    the next contiguous source tick. Repeated source samples, clock gaps, and
    the final row are deliberately invalid, exactly as in the real/sim report.
    """

    velocity = np.asarray(velocity, dtype=float)
    time = np.asarray(time, dtype=float)
    source_sequence = np.asarray(source_sequence, dtype=np.int64)
    if velocity.ndim != 2 or time.shape != (velocity.shape[0],):
        raise ValueError("velocity must be [samples, joints] with one time per sample")
    if source_sequence.shape != time.shape:
        raise ValueError("source_sequence must contain one value per sample")
    acceleration = np.full_like(velocity, np.nan, dtype=float)
    valid = np.zeros(time.shape, dtype=bool)
    if time.size < 2:
        return acceleration, valid
    dt = np.diff(time)
    adjacent = source_sequence[1:] == source_sequence[:-1] + 1
    usable = adjacent & np.isfinite(dt) & (dt >= 1.0e-4) & (dt <= 0.10)
    indices = np.flatnonzero(usable)
    acceleration[indices] = (
        velocity[indices + 1] - velocity[indices]
    ) / dt[indices, None]
    valid[indices] = np.all(np.isfinite(acceleration[indices]), axis=1)
    return acceleration, valid


def _disable_right_arm_actuators(model: object, actuator_ids: Iterable[int]) -> tuple[FloatArray, FloatArray]:
    ids = np.asarray(tuple(actuator_ids), dtype=int)
    kp = np.asarray(model.actuator_gainprm[ids, 0], dtype=float).copy()
    kd = -np.asarray(model.actuator_biasprm[ids, 2], dtype=float).copy()
    # The arm is driven through qfrc_applied below.  Removing gain and bias is
    # the explicit "no actuator model" path: no FOPDT, delay, motor filter, or
    # hidden XML position servo remains between RMP and the ideal controller.
    model.actuator_gainprm[ids, :] = 0.0
    model.actuator_biasprm[ids, :] = 0.0
    return kp, kd


def replay_in_mujoco(
    trace: RecordingTrace,
    model_path: str | Path,
    movax_config_path: str | Path,
    controller: IdealRMPControllerConfig = IdealRMPControllerConfig(),
    *,
    rmp_params: RMPParams | None = None,
) -> ReplayResult:
    """Execute target -> RMP q/qd/qdd -> ideal PD -> MuJoCo dynamics."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MuJoCo is required for replay") from exc

    model = mujoco.MjModel.from_xml_path(str(Path(model_path).expanduser().resolve()))
    data = mujoco.MjData(model)
    qpos_indices, dof_indices = single_dof_joint_indices(model, RIGHT_ARM_JOINTS)
    actuator_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in RIGHT_ARM_JOINTS],
        dtype=int,
    )
    if np.any(actuator_ids < 0):
        raise KeyError("one or more RightArm actuators are missing from the model")
    xml_kp = np.asarray(model.actuator_gainprm[actuator_ids, 0], dtype=float).copy()
    xml_kd = -np.asarray(model.actuator_biasprm[actuator_ids, 2], dtype=float).copy()
    kp_scale = np.full(7, float(controller.kp_scale), dtype=float)
    kd_scale = np.full(7, float(controller.kd_scale), dtype=float)
    if controller.kp_scale_per_joint:
        kp_scale = np.asarray(controller.kp_scale_per_joint, dtype=float)
    if controller.kd_scale_per_joint:
        kd_scale = np.asarray(controller.kd_scale_per_joint, dtype=float)
    if kp_scale.shape != (7,) or kd_scale.shape != (7,):
        raise ValueError("per-joint PD scales must contain seven values")
    if np.any(kp_scale <= 0.0) or np.any(kd_scale <= 0.0):
        raise ValueError("PD scales must be positive")
    kp = xml_kp * kp_scale
    kd = xml_kd * kd_scale
    params, limits = load_movax_rmp_config(movax_config_path, "RightArm")
    if rmp_params is not None:
        params = rmp_params

    rmp_dt = float(controller.rmp_period_s)
    if rmp_dt <= 0.0:
        raise ValueError("rmp_period_s must be positive")
    model_dt = float(model.opt.timestep)
    if not np.isclose(model_dt, rmp_dt, rtol=0.0, atol=1.0e-10):
        model.opt.timestep = rmp_dt
        model_dt = rmp_dt

    # Reproduce simcode5's stable full-body warm posture before replacing the
    # right-arm actuator interface.  Starting every non-arm joint at zero puts
    # the leg inside the floor and destabilizes the floating planar base.
    for name, degrees in REST_POSTURE_DEG.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id >= 0:
            data.qpos[int(model.jnt_qposadr[joint_id])] = np.deg2rad(degrees)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = np.deg2rad(degrees)
    data.qpos[qpos_indices] = trace.actual_q[0]
    data.qvel[dof_indices] = 0.0
    data.ctrl[actuator_ids] = trace.actual_q[0]
    for _ in range(700):
        mujoco.mj_step(model, data)
    data.qpos[qpos_indices] = trace.actual_q[0]
    data.qvel[dof_indices] = trace.actual_qd[0]
    _disable_right_arm_actuators(model, actuator_ids)
    mujoco.mj_forward(model, data)

    planner = JointRMPPlanner(
        limits,
        params,
        rmp_dt,
        use_safety_bounds=controller.use_rmp_safety_bounds,
        use_jerk_limit=controller.use_rmp_jerk_limit,
    )
    planner.reset(
        trace.actual_q[0],
        velocity=trace.actual_qd[0],
        acceleration=trace.actual_qdd[0],
        target_position=trace.target_q[0],
    )

    count = trace.time.size
    rmp_q = np.empty((count, 7), dtype=float)
    rmp_qd = np.empty_like(rmp_q)
    rmp_qdd = np.empty_like(rmp_q)
    sim_q = np.empty_like(rmp_q)
    sim_qd = np.empty_like(rmp_q)
    sim_qacc = np.empty_like(rmp_q)
    applied_tau = np.empty_like(rmp_q)
    last_tau = np.zeros(7, dtype=float)
    full_mass = np.empty((model.nv, model.nv), dtype=float)

    previous_time = trace.time[0]
    last_command = None
    for sample_index in range(count):
        sim_q[sample_index] = data.qpos[qpos_indices]
        sim_qd[sample_index] = data.qvel[dof_indices]
        sim_qacc[sample_index] = data.qacc[dof_indices]
        applied_tau[sample_index] = last_tau
        interval = trace.median_dt if sample_index == 0 else trace.time[sample_index] - previous_time
        previous_time = trace.time[sample_index]
        servo_steps = max(1, int(round(interval / rmp_dt)))
        for servo_index in range(servo_steps):
            last_command = planner.step(
                trace.target_q[sample_index],
                target_updated=servo_index == 0,
            )
            mujoco.mj_forward(model, data)
            try:
                # MuJoCo 3.10+ obtains the sparse inertia from MjData.
                mujoco.mj_fullM(model, data, full_mass)
            except TypeError:  # pragma: no cover - older MuJoCo Python ABI
                mujoco.mj_fullM(model, full_mass, data.qM)
            desired_full_acc = np.zeros(model.nv, dtype=float)
            desired_full_acc[dof_indices] = last_command.acceleration
            inverse_dynamics = full_mass @ desired_full_acc
            inverse_dynamics += np.asarray(data.qfrc_bias) - np.asarray(data.qfrc_passive)
            position_error = last_command.position - data.qpos[qpos_indices]
            velocity_error = last_command.velocity - data.qvel[dof_indices]
            dynamic_acc = (full_mass @ desired_full_acc)[dof_indices]
            bias = inverse_dynamics[dof_indices] - dynamic_acc
            last_tau = (
                kp * position_error
                + kd * velocity_error
                + float(controller.qdd_feedforward) * dynamic_acc
                + float(controller.inverse_dynamics_feedforward) * bias
            )
            limit = float(controller.torque_limit_nm)
            if limit > 0.0:
                last_tau = np.clip(last_tau, -limit, limit)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_indices] = last_tau
            mujoco.mj_step(model, data)
        if last_command is None:  # pragma: no cover - servo_steps is at least one
            raise RuntimeError("RMP did not advance")
        rmp_q[sample_index] = last_command.position
        rmp_qd[sample_index] = last_command.velocity
        rmp_qdd[sample_index] = last_command.acceleration

    mask = trace.policy_mask if np.any(trace.policy_mask) else np.ones(count, dtype=bool)
    position_error = sim_q[mask] - trace.actual_q[mask]
    velocity_error = sim_qd[mask] - trace.actual_qd[mask]
    actual_qdd_observed, actual_qdd_valid = forward_velocity_acceleration(
        trace.actual_qd, trace.joint_time, trace.source_sequence
    )
    sim_qdd, sim_qdd_valid = forward_velocity_acceleration(
        sim_qd, trace.time, trace.source_sequence
    )
    acceleration_mask = mask & actual_qdd_valid & sim_qdd_valid
    if not np.any(acceleration_mask):
        raise ValueError("recording has no valid aligned acceleration samples")
    acceleration_error = (
        sim_qdd[acceleration_mask] - actual_qdd_observed[acceleration_mask]
    )
    rmp_error = rmp_q[mask] - trace.actual_q[mask]
    per_position = np.sqrt(np.mean(position_error**2, axis=0))
    per_velocity = np.sqrt(np.mean(velocity_error**2, axis=0))
    per_acceleration = np.sqrt(np.mean(acceleration_error**2, axis=0))
    position_rms = float(np.sqrt(np.mean(position_error**2)))
    metrics = ReplayMetrics(
        samples=count,
        policy_samples=int(np.count_nonzero(mask)),
        acceleration_samples=int(np.count_nonzero(acceleration_mask)),
        position_rms_rad=position_rms,
        position_rms_deg=float(np.rad2deg(position_rms)),
        position_rms_per_joint_rad=tuple(float(v) for v in per_position),
        velocity_rms_rad_s=float(np.sqrt(np.mean(velocity_error**2))),
        velocity_rms_per_joint_rad_s=tuple(float(v) for v in per_velocity),
        acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration_error**2))),
        acceleration_rms_per_joint_rad_s2=tuple(
            float(v) for v in per_acceleration
        ),
        rmp_position_rms_rad=float(np.sqrt(np.mean(rmp_error**2))),
        maximum_position_error_rad=float(np.max(np.abs(position_error))),
    )
    return ReplayResult(
        trace=trace,
        rmp_q=rmp_q,
        rmp_qd=rmp_qd,
        rmp_qdd=rmp_qdd,
        sim_q=sim_q,
        sim_qd=sim_qd,
        sim_qacc=sim_qacc,
        sim_qdd=sim_qdd,
        actual_qdd_observed=actual_qdd_observed,
        acceleration_valid_mask=acceleration_mask,
        applied_tau=applied_tau,
        metrics=metrics,
        controller=controller,
        rmp_params=params,
        kp=kp,
        kd=kd,
    )


def save_replay_result(result: ReplayResult, output_directory: str | Path) -> tuple[Path, Path]:
    """Write a compact numerical trace and a human-readable JSON report."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "rmp_replay_trace.npz"
    report_path = output / "rmp_replay_report.json"
    np.savez_compressed(
        trace_path,
        time=result.trace.time,
        source_sequence=result.trace.source_sequence,
        joint_time=result.trace.joint_time,
        target_q=result.trace.target_q,
        actual_q=result.trace.actual_q,
        actual_qd=result.trace.actual_qd,
        actual_qdd=result.actual_qdd_observed,
        actual_qdd_raw=result.trace.actual_qdd,
        policy_mask=result.trace.policy_mask,
        rmp_q=result.rmp_q,
        rmp_qd=result.rmp_qd,
        rmp_qdd=result.rmp_qdd,
        sim_q=result.sim_q,
        sim_qd=result.sim_qd,
        sim_qacc=result.sim_qacc,
        sim_qdd=result.sim_qdd,
        acceleration_valid_mask=result.acceleration_valid_mask,
        applied_tau=result.applied_tau,
    )
    report = {
        "recording": str(result.trace.source),
        "control_chain": "recorded q target -> recovered RMP q/qd/qdd -> ideal torque PD -> MuJoCo",
        "actuator_model_enabled": False,
        "acceleration_measurement": (
            "forward velocity difference; real dt=ros.jointSampleStampS, "
            "sim dt=sourceTimeS, contiguous sourceSeq only"
        ),
        "recording_median_dt_s": result.trace.median_dt,
        "controller": asdict(result.controller),
        "rmp_params": asdict(result.rmp_params),
        "kp": result.kp.tolist(),
        "kd": result.kd.tolist(),
        "metrics": asdict(result.metrics),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return trace_path, report_path
