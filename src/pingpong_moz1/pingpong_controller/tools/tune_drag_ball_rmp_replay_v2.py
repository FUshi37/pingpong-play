#!/usr/bin/env python3
"""Second-stage per-joint recovered-RMP identification for drag-ball data."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_rmp_tuning_v2_mpl")

import numpy as np

import analyze_drag_ball_sim2real as drag
import analyze_valid_juggle_sim2real as base


Q_SCALE_RAD = 0.01
DQ_SCALE_RAD_S = 0.20
ONES = (1.0,) * 7
ZEROS = (0.0,) * 7


@dataclass(frozen=True)
class CandidateV2:
    jnt_kp_scale_per_joint: tuple[float, ...] = ONES
    jnt_kd_scale_per_joint: tuple[float, ...] = ONES
    output_filter_length: int = 35
    target_filter_length: int = 10
    target_update_delay_steps: int = 4
    output_delay_steps_per_joint: tuple[int, ...] = (
        14, 13, 11, 14, 13, 13, 13)
    joint_velocity_feedforward: float = 0.5
    joint_acceleration_feedforward: float = 0.0
    estimator_process_scale: float = 1.0
    estimator_measure_scale: float = 1.0
    acceleration_limit_scale_per_joint: tuple[float, ...] = ONES
    dof_damping_scale_per_joint: tuple[float, ...] = ONES
    dof_armature_scale_per_joint: tuple[float, ...] = ONES
    acceleration_feedforward_s2_per_joint: tuple[float, ...] = ZEROS
    actuator_kp_scale_per_joint: tuple[float, ...] = ONES
    actuator_kv_scale_per_joint: tuple[float, ...] = ONES

    def key(self) -> tuple[Any, ...]:
        value = asdict(self)
        return tuple(
            tuple(round(float(item), 10) for item in current)
            if isinstance(current, tuple)
            else round(float(current), 10)
            if isinstance(current, float)
            else int(current)
            for current in value.values()
        )


@dataclass
class ReplayV2:
    candidate: CandidateV2
    q: np.ndarray
    dq: np.ndarray
    metrics: dict[str, Any]


def _vector7(value: Iterable[float], name: str) -> np.ndarray:
    output = np.asarray(tuple(value), dtype=np.float64)
    if output.shape != (7,) or not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must contain seven finite values")
    return output


def apply_candidate_v2_to_mirror(
    mirror: Any,
    candidate: CandidateV2,
    *,
    base_state: dict[str, np.ndarray | Any] | None = None,
) -> None:
    """Apply a V2 candidate; scales are relative to the checkpoint plant."""
    from recovered_rmp import JointLimits, JointRMPPlanner

    if mirror.rmp_planner is None:
        raise RuntimeError("recovered-RMP planner is unavailable")
    state = base_state or {
        "params": mirror.rmp_planner.params,
        "limits": mirror.rmp_planner.limits,
        "dof_damping": mirror.model.dof_damping[mirror.arm_vadr].copy(),
        "dof_armature": mirror.model.dof_armature[mirror.arm_vadr].copy(),
        "acceleration_feedforward_s2": (
            mirror.rmp_acceleration_feedforward_s2.copy()),
        "actuator_gainprm_0": mirror.model.actuator_gainprm[
            mirror.arm_aids, 0].copy(),
        "actuator_biasprm_1": mirror.model.actuator_biasprm[
            mirror.arm_aids, 1].copy(),
        "actuator_biasprm_2": mirror.model.actuator_biasprm[
            mirror.arm_aids, 2].copy(),
    }
    params0 = state["params"]
    limits0 = state["limits"]
    kp = _vector7(candidate.jnt_kp_scale_per_joint, "jnt_kp_scale_per_joint")
    kd = _vector7(candidate.jnt_kd_scale_per_joint, "jnt_kd_scale_per_joint")
    acceleration = _vector7(
        candidate.acceleration_limit_scale_per_joint,
        "acceleration_limit_scale_per_joint")
    if np.any(kp <= 0.0) or np.any(kd <= 0.0) or np.any(acceleration <= 0.0):
        raise ValueError("RMP gain and acceleration scales must be positive")
    params = replace(
        params0,
        jnt_kp=np.asarray(params0.jnt_kp, dtype=np.float64) * kp,
        jnt_kd=np.asarray(params0.jnt_kd, dtype=np.float64) * kd,
        filter_length=int(candidate.output_filter_length),
        target_filter_length=int(candidate.target_filter_length),
        joint_vel_ffd=float(candidate.joint_velocity_feedforward),
        joint_acc_ffd=float(candidate.joint_acceleration_feedforward),
        estimator_joint_process_var=(
            float(params0.estimator_joint_process_var)
            * float(candidate.estimator_process_scale)),
        estimator_joint_measure_var=(
            float(params0.estimator_joint_measure_var)
            * float(candidate.estimator_measure_scale)),
    )
    limits = JointLimits(
        q_min=limits0.q_min,
        q_max=limits0.q_max,
        qd_max=limits0.qd_max,
        qdd_max=np.asarray(limits0.qdd_max) * acceleration,
        qddd_max=(
            None if limits0.qddd_max is None
            else np.asarray(limits0.qddd_max) * acceleration),
    )
    mirror.rmp_planner = JointRMPPlanner(
        limits,
        params,
        float(mirror.model.opt.timestep),
        use_safety_bounds=mirror.rmp_planner.use_safety_bounds,
        use_jerk_limit=mirror.rmp_planner.use_jerk_limit,
    )
    delays = np.asarray(candidate.output_delay_steps_per_joint, dtype=np.float64)
    if delays.shape != (7,) or np.any(delays < 0):
        raise ValueError("output delays must be seven nonnegative values")
    mirror.rmp_output_delay_steps = delays.copy()
    mirror.rmp_target_update_delay_steps = int(
        candidate.target_update_delay_steps)
    if not 0 <= mirror.rmp_target_update_delay_steps < mirror.substeps:
        raise ValueError("target update delay must lie within one control tick")
    damping = _vector7(
        candidate.dof_damping_scale_per_joint,
        "dof_damping_scale_per_joint")
    armature = _vector7(
        candidate.dof_armature_scale_per_joint,
        "dof_armature_scale_per_joint")
    if np.any(damping <= 0.0) or np.any(armature <= 0.0):
        raise ValueError("damping and armature scales must be positive")
    mirror.model.dof_damping[mirror.arm_vadr] = (
        np.asarray(state["dof_damping"]) * damping)
    mirror.model.dof_armature[mirror.arm_vadr] = (
        np.asarray(state["dof_armature"]) * armature)
    mirror.rmp_acceleration_feedforward_s2 = _vector7(
        candidate.acceleration_feedforward_s2_per_joint,
        "acceleration_feedforward_s2_per_joint")
    actuator_kp = _vector7(
        candidate.actuator_kp_scale_per_joint,
        "actuator_kp_scale_per_joint")
    actuator_kv = _vector7(
        candidate.actuator_kv_scale_per_joint,
        "actuator_kv_scale_per_joint")
    if np.any(actuator_kp <= 0.0) or np.any(actuator_kv <= 0.0):
        raise ValueError("actuator Kp/Kv scales must be positive")
    gainprm_0 = np.asarray(state.get(
        "actuator_gainprm_0",
        mirror.model.actuator_gainprm[mirror.arm_aids, 0]), dtype=np.float64)
    biasprm_1 = np.asarray(state.get(
        "actuator_biasprm_1",
        mirror.model.actuator_biasprm[mirror.arm_aids, 1]), dtype=np.float64)
    biasprm_2 = np.asarray(state.get(
        "actuator_biasprm_2",
        mirror.model.actuator_biasprm[mirror.arm_aids, 2]), dtype=np.float64)
    mirror.model.actuator_gainprm[mirror.arm_aids, 0] = (
        gainprm_0 * actuator_kp)
    mirror.model.actuator_biasprm[mirror.arm_aids, 1] = (
        biasprm_1 * actuator_kp)
    mirror.model.actuator_biasprm[mirror.arm_aids, 2] = (
        biasprm_2 * actuator_kv)
    mirror.rmp_controller_kp = gainprm_0 * actuator_kp
    mirror.rmp_controller_kv = -biasprm_2 * actuator_kv


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else math.nan


def _window(
    real_q: np.ndarray,
    real_dq: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    q_error = q[mask] - real_q[mask]
    dq_error = dq[mask] - real_dq[mask]
    q_joint = np.sqrt(np.mean(q_error * q_error, axis=0))
    dq_joint = np.sqrt(np.mean(dq_error * dq_error, axis=0))
    joint_score = np.sqrt(
        0.5 * (q_joint / Q_SCALE_RAD) ** 2
        + 0.5 * (dq_joint / DQ_SCALE_RAD_S) ** 2)
    q_rms = _rms(q_error)
    dq_rms = _rms(dq_error)
    return {
        "samples": int(np.count_nonzero(mask)),
        "q_rms_rad": q_rms,
        "dq_rms_rad_s": dq_rms,
        "q_per_joint_rms_rad": q_joint.tolist(),
        "dq_per_joint_rms_rad_s": dq_joint.tolist(),
        "joint_normalized_score": joint_score.tolist(),
        "normalized_score": math.sqrt(
            0.5 * (q_rms / Q_SCALE_RAD) ** 2
            + 0.5 * (dq_rms / DQ_SCALE_RAD_S) ** 2),
    }


def _metrics(
    elapsed: np.ndarray,
    real_q: np.ndarray,
    real_dq: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
) -> dict[str, Any]:
    windows = {
        "full": np.ones(len(elapsed), dtype=bool),
        "post_warmup": elapsed >= 1.0,
        "train_1_to_9_s": (elapsed >= 1.0) & (elapsed < 9.0),
        "validation_9_to_14_s": (elapsed >= 9.0) & (elapsed < 14.0),
        "held_out_14_s_to_end": elapsed >= 14.0,
    }
    output = {
        name: _window(real_q, real_dq, q, dq, mask)
        for name, mask in windows.items()
        if np.any(mask)
    }
    output["development_score"] = (
        0.65 * output["train_1_to_9_s"]["normalized_score"]
        + 0.35 * output["validation_9_to_14_s"]["normalized_score"])
    train_joint = np.asarray(
        output["train_1_to_9_s"]["joint_normalized_score"])
    validation_joint = np.asarray(
        output["validation_9_to_14_s"]["joint_normalized_score"])
    output["joint_development_score"] = (
        0.65 * train_joint + 0.35 * validation_joint).tolist()
    return output


class TunerV2:
    def __init__(
        self,
        recording: Path,
        aligned: list[dict[str, Any]],
        real_q: np.ndarray,
        real_dq: np.ndarray,
        commands: np.ndarray,
        elapsed: np.ndarray,
        output: Path,
        *,
        reset_log: bool = True,
    ) -> None:
        self.aligned = aligned
        self.real_q = real_q
        self.real_dq = real_dq
        self.commands = commands
        self.elapsed = elapsed
        self.output = output
        config = drag._mirror_config(
            recording, base.DEFAULT_CHECKPOINT, base.DEFAULT_XML)
        from pingpong_controller.sim_mirror import RealBallSimMirror

        self.mirror = RealBallSimMirror(config)
        drag._disable_ball_collisions(self.mirror)
        if self.mirror.rmp_planner is None:
            raise RuntimeError("recovered-RMP planner is unavailable")
        self.base_state = {
            "params": self.mirror.rmp_planner.params,
            "limits": self.mirror.rmp_planner.limits,
            "dof_damping": self.mirror.model.dof_damping[
                self.mirror.arm_vadr].copy(),
            "dof_armature": self.mirror.model.dof_armature[
                self.mirror.arm_vadr].copy(),
            "acceleration_feedforward_s2": (
                self.mirror.rmp_acceleration_feedforward_s2.copy()),
            "actuator_gainprm_0": self.mirror.model.actuator_gainprm[
                self.mirror.arm_aids, 0].copy(),
            "actuator_biasprm_1": self.mirror.model.actuator_biasprm[
                self.mirror.arm_aids, 1].copy(),
            "actuator_biasprm_2": self.mirror.model.actuator_biasprm[
                self.mirror.arm_aids, 2].copy(),
        }
        self.cache: dict[tuple[Any, ...], ReplayV2] = {}
        self.evaluations = 0
        self.log_path = output / "candidate_search.jsonl"
        if reset_log:
            self.log_path.write_text("", encoding="utf-8")

    def close(self) -> None:
        self.mirror.close()

    def evaluate(
        self,
        candidate: CandidateV2,
        *,
        search_only: bool = True,
    ) -> ReplayV2:
        count = int(np.searchsorted(self.elapsed, 14.0, side="left"))
        if not search_only:
            count = len(self.elapsed)
        cache_key = (candidate.key(), count)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        apply_candidate_v2_to_mirror(
            self.mirror, candidate, base_state=self.base_state)
        drag._reset_replay_mirror(
            self.mirror,
            self.real_q[0],
            self.real_dq[0],
            self.aligned[0],
        )
        drag._park_replay_ball(self.mirror)
        q_rows: list[np.ndarray] = []
        dq_rows: list[np.ndarray] = []
        for command in self.commands[:count]:
            drag._park_replay_ball(self.mirror)
            q_rows.append(
                self.mirror.data.qpos[self.mirror.arm_qadr]
                .astype(np.float64).copy())
            dq_rows.append(
                self.mirror.data.qvel[self.mirror.arm_vadr]
                .astype(np.float64).copy())
            self.mirror._step_recovered_rmp(command.astype(np.float32))
        q = np.asarray(q_rows)
        dq = np.asarray(dq_rows)
        replay = ReplayV2(
            candidate,
            q,
            dq,
            _metrics(
                self.elapsed[:count],
                self.real_q[:count],
                self.real_dq[:count],
                q,
                dq,
            ),
        )
        self.cache[cache_key] = replay
        self.evaluations += 1
        row = {
            "evaluation": self.evaluations,
            "search_only": search_only,
            "development_score": replay.metrics["development_score"],
            "candidate": asdict(candidate),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"eval={self.evaluations:03d} "
            f"dev={replay.metrics['development_score']:.7f} "
            f"q={replay.metrics['post_warmup']['q_rms_rad']:.6f} "
            f"dq={replay.metrics['post_warmup']['dq_rms_rad_s']:.5f}",
            flush=True,
        )
        return replay


def _replace_joint(
    candidate: CandidateV2,
    field: str,
    joint: int,
    value: float | int,
) -> CandidateV2:
    values = list(getattr(candidate, field))
    values[joint] = value
    if field == "output_delay_steps_per_joint":
        values = [int(item) for item in values]
    else:
        values = [float(item) for item in values]
    return replace(candidate, **{field: tuple(values)})


def _unique(values: Iterable[float | int]) -> list[float | int]:
    output: list[float | int] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _per_joint_sweep(
    tuner: TunerV2,
    initial: ReplayV2,
    field: str,
    value_fn: Any,
    decisions: list[dict[str, Any]],
) -> ReplayV2:
    current = initial
    for joint in range(7):
        before = current
        current_value = getattr(current.candidate, field)[joint]
        candidates = [
            tuner.evaluate(_replace_joint(
                current.candidate, field, joint, value))
            for value in _unique(value_fn(current_value))
            if field != "output_delay_steps_per_joint" or int(value) >= 0
        ]
        winner = min(
            candidates,
            key=lambda item: item.metrics["development_score"])
        if winner.metrics["development_score"] < (
            current.metrics["development_score"] - 1.0e-7
        ):
            current = winner
        decisions.append({
            "stage": field,
            "joint": joint + 1,
            "before": current_value,
            "selected": getattr(current.candidate, field)[joint],
            "before_development_score": before.metrics["development_score"],
            "selected_development_score": current.metrics["development_score"],
        })
        print(
            f"select {field} J{joint + 1}="
            f"{getattr(current.candidate, field)[joint]} "
            f"dev={current.metrics['development_score']:.7f}",
            flush=True,
        )
    return current


def _global_sweep(
    tuner: TunerV2,
    initial: ReplayV2,
    field: str,
    values: Iterable[float | int],
    decisions: list[dict[str, Any]],
) -> ReplayV2:
    before = initial
    candidates = [
        tuner.evaluate(replace(initial.candidate, **{field: value}))
        for value in _unique(values)
    ]
    winner = min(candidates, key=lambda item: item.metrics["development_score"])
    current = (
        winner if winner.metrics["development_score"]
        < initial.metrics["development_score"] - 1.0e-7
        else initial)
    decisions.append({
        "stage": field,
        "before": getattr(before.candidate, field),
        "selected": getattr(current.candidate, field),
        "before_development_score": before.metrics["development_score"],
        "selected_development_score": current.metrics["development_score"],
    })
    print(
        f"select {field}={getattr(current.candidate, field)} "
        f"dev={current.metrics['development_score']:.7f}",
        flush=True,
    )
    return current


def _proposal(tuner: TunerV2, candidate: CandidateV2) -> dict[str, Any]:
    cfg = tuner.mirror.checkpoint_env_cfg
    base_kp = np.asarray(cfg.get(
        "recovered_rmp_jnt_kp_scale_per_joint", ONES), dtype=np.float64)
    base_kd = np.asarray(cfg.get(
        "recovered_rmp_jnt_kd_scale_per_joint",
        (1.0, 1.0, 0.92, 1.0, 1.0, 1.0, 1.0)), dtype=np.float64)
    base_damping = np.asarray(cfg.get(
        "recovered_rmp_dof_damping_scale_per_joint",
        (0.6084, 0.9984, 0.9984, 1.92, 0.58, 1.17, 0.78)),
        dtype=np.float64)
    base_armature = np.asarray(cfg.get(
        "recovered_rmp_dof_armature_scale_per_joint",
        (0.4524, 1.65, 0.92, 1.22, 0.58, 0.5336, 0.58)),
        dtype=np.float64)
    base_acceleration = np.asarray(cfg.get(
        "arm_acc_limit_deg_s2",
        (1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0)),
        dtype=np.float64)
    return {
        "recovered_rmp_jnt_kp_scale_per_joint": (
            base_kp * np.asarray(candidate.jnt_kp_scale_per_joint)).tolist(),
        "recovered_rmp_jnt_kd_scale_per_joint": (
            base_kd * np.asarray(candidate.jnt_kd_scale_per_joint)).tolist(),
        "recovered_rmp_output_filter_length": candidate.output_filter_length,
        "recovered_rmp_target_filter_length": candidate.target_filter_length,
        "recovered_rmp_target_update_delay_steps": (
            candidate.target_update_delay_steps),
        "recovered_rmp_output_delay_steps_per_joint": list(
            candidate.output_delay_steps_per_joint),
        "recovered_rmp_joint_velocity_feedforward": (
            candidate.joint_velocity_feedforward),
        "recovered_rmp_joint_acceleration_feedforward": (
            candidate.joint_acceleration_feedforward),
        "recovered_rmp_estimator_joint_process_var": (
            float(tuner.base_state["params"].estimator_joint_process_var)
            * candidate.estimator_process_scale),
        "recovered_rmp_estimator_joint_measure_var": (
            float(tuner.base_state["params"].estimator_joint_measure_var)
            * candidate.estimator_measure_scale),
        "arm_acc_limit_deg_s2": (
            base_acceleration
            * np.asarray(candidate.acceleration_limit_scale_per_joint)).tolist(),
        "recovered_rmp_dof_damping_scale_per_joint": (
            base_damping
            * np.asarray(candidate.dof_damping_scale_per_joint)).tolist(),
        "recovered_rmp_dof_armature_scale_per_joint": (
            base_armature
            * np.asarray(candidate.dof_armature_scale_per_joint)).tolist(),
        "recovered_rmp_acceleration_feedforward_s2": list(
            candidate.acceleration_feedforward_s2_per_joint),
        "recovered_rmp_actuator_kp_scale_per_joint": list(
            candidate.actuator_kp_scale_per_joint),
        "recovered_rmp_actuator_kv_scale_per_joint": list(
            candidate.actuator_kv_scale_per_joint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording", nargs="?", type=Path,
        default=base.ROOT / (
            "src/pingpong_moz1/pingpong_controller/outputs/"
            "interactive_recordings/probe_recording_20260829_180205"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--low-level-only", action="store_true",
        help="resume the selected V2 candidate and tune damping/armature",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording = args.recording.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else recording / "analysis" / "v91_rmp_tuning_drag_ball_v2")
    output.mkdir(parents=True, exist_ok=True)
    _stage, aligned, _excluded = base.load_alignment(recording)
    elapsed = np.asarray([
        item["time"] - aligned[0]["time"] for item in aligned])
    real_q = base.aligned_matrix(aligned, "real_q")
    real_dq = base.aligned_matrix(aligned, "real_dq")
    commands = np.asarray([
        drag._published_command(item["real"]) for item in aligned])

    v1 = json.loads((
        recording / "analysis" / "v91_rmp_tuning_drag_ball"
        / "rmp_tuning_results.json").read_text(encoding="utf-8"))
    proposal = v1["proposed_env_cfg_override"]
    initial_candidate = CandidateV2(
        output_filter_length=int(
            proposal["recovered_rmp_output_filter_length"]),
        target_filter_length=int(
            proposal["recovered_rmp_target_filter_length"]),
        target_update_delay_steps=int(
            proposal["recovered_rmp_target_update_delay_steps"]),
        output_delay_steps_per_joint=tuple(
            int(value) for value in
            proposal["recovered_rmp_output_delay_steps_per_joint"]),
        joint_velocity_feedforward=float(
            proposal["recovered_rmp_joint_velocity_feedforward"]),
    )
    original_candidate = CandidateV2(
        target_update_delay_steps=3,
        output_delay_steps_per_joint=(18, 19, 19, 18, 18, 18, 18),
    )

    prior_result: dict[str, Any] | None = None
    if args.low_level_only:
        result_path = output / "rmp_tuning_v2_results.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"--low-level-only requires existing {result_path}")
        prior_result = json.loads(result_path.read_text(encoding="utf-8"))
        selected = dict(prior_result["selected_candidate"])
        defaults = CandidateV2()
        for field in defaults.__dataclass_fields__:
            if field not in selected:
                selected[field] = getattr(defaults, field)
            if isinstance(getattr(defaults, field), tuple):
                selected[field] = tuple(selected[field])
        resumed_candidate = CandidateV2(**selected)
    else:
        resumed_candidate = initial_candidate

    tuner = TunerV2(
        recording, aligned, real_q, real_dq, commands, elapsed, output,
        reset_log=not args.low_level_only)
    decisions: list[dict[str, Any]] = (
        [] if prior_result is None
        else list(prior_result.get("search_decisions", [])))
    try:
        original = tuner.evaluate(original_candidate)
        initial = tuner.evaluate(initial_candidate)
        current = tuner.evaluate(resumed_candidate)
        if args.low_level_only:
            current = _per_joint_sweep(
                tuner, current, "dof_damping_scale_per_joint",
                lambda _value: (0.50, 0.75, 1.00, 1.25, 1.50), decisions)
            current = _per_joint_sweep(
                tuner, current, "dof_armature_scale_per_joint",
                lambda _value: (0.50, 0.75, 1.00, 1.25, 1.50), decisions)
            current = _per_joint_sweep(
                tuner, current, "acceleration_feedforward_s2_per_joint",
                lambda value: (
                    float(value) - 2.5e-5,
                    float(value),
                    float(value) + 2.5e-5,
                ), decisions)
            current = _global_sweep(
                tuner, current, "joint_acceleration_feedforward",
                (0.05, 0.075, 0.10, 0.125, 0.15), decisions)
        else:
            current = _per_joint_sweep(
                tuner, current, "output_delay_steps_per_joint",
                lambda value: range(int(value) - 2, int(value) + 3), decisions)
            current = _per_joint_sweep(
                tuner, current, "acceleration_limit_scale_per_joint",
                lambda _value: (0.80, 0.90, 1.00, 1.10, 1.20), decisions)
            current = _per_joint_sweep(
                tuner, current, "jnt_kp_scale_per_joint",
                lambda _value: (0.75, 0.90, 1.00, 1.10, 1.25), decisions)
            current = _per_joint_sweep(
                tuner, current, "jnt_kd_scale_per_joint",
                lambda _value: (0.75, 0.90, 1.00, 1.10, 1.25), decisions)
            current = _per_joint_sweep(
                tuner, current, "acceleration_feedforward_s2_per_joint",
                lambda _value: (-5.0e-5, -2.5e-5, 0.0, 2.5e-5, 5.0e-5),
                decisions)
            current = _global_sweep(
                tuner, current, "joint_velocity_feedforward",
                (0.35, 0.425, 0.50, 0.575, 0.65), decisions)
            current = _global_sweep(
                tuner, current, "joint_acceleration_feedforward",
                (-0.20, -0.10, 0.0, 0.10, 0.20), decisions)
            current = _global_sweep(
                tuner, current, "estimator_process_scale",
                (0.5, 1.0, 2.0), decisions)
            current = _global_sweep(
                tuner, current, "estimator_measure_scale",
                (0.5, 1.0, 2.0), decisions)
            current = _global_sweep(
                tuner, current, "output_filter_length",
                (30, 33, 35, 37, 40), decisions)
            current = _global_sweep(
                tuner, current, "target_filter_length",
                (6, 8, 10, 12, 14), decisions)
            current = _global_sweep(
                tuner, current, "target_update_delay_steps",
                (2, 3, 4), decisions)
            current = _per_joint_sweep(
                tuner, current, "output_delay_steps_per_joint",
                lambda value: range(int(value) - 1, int(value) + 2), decisions)

        original_full = tuner.evaluate(original.candidate, search_only=False)
        initial_full = tuner.evaluate(initial.candidate, search_only=False)
        selected_full = tuner.evaluate(current.candidate, search_only=False)
        result = {
            "recording": recording.name,
            "ticks": len(aligned),
            "duration_s": float(elapsed[-1]),
            "objective": (
                "same-real-q_ref: 0.65*train(1-9s) + "
                "0.35*validation(9-14s); 14s+ held out"),
            "original_candidate": asdict(original.candidate),
            "v1_candidate": asdict(initial.candidate),
            "selected_candidate": asdict(selected_full.candidate),
            "original_metrics": original_full.metrics,
            "v1_metrics": initial_full.metrics,
            "selected_metrics": selected_full.metrics,
            "search_decisions": decisions,
            "evaluation_count": (
                tuner.evaluations
                + (0 if prior_result is None
                   else int(prior_result.get("evaluation_count", 0)))),
            "proposed_env_cfg_override": _proposal(
                tuner, selected_full.candidate),
        }
        (output / "rmp_tuning_v2_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n", encoding="utf-8")
        (output / "proposed_rmp_env_cfg_override.json").write_text(
            json.dumps(
                result["proposed_env_cfg_override"], indent=2,
                ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8")
        np.savez_compressed(
            output / "same_command_replay_v2.npz",
            elapsed=elapsed,
            real_q=real_q,
            real_dq=real_dq,
            original_q=original_full.q,
            original_dq=original_full.dq,
            v1_q=initial_full.q,
            v1_dq=initial_full.dq,
            selected_q=selected_full.q,
            selected_dq=selected_full.dq,
        )
        print(json.dumps({
            "output": str(output),
            "evaluations": tuner.evaluations,
            "selected_candidate": result["selected_candidate"],
            "original_post_warmup": original_full.metrics["post_warmup"],
            "v1_post_warmup": initial_full.metrics["post_warmup"],
            "selected_post_warmup": selected_full.metrics["post_warmup"],
            "selected_held_out": selected_full.metrics[
                "held_out_14_s_to_end"],
        }, indent=2, ensure_ascii=False))
    finally:
        tuner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
