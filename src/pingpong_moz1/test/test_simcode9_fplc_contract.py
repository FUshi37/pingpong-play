"""Frozen FPLC source pairing and hardware-boundary tests; no ROS actuation."""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace
from typing import NamedTuple
import xml.etree.ElementTree as ET

import numpy as np
import mujoco
import pytest

from pingpong_controller.fplc_deployment import (
    FPLC_CONTRACTS, FPLC_PROFILES, REPO_ROOT, fplc_assets,
    fplc_workspace_intersection,
)
from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller import fplc_numpy as port
from pingpong_controller.mjx_rl_policy import _CheckpointUnpickler, _flatten_params
from pingpong_controller.policy_release_manifest import validate_policy_release
from pingpong_controller.sim_mirror import (
    RealBallSimMirror, SimMirrorConfig, sim_mirror_config_for_policy_profile,
)
from pingpong_controller.tools import interactive_policy_safety_probe as probe_module

SIM_SOURCE = REPO_ROOT / 'simcode9/pingpong_moz1-main/pingpong_controller/tools/rl_sim'
REAL_LIMITS = ((-.348, .263), (-.702, -.230), (.842, 1.462))


class _Array(np.ndarray):
    """Evaluate original JAX .at and fori_loop operations without installing JAX."""
    @property
    def at(self):
        parent = self
        class Update:
            def __getitem__(self, key):
                self.key = key
                return self
            def set(self, value):
                out = parent.copy()
                out[self.key] = value
                return out
            def add(self, value):
                out = parent.copy()
                out[self.key] += value
                return out
        return Update()


class _NumpyJax:
    def __init__(self, module=np):
        self.module = module
    def __getattr__(self, name):
        value = getattr(self.module, name)
        if name == 'linalg':
            return _NumpyJax(value)
        if not callable(value) or isinstance(value, type):
            return value
        def call(*args, **kwargs):
            result = value(*args, **kwargs)
            return result.view(_Array) if isinstance(result, np.ndarray) else result
        return call


def _fori_loop(start, end, body, state):
    for index in range(start, end):
        state = body(index, state)
    return state


def _source_functions(filename, names, constants=None):
    """Compile unchanged definitions from downloaded source, replacing only JAX."""
    tree = ast.parse((SIM_SOURCE / filename).read_text())
    nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert {node.name for node in nodes} == set(names)
    namespace = dict(
        __name__=__name__, np=np, jnp=_NumpyJax(), math=math,
        jax=SimpleNamespace(Array=np.ndarray, lax=SimpleNamespace(fori_loop=_fori_loop)),
        NamedTuple=NamedTuple, dataclass=dataclasses.dataclass,
    )
    namespace.update(constants or {})
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SIM_SOURCE / filename), 'exec'), namespace)
    return namespace


def _literal_source_constants(filename):
    result = {}
    for node in ast.parse((SIM_SOURCE / filename).read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    result[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return result


@pytest.fixture(params=FPLC_PROFILES)
def controller(request):
    return FPLCPolicyController(observation_profile=request.param)


def _predict(controller, **overrides):
    options = dict(ball_valid=True, ball_obs_age_s=.012,
                   arm_q=controller.warm_arm_q.copy(), arm_dq=np.linspace(-.03, .03, 7),
                   direct_ball_position_m=np.array([.01, -.48, 1.16], np.float32),
                   direct_ball_age_s=.012, direct_ball_stamp_s=1.0,
                   ball_observed=True)
    options.update(overrides)
    return controller.predict(np.array([.02, -.47, 1.18]), np.array([.1, -.1, -.4]), **options)


def _full_forward_reference(controller, q, **_kwargs):
    for jid, value in zip(controller._arm_jids, q):
        controller.mj_data.qpos[int(controller.mj_model.jnt_qposadr[jid])] = value
    mujoco.mj_forward(controller.mj_model, controller.mj_data)


def test_kinematics_only_matches_full_forward_geometry_and_jacobians(controller, monkeypatch):
    kinematics = controller._update_kinematics
    rng = np.random.default_rng(42)
    for dtype in (np.float32, np.float64):
        for _ in range(40):
            q = np.clip(controller.native_reset_arm_q + rng.normal(0, .3, 7),
                        controller.arm_q_low, controller.arm_q_high).astype(dtype)
            monkeypatch.setattr(controller, '_update_kinematics', kinematics)
            fast_pose = controller._pose(q)
            fast_base = controller._compute_racket_pose_quat_base(q)
            fast_position = controller._compute_racket_pose_base(q)
            monkeypatch.setattr(controller, '_update_kinematics',
                                lambda q, **kw: _full_forward_reference(controller, q, **kw))
            for left, right in zip(fast_pose, controller._pose(q)):
                np.testing.assert_array_equal(left, right)
            for left, right in zip(fast_base, controller._compute_racket_pose_quat_base(q)):
                np.testing.assert_array_equal(left, right)
            for left, right in zip(fast_position, controller._compute_racket_pose_base(q)):
                np.testing.assert_array_equal(left, right)


def test_optimized_fk_preserves_closed_loop_policy_outputs(controller, monkeypatch):
    reference = FPLCPolicyController(observation_profile=controller.fplc_profile)
    monkeypatch.setattr(reference, '_update_kinematics',
                        lambda q, **kw: _full_forward_reference(reference, q, **kw))
    q, dq = controller.native_reset_arm_q.copy(), np.zeros(7, np.float32)
    for index in range(160):
        t = index * .005
        ball = np.array([.004*np.sin(t), -.49+.005*np.sin(2*t),
                         1.14+.06*np.sin(10*t)], np.float32)
        velocity = np.array([.004*np.cos(t), .01*np.cos(2*t),
                             .6*np.cos(10*t)], np.float32)
        kwargs = dict(ball_valid=True, arm_q=q, arm_dq=dq,
                      direct_ball_position_m=ball, direct_ball_age_s=(index % 3)*.003,
                      direct_ball_stamp_s=t, confirmed_hit_count=index // 40)
        command = reference.predict(ball, velocity, **kwargs)
        controller.predict(ball, velocity, **kwargs)
        for field in ('last_obs', 'last_action', 'arm_cmd_q', 'arm_cmd_qvel',
                      'prev_action', 'ball_history', 'racket_history', 'age_history'):
            np.testing.assert_array_equal(getattr(controller, field), getattr(reference, field))
        assert controller.last_projection == reference.last_projection
        new_q = q + np.float32(.12) * (command-q)
        dq, q = (new_q-q)/np.float32(.005), new_q


@pytest.mark.parametrize('profile', [*FPLC_PROFILES, 'gpu0_qvel_real_rmp_v85'])
def test_fplc_control_scheduler_is_independent_of_ros_executor(profile, monkeypatch):
    calls = []
    class Thread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def start(self):
            calls.append(('thread', self.kwargs))
    monkeypatch.setattr(probe_module.threading, 'Thread', Thread)
    bridge = SimpleNamespace(start_backend_control_timer=lambda *args: calls.append(('ros', args)))
    probe = SimpleNamespace(control_loop_thread=None, ros_bridge=bridge,
        active_policy_profile=profile, control_loop_stop=__import__('threading').Event(),
        cfg=SimpleNamespace(control_dt=.005), _backend_control_loop=lambda: None,
        _backend_control_timer_callback=lambda: None)
    probe_module.PolicySafetyProbe._start_backend_control_loop(probe)
    expected = 'thread' if profile in {*FPLC_PROFILES, 'gpu0_qvel_real_rmp_v85'} else 'ros'
    assert len(calls) == 1 and calls[0][0] == expected
    assert probe.control_loop_uses_ros_timer == (expected == 'ros')


def test_authenticated_actor_and_manifest(controller):
    profile = controller.fplc_profile
    model, xml = fplc_assets(profile)
    verified = validate_policy_release(
        REPO_ROOT / 'config/policy_release_manifest.json', profile, REPO_ROOT,
        policy_profile=profile, observation_dim=controller._expected_obs_dim,
        checkpoint_path=model, policy_xml_path=xml, include_git=False)
    assert verified.release_id == profile
    with model.open('rb') as handle:
        params = _flatten_params(_CheckpointUnpickler(handle).load()['params'])
    _predict(controller)
    x = controller.last_obs.astype(np.float64)
    for layer in ('l1', 'l2', 'out'):
        x = x @ np.asarray(params[f'pi/{layer}/w'], dtype=np.float64) + np.asarray(
            params[f'pi/{layer}/b'], dtype=np.float64)
        if layer != 'out':
            x = np.tanh(x)
    np.testing.assert_allclose(controller.last_action, x, atol=3e-6, rtol=1e-6)
    assert controller._expected_obs_dim == FPLC_CONTRACTS[profile]['actor_dim']
    assert controller.pose_is_safe(controller.warm_arm_q)
    assert not controller.pose_is_safe(controller.warm_arm_q, fplc_workspace_intersection(REAL_LIMITS))
    np.testing.assert_array_equal(controller.episode_target_xy, np.array([0, -.49], np.float32))


def test_profile_identity_and_xml_fail_before_loading(tmp_path):
    with pytest.raises(RuntimeError, match='checkpoint SHA'):
        FPLCPolicyController(fplc_assets('fplc_gpu1_v151')[0], observation_profile='fplc_gpu1_v172')
    invalid_xml = tmp_path / 'changed.xml'
    invalid_xml.write_text('<mujoco/>')
    with pytest.raises(RuntimeError, match='XML SHA'):
        FPLCPolicyController(observation_profile='fplc_gpu1_v172', robot_xml_path=invalid_xml)
    with pytest.raises(RuntimeError, match='200 Hz'):
        FPLCPolicyController(observation_profile='fplc_gpu1_v172', dt=.01)


def test_observation_history_reset_and_fractional_camera_clock(controller):
    controller._policy_action = lambda obs: np.zeros(7, np.float32)
    _predict(controller)
    assert controller.last_projection['accepted']
    np.testing.assert_array_equal(controller.last_obs[14:28], np.zeros(14))
    original_direct = controller.ball_history[0, 0].copy()
    for tick in (1, 2):  # 90/200 fractional clock refreshes next at tick 3.
        _predict(controller, direct_ball_position_m=np.full(3, tick), direct_ball_age_s=.030)
        np.testing.assert_array_equal(controller.ball_history[0, 0], original_direct)
        assert controller.age_history[0, 0] == pytest.approx(.012 + tick * .005)
    fresh_direct = np.array([.015, -.485, 1.17], np.float32)
    _predict(controller, direct_ball_position_m=fresh_direct, direct_ball_age_s=.030)
    np.testing.assert_array_equal(controller.ball_history[0, 0], fresh_direct)
    assert controller.age_history[0, 0] == pytest.approx(.030)
    q = controller.warm_arm_q
    pos, rot, *_ = controller._pose(q)
    pose = np.concatenate((pos, port.rotation_matrix_to_6d(rot[None])[0]))[None]
    common = dict(q=q[None], dq=np.linspace(-.03, .03, 7, dtype=np.float32)[None],
                  q_ref_error=np.zeros((1, 7)), previous_action=np.zeros((1, 7)),
                  ball_positions=controller.ball_history, ball_ages=controller.age_history,
                  target_xy_m=np.array([[0, -.49]], np.float32))
    if controller._expected_obs_dim == 72:
        reference = _source_functions('fixed_plane_v149.py', ['make_simple_position_observation'])
        expected = reference['make_simple_position_observation'](
            **common, racket_pose_now=pose, ball_position_valid=np.array([True]))[0]
        np.testing.assert_array_equal(controller.last_obs, expected)
    else:
        reference = _source_functions('fixed_plane_v92.py', ['make_history_actor_observation'],
                                      {'HISTORY_LAGS_TICKS': tuple(range(8)), 'ACTOR_OBS_DIM': 137})
        expected = reference['make_history_actor_observation'](
            **common, racket_poses=controller.racket_history,
            desired_apex_z_m=np.array([1.14], np.float32), hit_count=np.array([0]),
            ball_valid=np.array([True]))[0]
        np.testing.assert_array_equal(controller.last_obs[:137], expected)
    controller.invalidate_after_intervention()
    assert controller.ball_history is None and controller._fplc_requires_measured_reset
    _predict(controller)
    np.testing.assert_array_equal(controller.last_obs[14:28], np.zeros(14))
    assert np.all(controller.age_history[0, 1:] == 1)


def test_original_dynamic_contact_solver_and_context_pairing():
    reference = _source_functions('fixed_contact_plane_training.py', [
        'ContactTargetConfig', 'ContactTarget', 'propagate_ballistic_state_to_age',
        'bounded_next_crossing_target_xy', 'predict_descending_plane_crossing',
        '_rotation_from_normal', 'compute_dynamic_contact_target'])
    pos = np.array([[.01, -.48, 1.18], [.15, -.4, 1.02], [0, -.49, 1.1]], np.float32)
    vel = np.array([[.1, -.1, -.4], [.2, .1, -1], [0, 0, 1]], np.float32)
    kw = dict(workspace_center_xy=(0, -.49), contact_surface_z_m=1.025,
              ball_radius_m=.02, target_apex_z_m=np.array([1.14, 1.4, 1.425], np.float32))
    actual = port.compute_dynamic_contact_target(pos, vel, np.eye(3, dtype=np.float32),
                                                 cfg=port.ContactTargetConfig(), **kw)
    expected = reference['compute_dynamic_contact_target'](
        pos, vel, np.eye(3, dtype=np.float32), cfg=reference['ContactTargetConfig'](), **kw)
    for field in expected._fields:
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1e-7, rtol=0)
    rotation_reference = _source_functions('fixed_plane_v92.py', ['rotation_matrix_to_6d'])
    ref_context = _source_functions('fixed_plane_v93.py', ['make_predicted_contact_context'], rotation_reference)
    context_kw = dict(predicted_ball_position_m=expected.predicted_ball_position_at_contact_m,
                      time_to_contact_s=expected.time_to_contact_s, contact_valid=expected.valid,
                      racket_target_position_m=expected.position_m,
                      racket_target_rotation=expected.rotation_matrix,
                      racket_surface_position_m=np.zeros((3, 3)),
                      next_crossing_xy_m=expected.next_crossing_target_xy_m)
    np.testing.assert_array_equal(port.make_predicted_contact_context(**context_kw),
                                  ref_context['make_predicted_contact_context'](**context_kw))


@pytest.mark.parametrize('dtype,nominal_dtype', [
    (np.float32, np.float32), (np.float64, np.float64), (np.float64, np.float32),
])
def test_compiled_single_lane_projection_matches_original_arithmetic(monkeypatch, dtype, nominal_dtype):
    if port._dykstra_single_compiled is None:
        pytest.skip('optional compiled projection unavailable')
    port.warm_fplc_projection()
    rng = np.random.default_rng(405)
    kernel = port._dykstra_single_compiled
    for case in range(80):
        rows = rng.normal(size=(1, 10, 7)).astype(dtype)
        if case % 4 == 0:
            rows[0, 0] *= dtype(1e-12)
        values = (
            rng.normal(0, 4, (1, 7)).astype(nominal_dtype), rows,
            rng.uniform(-.1, 2., (1, 10)).astype(dtype),
            -rng.uniform(.1, 5., (1, 7)).astype(dtype),
            rng.uniform(.1, 5., (1, 7)).astype(dtype),
            rng.uniform(.1, 5., 7).astype(nominal_dtype), 24,
        )
        monkeypatch.setattr(port, '_dykstra_single_compiled', kernel)
        compiled = port._project_halfspaces_dykstra_numpy(*values)
        monkeypatch.setattr(port, '_dykstra_single_compiled', None)
        original = port._project_halfspaces_dykstra_numpy(*values)
        for accelerated, reference in zip(compiled, original):
            np.testing.assert_array_equal(accelerated, reference)


def test_original_24_sweep_weighted_projection_pairing():
    reference = _source_functions('cartesian_pose_action_projection_jax.py', [
        'CartesianPoseProjectionJaxResult', '_project_halfspaces_dykstra_jax',
        'project_cartesian_pose_qvel_jax'])
    rng = np.random.default_rng(123)
    n = 12
    kw = dict(nominal_qvel=rng.normal(size=(n, 7)).astype(np.float32),
              position_m=np.tile([0, -.49, 1.08], (n, 1)).astype(np.float32),
              relative_rpy_rad=rng.normal(0, .012, (n, 3)).astype(np.float32),
              translation_jacobian=rng.normal(0, .15, (n, 3, 7)).astype(np.float32),
              relative_rpy_jacobian=rng.normal(0, .3, (n, 3, 7)).astype(np.float32),
              position_lower_m=np.array([-.3475, -.6995, 1.0005], np.float32),
              position_upper_m=np.array([.3475, -.2805, 1.1495], np.float32),
              roll_pitch_limit_rad=np.deg2rad(np.array([1.9, 1.9], np.float32)),
              velocity_lower=np.full((n, 7), -3, np.float32),
              velocity_upper=np.full((n, 7), 3, np.float32),
              velocity_scale=np.array([3, 3, 4, 4, 5, 5, 5], np.float32),
              position_gain_s=20., orientation_gain_s=20., projection_sweeps=24)
    expected = reference['project_cartesian_pose_qvel_jax'](**kw)
    actual = port.project_cartesian_pose_qvel_numpy(**kw)
    for field in expected._fields:
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=2e-6, rtol=0)


def test_non_linear_backstop_and_reference_integration(controller, monkeypatch):
    controller._policy_action = lambda obs: np.full(7, .05, np.float32)
    _predict(controller)
    before = controller.arm_cmd_q.copy()
    old_action = controller.prev_action.copy()
    _predict(controller)
    np.testing.assert_allclose(controller.last_obs[14:21], before - controller.warm_arm_q, atol=1e-7)
    np.testing.assert_array_equal(controller.last_obs[21:28], old_action)
    np.testing.assert_allclose(controller.arm_cmd_q, before + controller.arm_cmd_qvel * .005, atol=1e-7)
    before = controller.arm_cmd_q.copy()
    real_pose = controller._pose
    calls = []
    def unsafe_candidate(q, **kwargs):
        result = list(real_pose(q, **kwargs))
        calls.append(q)
        if len(calls) == 2:
            result[2] = np.array([0, -.49, 2], np.float32)
        return tuple(result)
    monkeypatch.setattr(controller, '_pose', unsafe_candidate)
    held, dq = controller._project_reference(controller.warm_arm_q, np.ones(7, np.float32))
    np.testing.assert_array_equal(held, before)
    np.testing.assert_array_equal(dq, np.zeros(7))
    assert controller.last_projection['nonlinearBackstop']


def test_gpu0_context_geometry_and_apex_stages():
    controller = FPLCPolicyController(observation_profile='fplc_gpu0_v161')
    controller._policy_action = lambda obs: np.zeros(7, np.float32)
    reference = _source_functions('fixed_contact_plane_training.py', [
        'ContactTargetConfig', 'ContactTarget', 'propagate_ballistic_state_to_age',
        'bounded_next_crossing_target_xy', 'predict_descending_plane_crossing',
        '_rotation_from_normal', 'compute_dynamic_contact_target'])
    rotation_reference = _source_functions('fixed_plane_v92.py', ['rotation_matrix_to_6d'])
    context_reference = _source_functions('fixed_plane_v93.py', ['make_predicted_contact_context'], rotation_reference)
    cfg = controller._checkpoint_env_cfg
    geom = controller._racket_geom_id
    original_geom_pos = controller.mj_model.geom_pos[geom].copy()
    for hits, apex in ((0, 1.14), (1, 1.249), (2, 1.4), (3, 1.425), (7, 1.425)):
        controller.reset()
        _predict(controller, confirmed_hit_count=hits)
        assert controller.last_obs[132] == pytest.approx(apex)
        assert controller.last_obs[136] == pytest.approx(min(hits / 4, 1))
        _, rotation, *_ = controller._pose(controller.warm_arm_q)
        # Reproduce the simulator's compiled +3 mm collision geometry for the
        # reference feature, then restore the actual FK model unchanged.
        controller.mj_model.geom_pos[geom, 2] += .003
        mujoco.mj_forward(controller.mj_model, controller.mj_data)
        surface = controller.mj_data.geom_xpos[geom] + rotation[:, 2] * controller.mj_model.geom_size[geom, 0]
        controller.mj_model.geom_pos[geom] = original_geom_pos
        mujoco.mj_forward(controller.mj_model, controller.mj_data)
        target = reference['compute_dynamic_contact_target'](
            np.array([[.02, -.47, 1.18]], np.float32), np.array([[.1, -.1, -.4]], np.float32),
            controller.reference_rotation, workspace_center_xy=(0, -.49),
            contact_surface_z_m=1.025, ball_radius_m=.02,
            target_apex_z_m=np.array([apex], np.float32),
            cfg=reference['ContactTargetConfig'](**controller.contract['contact_target_config']))
        reachable = all(low <= target.position_m[0, axis] <= high for axis, (low, high) in enumerate(
            (cfg['fixed_contact_racket_x_bounds_m'], cfg['fixed_contact_racket_y_bounds_m'],
             cfg['fixed_contact_racket_z_bounds_m'])))
        expected = context_reference['make_predicted_contact_context'](
            predicted_ball_position_m=target.predicted_ball_position_at_contact_m,
            time_to_contact_s=target.time_to_contact_s, contact_valid=target.valid & reachable,
            racket_target_position_m=target.position_m, racket_target_rotation=target.rotation_matrix,
            racket_surface_position_m=surface[None], next_crossing_xy_m=target.next_crossing_target_xy_m)
        np.testing.assert_allclose(controller.last_obs[137:156], expected[0], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(controller.mj_model.geom_pos[geom], original_geom_pos)


def test_frozen_fk_and_warm_posture_match_original_cpu_xml_pipeline(controller):
    """Run original XML builders and 700-step warm without importing the trainer."""
    dependencies = dict(ET=ET, Path=Path, tempfile=tempfile, hashlib=hashlib)
    original_cpu = _source_functions('rl_juggle_env_random.py', ['_build_temp_xml_with_ball'], dependencies)
    smoke_constants = _literal_source_constants('mjx_smoke.py')
    smoke = _source_functions('mjx_smoke.py', [
        '_finite_racket_mesh16_vertices', '_set_ball_racket_collision_geometry',
        '_write_mjx_contact_only_xml'], {**dependencies, **smoke_constants})
    pd = _source_functions('mjx_juggle_env.py', ['_apply_right_arm_pd_profile'], {
        **dependencies, **_literal_source_constants('mjx_juggle_env.py')})
    temporary = original_cpu['_build_temp_xml_with_ball'](fplc_assets(controller.fplc_profile)[1])
    contact_xml = smoke['_write_mjx_contact_only_xml'](temporary, simulation_base_mode='aligned_fixed')
    try:
        prepared = pd['_apply_right_arm_pd_profile'](contact_xml, 'recovered_rmp_rmpmd_v2')
        model = mujoco.MjModel.from_xml_path(str(prepared))
        data = mujoco.MjData(model)
        targets = _literal_source_constants('rl_juggle_env_random.py')['TARGET_DEGREES']
        targets = {**targets, **dict(zip(
            probe_module.RIGHT_ARM_JOINTS, controller._checkpoint_env_cfg['right_arm_reset_degrees']))}
        for name, degrees in targets.items():
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = np.float32(np.deg2rad(degrees))
        for _ in range(700):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right_ee_site')
        np.testing.assert_allclose(controller.reference_rotation, data.site_xmat[site].reshape(3, 3),
                                   atol=2e-7, rtol=0)
        q_ref = [data.qpos[model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name)]] for name in probe_module.RIGHT_ARM_JOINTS]
        np.testing.assert_allclose(controller.warm_arm_q, q_ref, atol=2e-7, rtol=0)
        pos, *_ = controller._pose(controller.warm_arm_q)
        np.testing.assert_allclose(pos, data.site_xpos[site], atol=2e-7, rtol=0)
        # Execute the unmodified nested solver from the downloaded trainer,
        # using the independent original XML and original 700-step warm.
        nested = next(node for node in ast.walk(ast.parse(
            (SIM_SOURCE / 'mjx_juggle_env.py').read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == 'solve_follow_xy_pose')
        qadr = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in probe_module.RIGHT_ARM_JOINTS]
        vadr = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
                for name in probe_module.RIGHT_ARM_JOINTS]
        warm_q = data.qpos[qadr].astype(np.float32).astype(np.float64)
        namespace = dict(
            np=np, mujoco=mujoco, warm=data,
            self=SimpleNamespace(mj_model=model, racket_site_id=site, arm_qadr_np=qadr,
                                 arm_vadr_np=vadr, act_dim=7),
            cfg=SimpleNamespace(**controller._checkpoint_env_cfg),
            workspace_center_xy=np.array([0., -.49]),
            warm_racket_pos=data.site_xpos[site].copy(),
            warm_racket_rot=data.site_xmat[site].reshape(3, 3).copy(),
            warm_arm_q_np=warm_q, arm_lo_np=controller.arm_q_low.astype(np.float64),
            arm_hi_np=controller.arm_q_high.astype(np.float64))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[nested], type_ignores=[])),
                     str(SIM_SOURCE / 'mjx_juggle_env.py'), 'exec'), namespace)
        native_q, source_error = namespace['solve_follow_xy_pose'](0., 0., warm_q)
        np.testing.assert_allclose(controller.native_reset_arm_q, native_q, atol=2e-6, rtol=0)
        np.testing.assert_allclose(controller.native_reset_position_m, data.site_xpos[site], atol=2e-7, rtol=0)
        assert controller.native_reset_error_m == pytest.approx(source_error, abs=2e-7)
        assert controller.pose_is_safe(controller.native_reset_arm_q, fplc_workspace_intersection(REAL_LIMITS))
        validate_bounds = np.array([[-.034, .014], [-.522, -.481], [1.010, 1.123]])
        assert np.all(controller.native_reset_position_m >= validate_bounds[:, 0])
        assert np.all(controller.native_reset_position_m <= validate_bounds[:, 1])
        np.testing.assert_array_equal(controller.native_reset_target_m, np.array([0., -.49, 1.025], np.float32))
        np.testing.assert_array_equal(controller.initial_arm_q_rad, controller.native_reset_arm_q)
    finally:
        temporary.unlink(missing_ok=True)
        contact_xml.unlink(missing_ok=True)


def test_invalid_inputs_and_direct_observed_flag(controller):
    for kwargs, message in (
            ({'ball_obs_age_s': .351}, '350 ms'),
            ({'ball_valid': False}, '350 ms'),
            ({'arm_q': None}, 'atomic measured'),
            ({'arm_dq': np.full(7, np.nan)}, 'finite length'),
            ({'dt': .006}, '5 ms'),
            ({'observation_scales': np.full(controller._expected_obs_dim, .5)}, 'scaling')):
        controller.reset()
        with pytest.raises(RuntimeError, match=message):
            _predict(controller, **kwargs)
    controller.reset()
    with pytest.raises(RuntimeError, match='direct ball sample'):
        _predict(controller, ball_observed=False)
    controller._policy_action = lambda obs: np.zeros(7, np.float32)
    _predict(controller)
    for _ in range(3):
        _predict(controller, ball_observed=False)
    valid_index = 71 if controller._expected_obs_dim == 72 else 135
    assert controller.last_obs[valid_index] == 0  # Valid forecast is not a direct observation.


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(probe_module.AsyncProbeRecorder, '_spawn_standby_writer', lambda self: None)
    args = probe_module.build_arg_parser().parse_args([
        '--policy-profile', 'fplc_gpu1_v172', '--policy-obs-dim', '72',
        '--ee-x-limits-m=-.348,.263', '--ee-y-limits-m=-.702,-.230',
        '--ee-z-limits-m=.842,1.462', '--disable-ball-state-estimator',
        '--policy-plant-matcher-mode', 'off'])
    cfg = probe_module.config_from_args(args)
    cfg.enable_sim_mirror = False
    cfg.backend_control_loop = False
    cfg.enable_compensation = False
    p = probe_module.PolicySafetyProbe(cfg)
    p.reset_hold_latched = False
    p.reset_workspace_bypass_until_wall = 0
    p.slow_reset_trajectory.active = False
    yield p
    p.ros_bridge = None
    p.shutdown()


def _policy_options(probe, profile='fplc_gpu1_v172'):
    return probe._runtime_options_from_payload(dict(
        controlMode='policy', policyProfile=profile, enableCaptureGate=False, enableReleaseGate=False,
        enableLostFailsafe=False, enableDoneMonitoring=False, workspaceSafetyMode='off',
        enableCompensation=True, enableContactActionShield=True,
        ballObservationRateHz=200, policyObservationScales=[.5] * 156))


def test_initial_fplc_workspace_preserves_general_configuration(probe):
    assert probe.base_ee_limits_m == REAL_LIMITS
    small_box = probe_module.box_payload(*fplc_workspace_intersection(REAL_LIMITS))
    assert probe.latest_snapshot_payload['eeBox'] == small_box
    assert probe.browser_config()['eeBox'] == small_box


def test_v220_probe_startup_uses_archived_actor_xml_and_fplc_workspace(probe):
    profile = 'fplc_v220_s17_u839_20261013_r1'
    cfg = dataclasses.replace(
        probe.cfg, policy_profile=profile,
        ee_x_limits_m=REAL_LIMITS[0], ee_y_limits_m=REAL_LIMITS[1],
        ee_z_limits_m=REAL_LIMITS[2])
    new_probe = probe_module.PolicySafetyProbe(cfg)
    try:
        model, xml = fplc_assets(profile)
        archived = json.loads((model.parent / 'release.json').read_text())
        browser = new_probe.browser_config()
        assert browser['policy']['defaultProfile'] == profile
        assert browser['policy']['activeProfile'] == profile
        assert browser['policy']['modelPath'] == str(model)
        assert browser['policy']['policyXmlPath'] == str(xml)
        assert xml.name == 'moz1_pd_1ms_v98.xml'
        assert new_probe.controller.contract['sha256'] == archived['policy_sha256']
        assert new_probe.controller.contract['policy_xml_sha256'] == archived['xml_sha256']
        assert new_probe.controller.contract['actor_dim'] == 72
        assert new_probe.controller.contract['critic_dim'] == 477
        assert new_probe.controller.contract['stage_update'] == 839
        assert browser['eeBox'] == probe_module.box_payload(*fplc_workspace_intersection(REAL_LIMITS))
        assert new_probe.base_ee_limits_m == REAL_LIMITS
        assert new_probe.active_policy_model == profile
    finally:
        new_probe.shutdown()


def test_web_switch_and_forbidden_controls(probe):
    for profile in FPLC_PROFILES:
        options = _policy_options(probe, profile)
        assert options.workspace_safety_mode == 'position'
        assert options.ball_observation_rate_hz == 90
        assert not options.enable_compensation and not options.contact_action_shield.enabled
        assert all(scale == 1 for scale in options.policy_observation_scales)
        result = probe._run_one_control_step(np.array([0, -.49, 1.16]),
                                            np.array([0, 0, -.3]), True, .01, options)
        assert result[0] == 'policy' and result[4], result
        assert probe.controller.last_projection
        assert probe.active_policy_model == profile
        assert probe.controller.last_obs.size == FPLC_CONTRACTS[profile]['actor_dim']
        assert probe_module._policy_runtime_contract(probe.controller)['observationDim'] == probe.controller.last_obs.size
        assert probe.browser_config()['policy']['observationDim'] == probe.controller.last_obs.size
        assert probe.browser_config()['eeBox']['z0'] == .95
        assert probe.browser_config()['eeBox'] == probe_module.box_payload(
            (-.1, .1), (-.6, -.4), (.95, 1.2))
        assert probe.reset_initial_pose_id == 'simcode9_native'
        np.testing.assert_array_equal(probe._current_reset_pose_q(), probe.controller.native_reset_arm_q)
        assert probe.browser_config()['resetInitialPoses']['selected'] == 'simcode9_native'
        np.testing.assert_allclose(np.rad2deg(probe.controller.rp_outer), [2., 2.], atol=1e-6)


@pytest.mark.parametrize('fault', ['unsafe_target', 'stale_joint', 'changed_target', 'unsafe_measured',
                                 'missing_feedback', 'outside_real_box_target', 'outside_real_box_measured'])
def test_publish_boundary_latches_and_never_enters_limiter(probe, monkeypatch, fault):
    options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    measured = probe.controller.native_reset_arm_q.copy()
    published = []
    bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), .001, 'test'),
        get_joint_feedback=lambda: (measured, np.zeros(7)),
        publish_right_arm_command=lambda q: published.append(np.array(q)),
    )
    probe.ros_bridge = bridge
    probe.command_stream_active = True
    for method in ('filter', 'reset', 'observe_published'):
        monkeypatch.setattr(probe_module.RightArmCommandSafetyLimiter, method,
                            lambda *args, **kw: pytest.fail('entered safety_limiter'), raising=False)
    _predict(probe.controller, arm_q=measured.copy())
    target = probe.controller.arm_cmd_q.copy()
    if fault == 'unsafe_target':
        target[0] = 1.0
    elif fault == 'stale_joint':
        bridge.get_joint_feedback_with_age = lambda: (measured, np.zeros(7), .100, 'test')
    elif fault == 'missing_feedback':
        bridge.get_joint_feedback_with_age = lambda: (None, None, None, 'test')
    elif fault == 'unsafe_measured':
        measured[0] = 1.0
    elif fault == 'outside_real_box_target':
        target = probe.controller.warm_arm_q.copy()
        probe.controller.last_cmd_q = target.copy()
        assert probe.controller.pose_is_safe(target)
        assert not probe.controller.pose_is_safe(target, fplc_workspace_intersection(REAL_LIMITS))
    elif fault == 'outside_real_box_measured':
        measured[:] = probe.controller.warm_arm_q
        assert probe.controller.pose_is_safe(measured)
        assert not probe.controller.pose_is_safe(measured, fplc_workspace_intersection(REAL_LIMITS))
    else:
        # A downstream rewrite inside the box must not become the next q_ref.
        target[0] += .0001
    assert not probe._publish_control_command(target, options, profile='normal')
    assert probe._hard_workspace_stop_active()
    assert probe.publish_rearm_required and probe.controller._fplc_requires_measured_reset
    assert probe.publish_rearm_details['faults']
    assert probe.publish_rearm_details['maximumJointFeedbackAgeS'] == .010
    assert probe.publish_rearm_details['positionLimitsM'] == [list(pair) for pair in
        fplc_workspace_intersection(REAL_LIMITS)]
    assert not probe._publish_control_command(target, options, profile='normal')
    assert len(published) <= 1
    if published:
        np.testing.assert_allclose(published[0], measured, atol=1e-7)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('angle_deg,inside_real_envelope', [
    (-4.01, False), (-3.99, True), (3.99, True), (4.01, False),
])
def test_real_attitude_envelope_at_publish_boundary(probe, monkeypatch, axis, angle_deg,
                                                   inside_real_envelope):
    options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    measured = probe.controller.native_reset_arm_q.copy()
    issued = []
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), .001, 'test'),
        get_joint_feedback=lambda: (measured, np.zeros(7)),
        publish_right_arm_command=lambda q: issued.append(np.array(q)))
    for method in ('filter', 'reset', 'observe_published'):
        monkeypatch.setattr(probe_module.RightArmCommandSafetyLimiter, method,
                            lambda *args, **kw: pytest.fail('entered safety_limiter'), raising=False)
    probe.command_stream_active = True
    target = _predict(probe.controller, arm_q=measured.copy())
    reference = probe.controller.reference_rotation.copy()
    original_pose = probe.controller._pose
    def pose_with_test_attitude(q, **kwargs):
        parts = list(original_pose(q, **kwargs))
        parts[3] = np.zeros(3, np.float32)
        parts[3][axis] = np.deg2rad(angle_deg)
        return tuple(parts)
    monkeypatch.setattr(probe.controller, '_pose', pose_with_test_attitude)
    assert not probe.controller.pose_is_safe(target)
    np.testing.assert_allclose(np.rad2deg(probe.controller.rp_outer), [2., 2.], atol=1e-6)
    np.testing.assert_allclose(np.rad2deg(probe.controller.real_rp_outer), [4., 4.], atol=1e-6)
    assert probe._publish_control_command(target, options) == inside_real_envelope
    assert probe._hard_workspace_stop_active() == (not inside_real_envelope)
    np.testing.assert_array_equal(probe.controller.reference_rotation, reference)
    if inside_real_envelope:
        np.testing.assert_array_equal(issued[-1], target)
    else:
        np.testing.assert_allclose(probe.publish_rearm_details['relativeRollPitchLimitDeg'],
                                   [4., 4.], atol=1e-6)
        assert probe.publish_rearm_required


def test_fplc_ball_done_disarms_and_latches_until_reset(probe, monkeypatch):
    options = dataclasses.replace(_policy_options(probe), publish_commands=True, use_ros_state=True,
                                  enable_done_monitoring=True)
    measured = probe.controller.native_reset_arm_q.copy()
    issued = []
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_coherent=lambda: (measured.copy(), np.zeros(7), .001, 'test', probe.now_s),
        get_joint_feedback_with_age=lambda: (measured.copy(), np.zeros(7), .001, 'test'),
        get_joint_feedback=lambda: (measured.copy(), np.zeros(7)),
        get_cart_feedback=lambda: (None, None, None),
        publish_right_arm_command=lambda q: issued.append(np.array(q)))
    for method in ('filter', 'reset', 'observe_published'):
        monkeypatch.setattr(probe_module.RightArmCommandSafetyLimiter, method,
                            lambda *args, **kw: pytest.fail('entered safety_limiter'), raising=False)
    probe.command_stream_active = True
    probe.has_captured = probe.has_released = True
    probe.controller._policy_action = lambda obs: np.zeros(7, np.float32)
    for _ in range(5):
        result = probe._run_one_control_step(np.array([0., -.49, .2], np.float32),
            np.array([0., 0., -.5], np.float32), True, 0., options)
    assert result[0:2] == ('done_failsafe', 'ball_too_low')
    assert not result[4]
    assert probe.publish_rearm_required and probe.publish_rearm_reason == 'fplc_episode_done'
    assert probe._hard_workspace_stop_active()
    assert probe.controller._fplc_requires_measured_reset
    assert not probe._publish_control_command(probe.controller.current_arm_cmd_q(), options)
    assert len(issued) == 1
    np.testing.assert_array_equal(issued[0], measured)
    for _ in range(8):
        result = probe._run_one_control_step(np.array([0., -.49, 1.2], np.float32),
            np.array([0., 0., -.5], np.float32), True, 0., options)
        assert result[0] == 'done_failsafe' and not result[4]
        assert not probe._publish_control_command(probe.controller.current_arm_cmd_q(), options)
    assert len(issued) == 1


@pytest.mark.parametrize('age_s', [None, .0101, float('inf')])
@pytest.mark.parametrize('hold_method', ['_publish_hold_on_command_disable', '_publish_measured_stop_hold'])
def test_fplc_stop_hold_never_publishes_stale_feedback(probe, age_s, hold_method):
    measured = probe.controller.native_reset_arm_q.copy()
    issued = []
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), age_s, 'test'),
        publish_right_arm_command=lambda q: issued.append(np.array(q)))
    probe.command_stream_active = True
    assert not getattr(probe, hold_method)()
    assert not issued and not probe.command_stream_active


def test_safe_publish_and_supervised_reset_never_enter_limiter(probe, monkeypatch):
    options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    measured = probe.controller.native_reset_arm_q.copy()
    issued = []
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), .001, 'test'),
        get_joint_feedback=lambda: (measured, np.zeros(7)),
        publish_right_arm_command=lambda q: issued.append(np.array(q)))
    for method in ('filter', 'reset', 'observe_published'):
        monkeypatch.setattr(probe_module.RightArmCommandSafetyLimiter, method,
                            lambda *args, **kw: pytest.fail('entered safety_limiter'), raising=False)
    class ForbiddenLimiter:
        def __getattribute__(self, name):
            pytest.fail(f'FPLC touched safety_limiter.{name}')
        def __setattr__(self, name, value):
            pytest.fail(f'FPLC wrote safety_limiter.{name}')
    probe.safety_limiter = ForbiddenLimiter()
    assert probe._arm_command_stream_from_feedback()
    np.testing.assert_array_equal(issued[-1], measured)
    probe.command_stream_active = True
    target = _predict(probe.controller, arm_q=measured.copy())
    assert probe._publish_control_command(target, options)
    np.testing.assert_array_equal(issued[-1], target)
    probe._start_supervised_reset(measured, probe._current_reset_pose_q(), reason='test_reset')
    reset_target = probe._advance_slow_reset_command(advance=True)
    assert probe._publish_control_command(reset_target, options, profile='reset')
    np.testing.assert_array_equal(issued[-1], reset_target)
    probe._sync_policy_to_safe_pose(reset_target)
    assert probe.controller._fplc_requires_measured_reset
    assert probe._publish_hold_on_command_disable()
    np.testing.assert_array_equal(issued[-1], measured)
    probe.command_stream_active = True
    assert probe._publish_measured_stop_hold()
    np.testing.assert_array_equal(issued[-1], measured)


@pytest.mark.parametrize('z_m,inside_real_box', [
    (.96, True), (1.19, True), (.949, False), (1.201, False),
])
def test_real_z_envelope_is_independent_at_publish_boundary(probe, monkeypatch, z_m, inside_real_box):
    options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    measured = probe.controller.native_reset_arm_q.copy()
    issued = []
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), .001, 'test'),
        get_joint_feedback=lambda: (measured, np.zeros(7)),
        publish_right_arm_command=lambda q: issued.append(np.array(q)))
    for method in ('filter', 'reset', 'observe_published'):
        monkeypatch.setattr(probe_module.RightArmCommandSafetyLimiter, method,
                            lambda *args, **kw: pytest.fail('entered safety_limiter'), raising=False)
    probe.command_stream_active = True
    target = _predict(probe.controller, arm_q=measured.copy())
    original_pose = probe.controller._pose

    def pose_at_test_height(q, **kwargs):
        pose = list(original_pose(q, **kwargs))
        pose[2] = pose[2].copy()
        pose[2][2] = z_m
        return tuple(pose)

    # Isolate the physical height check from joint/attitude constraints.  All
    # cases are outside training Z, but only two exceed the requested real box.
    monkeypatch.setattr(probe.controller, '_pose', pose_at_test_height)
    assert not probe.controller.pose_is_safe(target)
    assert probe.controller.pose_is_safe(
        target, fplc_workspace_intersection(REAL_LIMITS)) == inside_real_box
    assert probe._publish_control_command(target, options) == inside_real_box
    assert probe._hard_workspace_stop_active() == (not inside_real_box)
    if inside_real_box:
        np.testing.assert_array_equal(issued[-1], target)
    else:
        assert probe.publish_rearm_required
        for command in issued:
            np.testing.assert_allclose(command, measured, atol=1e-7)


def test_stale_publish_request_cannot_rearm_after_inference_failure(probe, monkeypatch):
    options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    probe.ros_bridge = SimpleNamespace()
    probe.command_stream_active = False
    probe._require_publish_rearm('fplc_inference_fault')
    monkeypatch.setattr(probe, '_arm_command_stream_from_feedback',
                        lambda: pytest.fail('stale same-tick request rearmed the robot'))
    assert not probe._publish_control_command(probe.controller.warm_arm_q, options, profile='parking')


def test_20hz_feedback_arm_failure_preserves_trigger_details(probe, monkeypatch, capsys):
    measured = probe.controller.native_reset_arm_q.copy()
    probe.cfg.ros_publish_commands = True
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), .050, 'joint_state'),
        publish_right_arm_command=lambda q: pytest.fail('stale feedback armed publication'))
    probe._start_supervised_reset(measured, measured, reason='test_reset')
    probe.runtime_options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    assert not probe._arm_command_stream_from_feedback()
    assert not probe.runtime_options.publish_commands
    assert probe.publish_rearm_reason == 'reset_arm_fresh_joint_feedback_required'
    assert probe.publish_rearm_details == {
        'jointFeedbackAgeS': .050, 'maximumJointFeedbackAgeS': .010}
    # The browser must see the original cause even after feedback refreshes.
    probe._require_publish_rearm(probe.publish_rearm_reason, details={
        'jointFeedbackAgeS': .001, 'maximumJointFeedbackAgeS': .010})
    assert probe.publish_rearm_details['jointFeedbackAgeS'] == .050
    assert capsys.readouterr().out.count('[CommandSafety][STOP]') == 1
    assert probe._publish_rearm_blocks_request(publish_requested=False, switch_requested=False)
    assert probe.publish_rearm_details['jointFeedbackAgeS'] == .050
    assert not probe._publish_rearm_blocks_request(publish_requested=True, switch_requested=False)
    assert probe.publish_rearm_details == {}


def test_snapshot_reports_applied_send_state_not_publish_permission(probe, monkeypatch):
    probe.cfg.ros_bridge = True
    probe.cfg.ros_publish_commands = True
    probe.ros_bridge = SimpleNamespace(feedback_snapshot=lambda: {})
    monkeypatch.setattr(probe, '_ros_cart_pose', lambda: (None, None, None))
    probe.runtime_options = None

    def snapshot(**overrides):
        return probe.snapshot('reset', '', {}, np.array([0., -.49, 1.3]),
            np.array([0., -.49, 1.3]), np.zeros(3), False, 0., 0.,
            camera_feedback={}, **overrides)

    initial = snapshot()
    assert initial['ros']['publishAllowed']
    assert not initial['runtime']['publishCommands'] and not initial['ros']['publishCommands']
    armed = dataclasses.replace(_policy_options(probe), publish_commands=True)
    probe.runtime_options = armed
    assert snapshot()['runtime']['publishCommands']
    probe._require_publish_rearm('fplc_workspace_or_command_boundary_fault')
    stale_tick = snapshot(runtime_options=armed, publish_commands=True)
    assert not stale_tick['runtime']['publishCommands'] and not stale_tick['ros']['publishCommands']


@pytest.mark.parametrize('requested_preset', ['default_1', 'default_2_horizontal', 'simcode9_native'])
def test_fplc_reset_cannot_select_the_old_warm_seed(probe, requested_preset):
    reference = probe.controller.reference_rotation.copy()
    snapshot = probe.reset({'resetInitialPose': requested_preset})
    assert probe.reset_initial_pose_id == 'simcode9_native'
    np.testing.assert_array_equal(probe._current_reset_pose_q(), probe.controller.native_reset_arm_q)
    np.testing.assert_array_equal(probe.slow_reset_trajectory.target_rad, probe.controller.native_reset_arm_q)
    np.testing.assert_array_equal(probe.controller.reference_rotation, reference)
    assert probe.controller.pose_is_safe(probe._current_reset_pose_q(), fplc_workspace_intersection(REAL_LIMITS))
    assert snapshot['runtime']['resetInitialPose'] == 'simcode9_native'


def test_stale_atomic_actor_feedback_is_rejected_even_if_later_feedback_is_fresh(probe, monkeypatch):
    q = probe.controller.warm_arm_q.copy()
    dq = np.zeros(7)
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (q, dq, .001, 'fresh_later_callback'),
        get_joint_feedback=lambda: (q, dq))
    options = dataclasses.replace(_policy_options(probe), use_ros_state=True)
    monkeypatch.setattr(probe, '_joint_feedback_for_policy',
                        lambda: (q, dq, .030, 'atomic_stale', 1., {}))
    monkeypatch.setattr(probe.controller, '_policy_action',
                        lambda obs: pytest.fail('stale atomic q/dq reached the actor'))
    mode, reason, _conditions, _age, executed = probe._run_one_control_step(
        np.array([0, -.49, 1.16]), np.array([0, 0, -.3]), True, .01, options)
    assert mode == 'policy_feedback_hold' and not executed, reason
    assert probe.controller._fplc_requires_measured_reset


@pytest.mark.parametrize('profile', FPLC_PROFILES)
def test_launcher_config_only(profile):
    result = subprocess.run(['bash', str(REPO_ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh')],
                            cwd=REPO_ROOT, env={**os.environ, 'PROBE_CONFIG_ONLY': 'true',
                                              'POLICY_RELEASE_ID': profile,
                                              'ENABLE_ROS_BRIDGE': 'false',
                                              'ROS_PUBLISH_COMMANDS': 'false',
                                              'ENABLE_CAMERA_BALL': 'false',
                                              'ENABLE_MOCAP_BALL': 'false'},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'limiter=hard-disabled all_paths=true' in result.stdout
    assert f"[ProbeConfig] FPLC CPU threads openblas={os.environ.get('OPENBLAS_NUM_THREADS') or '1'}" in result.stdout
    assert 'ee_x=-0.1,0.1 ee_y=-0.6,-0.4 ee_z=0.95,1.2' in result.stdout
    assert 'reset_initial_pose=simcode9_native' in result.stdout
    assert f'policy observation={FPLC_CONTRACTS[profile]["actor_dim"]}D' in result.stdout


@pytest.mark.parametrize('override', [
    {'POLICY_OBS_DIM': '67'}, {'CONTROL_RATE_HZ': '100'},
    {'BALL_OBSERVATION_RATE_HZ': '60'}, {'POLICY_DQ_MODE': 'legacy_dq_only_ablation'},
    {'COMPENSATION_PRESET': 'current'}, {'DONE_EE_Z_LIMITS_M': '1.2,1.3'},
])
def test_launcher_rejects_invalid_contract_overrides(override):
    environment = {**os.environ, 'PROBE_CONFIG_ONLY': '1',
                   'POLICY_RELEASE_ID': 'fplc_gpu1_v172', 'ENABLE_ROS_BRIDGE': 'false',
                   'ROS_PUBLISH_COMMANDS': 'false', 'ENABLE_CAMERA_BALL': 'false',
                   'ENABLE_MOCAP_BALL': 'false', **override}
    result = subprocess.run(['bash', str(REPO_ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh')],
                            cwd=REPO_ROOT, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0


def test_workspace_malformed_and_empty_intersections_fail_closed():
    assert fplc_workspace_intersection(REAL_LIMITS) == ((-.1, .1), (-.6, -.4), (.95, 1.2))
    assert fplc_workspace_intersection(((-1., 1.), (-1., 1.), (0., 2.))) == (
        (-.1, .1), (-.6, -.4), (.95, 1.2))
    smaller = ((-.02, .01), (-.51, -.49), (1.01, 1.10))
    assert fplc_workspace_intersection(smaller) == smaller
    # Valid explicit physical sub-boxes also retain the requested Z extension.
    for z_limits in ((.96, .99), (1.16, 1.19)):
        limits = (*smaller[:2], z_limits)
        assert fplc_workspace_intersection(limits) == limits
    for limits in (REAL_LIMITS[:2], (None, *REAL_LIMITS[1:]),
                   ((float('nan'), .2), *REAL_LIMITS[1:]),
                   (REAL_LIMITS[0], REAL_LIMITS[1], (1.2, 1.3))):
        with pytest.raises(ValueError):
            fplc_workspace_intersection(limits)


def test_frontend_javascript_and_colcon_ignore(tmp_path):
    html = probe_module.HTML_TEMPLATE.replace('__CONFIG_JSON__', '{}')
    scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, flags=re.S)
    script_file = tmp_path / 'probe.js'
    script_file.write_text('\n'.join(scripts))
    subprocess.run(['node', '--check', str(script_file)], check=True, capture_output=True)
    control_function = re.search(r'function syncPolicyModelControl\(\) \{.*?\n\}', html, re.S).group()
    script_file.write_text('''
const policyProfileEl = {value: 'fplc_v220_s17_u839_20261013_r1'};
const inputs = {workspaceSafetyMode: {value: 'off'}};
const policyModelEl = {};
const policyModelRowEl = {classList: {toggle: () => {}}};
const resetInitialPoseEl = {value: 'default_2_horizontal', options: [
  {value: 'default_1'}, {value: 'default_2_horizontal'}, {value: 'simcode9_native'}]};
function setControlDisabled(el, value) { el.disabled = value; }
''' + control_function + '''
syncPolicyModelControl();
if (inputs.workspaceSafetyMode.value !== 'position' ||
    !inputs.workspaceSafetyMode.disabled || !policyModelEl.disabled ||
    resetInitialPoseEl.value !== 'simcode9_native' || !resetInitialPoseEl.disabled ||
    !resetInitialPoseEl.options[1].disabled) process.exit(1);
''')
    subprocess.run(['node', str(script_file)], check=True, capture_output=True)
    geometry_function = re.search(
        r'function updateConfigGeometryFromSnapshot\(snapshot\) \{.*?\n\}', html, re.S).group()
    script_file.write_text('const CONFIG = {};\n' + geometry_function + '''
const small = {x0: -.1, x1: .1, y0: -.6, y1: -.4, z0: .95, z1: 1.2};
const large = {x0: -.348, x1: .263, y0: -.702, y1: -.230, z0: .842, z1: 1.462};
// The first response after selection may still belong to the previous policy.
for (const eeBox of [small, small, large, small, large]) {
  updateConfigGeometryFromSnapshot({eeBox});
  if (JSON.stringify(CONFIG.eeBox) !== JSON.stringify(eeBox)) process.exit(1);
  if (CONFIG.eeBox === eeBox) process.exit(1);
}
updateConfigGeometryFromSnapshot({eeBox: null});
if (CONFIG.eeBox !== null) process.exit(1);
''')
    subprocess.run(['node', str(script_file)], check=True, capture_output=True)
    description = re.search(r'function commandStopDescription\(.*?\n\}', html, re.S).group()
    script_file.write_text(description + '''
const message = commandStopDescription({required: true,
  reason: 'reset_arm_fresh_joint_feedback_required',
  details: {jointFeedbackAgeS: .0496, maximumJointFeedbackAgeS: .010}}, {});
if (!message.includes('49.6 ms') || !message.includes('10.0 ms') ||
    !message.includes('关节反馈不够新鲜')) process.exit(1);
if (commandStopDescription({required: false}, {}) !== '') process.exit(1);
const poseMessage = commandStopDescription({required: true,
  reason: 'fplc_workspace_or_command_boundary_fault',
  details: {faults: ['measured_pose_outside_envelope'],
    relativeRollPitchLimitDeg: [4.0, 4.0],
    measuredPose: {relativeRpyDeg: [1.25, 4.006, 5.0]}}}, {});
if (!poseMessage.includes('实测位姿越界') || !poseMessage.includes('4.006') ||
    !poseMessage.includes('上限 ±4.0/±4.0°')) process.exit(1);
''')
    subprocess.run(['node', str(script_file)], check=True, capture_output=True)
    assert (REPO_ROOT / 'simcode9/COLCON_IGNORE').is_file()


@pytest.mark.parametrize('profile', ['fplc_gpu1_v172', 'fplc_v220_s17_u839_20261013_r1'])
def test_cpu_mirror_carries_direct_observation_separately(profile):
    config = sim_mirror_config_for_policy_profile(SimMirrorConfig(
        model_path=fplc_assets(profile)[0],
        policy_xml_path=fplc_assets(profile)[1],
        sim_xml_path=fplc_assets(profile)[1], policy_obs_dim=72,
        control_dt=.005, enable_video_frames=False), profile)
    mirror = RealBallSimMirror(config)
    try:
        np.testing.assert_array_equal(mirror.data.qpos[mirror.arm_qadr].astype(np.float32),
                                      mirror.policy.native_reset_arm_q)
        np.testing.assert_array_equal(mirror.default_ctrl[mirror.arm_aids], mirror.policy.warm_seed_arm_q)
        assert np.linalg.norm(mirror.warm_arm_q - mirror.policy.native_reset_arm_q) > .1
        direct = np.array([.01, -.48, 1.16], np.float32)
        mirror.step(np.array([.05, -.43, 1.19]), np.array([0, 0, -.3]),
                    ball_valid=True, ball_obs_age_s=.015,
                    options=SimpleNamespace(policy_observation_scales=(), rl_use_valid_ball_obs_age=True),
                    direct_ball_input=dict(direct_ball_position_m=direct,
                                           direct_ball_age_s=.015, ball_observed=True))
        np.testing.assert_array_equal(mirror.policy.last_obs[28:31], direct)
        np.testing.assert_array_equal(mirror.snapshot(ball_valid=True, ball_obs_age_s=.015)['policyBallObservation'],
                                      mirror.policy.last_obs[28:52])
    finally:
        mirror.close()


def test_recorded_actor_features_and_input_alignment():
    obs = list(range(72))
    assert probe_module._policy_observation_feature(obs, 'RacketPositionObs') == [52, 53, 54]
    assert probe_module._policy_observation_feature(obs, 'BallObservationObs') == [28, 29, 30]
    assert probe_module._policy_observation_feature(obs, 'RacketVelocityObs') == []
    assert probe_module._sample_ball_state_6d({'policyObs': obs, 'policyBallObservation': obs[28:52]}) is None
    state = probe_module._sample_ball_state_6d({'policyObs': obs, 'policyBallObservation': obs[28:52],
        'policyInput': {'ball': [1, 2, 3], 'velocity': [4, 5, 6]}})
    np.testing.assert_array_equal(state, [1, 2, 3, 4, 5, 6])
