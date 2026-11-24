"""V325 training workspace selection, browser switching and publish boundary."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.fplc_deployment import (
    FPLC_CONTRACTS, REPO_ROOT, fplc_assets, fplc_real_workspace_limits,
    fplc_workspace_for_profile, fplc_workspace_intersection,
)
from pingpong_controller.mjx_rl_policy import _CheckpointUnpickler
from pingpong_controller.tools import interactive_policy_safety_probe as runtime
from test_simcode9_fplc_contract import REAL_LIMITS, _policy_options, probe

PROFILE = 'fplc_gpu0_v325'
TRAINING_BOX = ((-.348, .348), (-.7, -.28), (1., 1.15))
SAFETY_BOX = ((-.398, .398), (-.75, -.23), (.95, 1.2))
OLD_BOX = ((-.15, .15), (-.6, -.35), (.95, 1.2))


def test_workspace_matches_actual_checkpoint_and_release():
    with fplc_assets(PROFILE)[0].open('rb') as handle:
        cfg = _CheckpointUnpickler(handle).load()['env_cfg']
    frozen = tuple(zip(cfg['cartesian_pose_position_lower_m'],
                       cfg['cartesian_pose_position_upper_m']))
    assert frozen == TRAINING_BOX
    assert FPLC_CONTRACTS[PROFILE]['real_workspace_margin_m'] == .05
    assert fplc_real_workspace_limits(PROFILE) == SAFETY_BOX
    assert fplc_workspace_for_profile(PROFILE, REAL_LIMITS) == SAFETY_BOX
    assert fplc_workspace_for_profile(PROFILE, OLD_BOX) == SAFETY_BOX
    for axis, pair in zip('xyz', frozen):
        assert tuple(cfg[f'fixed_contact_racket_{axis}_bounds_m']) == pair
    manifest = json.loads((REPO_ROOT / 'config/policy_release_manifest.json').read_text())
    release = next(r for r in manifest['releases'] if r['release_id'] == PROFILE)
    assert tuple(zip(release['training_workspace_lower_m'],
                     release['training_workspace_upper_m'])) == frozen


def test_old_models_keep_their_configured_intersection():
    smaller = ((-.04, .03), (-.55, -.45), (1.01, 1.10))
    for profile in FPLC_CONTRACTS:
        if profile not in (PROFILE, 'fplc_gpu0_v325_u416'):
            assert fplc_workspace_for_profile(profile, REAL_LIMITS) == OLD_BOX
            assert fplc_workspace_for_profile(profile, smaller) == smaller
    assert fplc_workspace_intersection(TRAINING_BOX, profile=PROFILE) == TRAINING_BOX
    assert fplc_workspace_intersection(SAFETY_BOX, profile=PROFILE) == SAFETY_BOX
    assert fplc_workspace_intersection(smaller, profile=PROFILE) == smaller


@pytest.mark.parametrize('load_yaml', ['true', 'false'])
def test_launcher_uses_model_safety_box_with_or_without_general_yaml(load_yaml):
    env = {**os.environ, 'PROBE_CONFIG_ONLY': 'true', 'POLICY_RELEASE_ID': PROFILE,
           'LOAD_CONFIG_DONE_MONITORING': load_yaml, 'ENABLE_ROS_BRIDGE': 'false',
           'ROS_PUBLISH_COMMANDS': 'false', 'ENABLE_CAMERA_BALL': 'false',
           'ENABLE_MOCAP_BALL': 'false'}
    result = subprocess.run(['bash', str(REPO_ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh')],
                            cwd=REPO_ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ee_x=-0.398,0.398 ee_y=-0.75,-0.23 ee_z=0.95,1.2' in result.stdout
    assert 'real_roll_pitch=20.0,20.0deg' in result.stdout


def test_browser_switches_boxes_without_carrying_previous_profile_limits(probe):
    for profile in (PROFILE, 'fplc_gpu1_v172', PROFILE,
                    'fplc_v321_s4_u2097_20261029_r1', PROFILE):
        options = _policy_options(probe, profile)
        probe._run_one_control_step(np.array([0., -.49, 1.16]),
                                    np.array([0., 0., -.3]), True, .01, options)
        expected = SAFETY_BOX if profile == PROFILE else OLD_BOX
        assert probe.browser_config()['eeBox'] == runtime.box_payload(*expected)
        assert (probe.cfg.ee_x_limits_m, probe.cfg.ee_y_limits_m,
                probe.cfg.ee_z_limits_m) == expected
        assert (probe.done_monitor.ee_x_limits, probe.done_monitor.ee_y_limits,
                probe.done_monitor.ee_z_limits) == expected
        assert probe.base_ee_limits_m == REAL_LIMITS
        probe._resolve_reset_initial_pose('simcode9_native')


def test_v325_initial_snapshot_uses_model_safety_box_despite_old_general_limits(probe):
    checkpoint, xml = fplc_assets(PROFILE)
    cfg = dataclasses.replace(probe.cfg, policy_profile=PROFILE, policy_obs_dim=114,
                              model_path=checkpoint, robot_xml_path=xml,
                              ee_x_limits_m=OLD_BOX[0], ee_y_limits_m=OLD_BOX[1],
                              ee_z_limits_m=OLD_BOX[2])
    selected = runtime.PolicySafetyProbe(cfg)
    try:
        expected = runtime.box_payload(*SAFETY_BOX)
        assert selected.latest_snapshot_payload['eeBox'] == expected
        assert selected.browser_config()['eeBox'] == expected
        assert selected.base_ee_limits_m == OLD_BOX
    finally:
        selected.shutdown()


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('inside', [True, False])
@pytest.mark.parametrize('checked_pose', ['measured', 'target'])
def test_publish_checks_both_poses_at_each_safety_box_face(
        probe, monkeypatch, axis, side, inside, checked_pose):
    probe._select_policy_profile(PROFILE)
    options = dataclasses.replace(_policy_options(probe, PROFILE), publish_commands=True)
    measured = probe.controller.native_reset_arm_q.copy()
    target = measured.copy()
    target[0] += .0001
    probe.controller.last_cmd_q = target.copy()
    issued = []
    probe.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), .001, 'test'),
        get_joint_feedback=lambda: (measured, np.zeros(7)),
        publish_right_arm_command=lambda q: issued.append(np.array(q)))
    probe.publish_rearm_required = False
    probe.command_stream_active = True
    point = np.array([0., -.49, 1.075])
    inward = 1. if side == 0 else -1.
    point[axis] = SAFETY_BOX[axis][side] + inward * (.001 if inside else -.001)
    original_pose = probe.controller._pose

    def selected_pose(q, **kwargs):
        parts = list(original_pose(q, **kwargs))
        is_measured = np.array_equal(q, measured)
        if is_measured == (checked_pose == 'measured'):
            parts[2] = point.copy()
        return tuple(parts)

    monkeypatch.setattr(probe.controller, '_pose', selected_pose)
    assert probe._publish_control_command(target, options) is inside
    if inside:
        np.testing.assert_array_equal(issued[-1], target)
        assert not probe.publish_rearm_required
    else:
        assert probe._hard_workspace_stop_active() and probe.publish_rearm_required
        fault = f'{checked_pose}_pose_outside_envelope'
        assert fault in probe.publish_rearm_details['faults']
        assert probe.publish_rearm_details['positionLimitsM'] == [list(p) for p in SAFETY_BOX]
