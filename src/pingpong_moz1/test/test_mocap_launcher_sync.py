"""Exercise launch-to-receiver clock settings without starting ROS or UDP."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pingpong_controller.tools import interactive_policy_safety_probe as probe

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh'


def launch_environment(tmp_path, **overrides):
    return {**os.environ, 'WORKSPACE_DIR': str(REPO_ROOT),
        'POLICY_RELEASE_ID': 'fplc_gpu1_v172', 'POLICY_PROFILE': 'fplc_gpu1_v172',
        'ENABLE_ROS_BRIDGE': 'false', 'ROS_PUBLISH_COMMANDS': 'false',
        'ENABLE_CAMERA_BALL': 'false', 'ENABLE_MOCAP_BALL': 'true',
        'ENABLE_SIM_MIRROR': 'false', 'PROBE_CONTROL_SUBPROCESS': 'false',
        'PROBE_CONFIG_ONLY': 'true', 'LOAD_CONFIG_DONE_MONITORING': 'false',
        'SETUP_FILE': str(tmp_path / 'setup.bash'),
        'ROS_LOG_DIR': str(tmp_path / 'ros_log'),
        'MOCAP_CLOCK_SYNC_MIN_SAMPLES': '8', 'MOCAP_CLOCK_SYNC_MIN_SPAN_S': '0.05',
        'MOCAP_CLOCK_SYNC_WINDOW_S': '10.0', **overrides}


def run_launcher(tmp_path, **overrides):
    return subprocess.run(['bash', str(LAUNCHER)], cwd=REPO_ROOT,
        env=launch_environment(tmp_path, **overrides), capture_output=True,
        text=True, timeout=30)


@pytest.mark.parametrize('enabled', ['true', '1', 'TRUE'])
def test_enabled_mode_requires_sync_in_config_only_without_initializing_ros(tmp_path, enabled):
    (tmp_path / 'setup.bash').write_text('exit 92\n')
    result = run_launcher(tmp_path, ENABLE_MOCAP_BALL=enabled)
    assert result.returncode == 0, result.stderr
    assert 'mocap timestamp_sync=required receiver_clock=ROS' in result.stdout
    assert 'min_samples=8 min_span=0.05s window=10.0s' in result.stdout
    assert 'unsynchronized_observation=invalid' in result.stdout
    assert not (tmp_path / 'ros_log').exists()


@pytest.mark.parametrize('override', [
    {'MOCAP_CLOCK_SYNC_MIN_SAMPLES': '1'},
    {'MOCAP_CLOCK_SYNC_MIN_SAMPLES': '8.5'},
    {'MOCAP_CLOCK_SYNC_MIN_SPAN_S': 'nan'},
    {'MOCAP_CLOCK_SYNC_MIN_SPAN_S': '0'},
    {'MOCAP_CLOCK_SYNC_WINDOW_S': 'inf'},
    {'MOCAP_CLOCK_SYNC_WINDOW_S': '0.01'},
])
def test_invalid_sync_settings_fail_before_receiver_start(tmp_path, override):
    result = run_launcher(tmp_path, **override)
    assert result.returncode == 2
    assert 'invalid mocap clock synchronization settings' in result.stderr
    assert not (tmp_path / 'ros_log').exists()


def test_disabled_mocap_does_not_apply_or_validate_sync_settings(tmp_path):
    result = run_launcher(tmp_path, ENABLE_MOCAP_BALL='false',
        MOCAP_CLOCK_SYNC_MIN_SAMPLES='unused', MOCAP_CLOCK_SYNC_MIN_SPAN_S='unused')
    assert result.returncode == 0, result.stderr
    assert 'mocap timestamp_sync=inactive' in result.stdout


@pytest.mark.parametrize('subprocess_mode', [True, False])
@pytest.mark.parametrize('profile', ['fplc_gpu1_v172', 'fplc_v220_s17_u839_20261013_r1'])
def test_actual_launch_arguments_reach_mocap_receiver_and_control_gate(tmp_path, subprocess_mode, profile):
    capture = tmp_path / 'backend_arguments.json'
    wrapper = tmp_path / 'python3'
    wrapper.write_text(f'#!{sys.executable}\n' + '''
import json
import os
from pathlib import Path
import sys
if len(sys.argv) > 1 and Path(sys.argv[1]).name == 'interactive_policy_safety_probe.py':
    Path(os.environ['MOCAP_TEST_CAPTURE']).write_text(json.dumps(sys.argv[2:]))
    raise SystemExit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    wrapper.chmod(0o755)
    (tmp_path / 'setup.bash').write_text('return 0\n')
    result = run_launcher(tmp_path, PROBE_CONFIG_ONLY='false',
        POLICY_RELEASE_ID=profile, POLICY_PROFILE=profile,
        PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
        MOCAP_SUBPROCESS='true' if subprocess_mode else 'false',
        MOCAP_TEST_CAPTURE=str(capture), MOCAP_CLOCK_SYNC_MIN_SAMPLES='16',
        MOCAP_CLOCK_SYNC_MIN_SPAN_S='0.2', MOCAP_CLOCK_SYNC_WINDOW_S='3.0')
    assert result.returncode == 0, result.stderr
    args = probe.build_arg_parser().parse_args(json.loads(capture.read_text()))
    cfg = probe.config_from_args(args)
    assert cfg.policy_profile == profile
    assert cfg.policy_obs_dim == 72
    assert cfg.control_rate_hz == 200
    assert cfg.ball_observation_rate_hz == 90
    assert cfg.workspace_safety_mode == 'position'
    assert cfg.reset_initial_pose_id == 'simcode9_native'
    assert not cfg.enable_safety_limiter
    assert not cfg.enable_compensation
    assert cfg.policy_plant_matcher_mode == 'off'
    assert not cfg.enable_policy_dq_adapter
    assert not cfg.enable_contact_action_shield
    assert not cfg.enable_vertical_contact_energy_shield
    # The FPLC launcher must pass the original envelope so other profiles can
    # restore it after the backend applies the FPLC-specific intersection.
    assert (cfg.ee_x_limits_m, cfg.ee_y_limits_m, cfg.ee_z_limits_m) == (
        (-.298, .163), (-.502, -.290), (.692, 1.462))
    assert cfg.enable_mocap_ball
    assert cfg.mocap_marker_x_limits_m == (-2., 2.)
    assert cfg.mocap_subprocess is subprocess_mode
    assert not cfg.enable_camera_ball
    assert (cfg.mocap_clock_sync_min_samples, cfg.mocap_clock_sync_min_span_s,
            cfg.mocap_clock_sync_window_s) == (16, .2, 3.)
    node = probe.RosBridgeNode.__new__(probe.RosBridgeNode)
    node.cfg = cfg
    node.now_sec = lambda: 1800000000.
    node.get_logger = lambda: SimpleNamespace(info=mock.Mock(), warn=mock.Mock())
    from pingpong_controller.mocap_process import MocapInputProcess
    input_class = MocapInputProcess if subprocess_mode else probe.MocapBallInput
    # Intercept receiver startup in either mode; construction must preserve
    # the same calibration/clock-fit settings in the original detector.
    with mock.patch.object(input_class, 'start', return_value=True):
        node._init_mocap_ball_source()
    if subprocess_mode:
        adapter = probe.MocapBallInput(**node.mocap_input.kwargs,
                                     receiver_clock=node.mocap_input.receiver_clock)
    else:
        adapter = node.mocap_input
    sync = adapter.detector.clock_sync
    assert adapter.detector.marker_x_limits_m == (-2., 2.)
    assert (sync.min_samples, sync.min_span_s, sync.window_s) == (16, .2, 3.)
    assert adapter.detector._receiver_clock() == node.now_sec()
    assert adapter.detector._socket is None
    assert not node.mocap_input.get_control_state()[2]


def test_backend_rejects_invalid_clock_settings_before_runtime_initialization():
    args = probe.build_arg_parser().parse_args([
        '--policy-profile', 'fplc_gpu1_v172', '--mocap-ball',
        '--mocap-clock-sync-min-samples', '1'])
    with pytest.raises(ValueError, match='clock synchronization'):
        probe.config_from_args(args)
