"""A bounded camera dropout reaches the real FPLC adapter without rearming."""
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller.tools.interactive_policy_safety_probe import PolicySafetyProbe
from pingpong_controller.ball_observation_model import BallObservationProcessor, query_observation_snapshot
from test_camera_short_dropout import supported_model


@pytest.mark.parametrize('rate', [90, 120, 200])
@pytest.mark.parametrize('misses', [1, 2])
def test_missing_camera_frames_pass_the_real_ball_handover_and_actor(rate, misses):
    holder = PolicySafetyProbe.__new__(PolicySafetyProbe)
    holder.cfg = SimpleNamespace(control_dt=.005, ball_lost_timeout_s=.6)
    holder._reset_policy_ball_sampling_contract()
    options = SimpleNamespace(policy_profile='fplc_gpu1_v172', ball_source='camera',
        use_ball_estimator=False, control_mode='policy', ball_observation_rate_hz=float(rate),
        enable_ball_dropout_reference_hold=False)
    actor = FPLCPolicyController(observation_profile='fplc_gpu1_v172')
    actor._policy_action = lambda obs: np.zeros(7, np.float32)
    processor = BallObservationProcessor()
    processor.model, p, v, stamp = supported_model()
    actor.predict(p, v, ball_valid=True, ball_obs_age_s=.01,
        arm_q=actor.native_reset_arm_q, arm_dq=np.zeros(7), ball_observed=True,
        direct_ball_position_m=p, direct_ball_age_s=.01, direct_ball_stamp_s=stamp,
        ball_observation_rate_hz=rate, enable_dropout_reference_hold=False)
    for index in range(1, misses+1):
        now = stamp+index/90.
        obs = dict(detected=False, position=np.zeros(3), velocity=np.zeros(3),
                   raw_measurement=dict(ball_observation_version=1))
        processor.process(obs, now)
        q = query_observation_snapshot(obs['raw_measurement'], now, now+.008, np.zeros(3))
        holder.last_high_ball_occlusion_info = dict(model='quality_bearing_range_flight',
            observationRangeAgeS=q['range_age_s'], positionPredicted=True, velocityPredicted=True)
        holder.fplc_direct_ball_input = dict(direct_ball_position_m=np.zeros(3),
            direct_ball_age_s=.6, direct_ball_stamp_s=now, ball_observed=False,
            ball_state_already_propagated=True)
        position, velocity, valid, age = holder._apply_policy_ball_sampling_contract(
            q['position'], q['velocity'], q['valid'], 0., options, source_timestamp_s=now)
        assert valid and not holder.fplc_direct_ball_input['ball_observed']
        actor.predict(position, velocity, ball_valid=valid, ball_obs_age_s=age,
            arm_q=actor.native_reset_arm_q, arm_dq=np.zeros(7), **holder.fplc_direct_ball_input)
        assert not actor._fplc_requires_measured_reset
        assert holder.last_policy_ball_handover_info['representedStateValid']
        if rate > 90:
            assert holder.fplc_direct_ball_input['ball_history_measurement_age_s'] == pytest.approx(
                now+.008-stamp)
    assert processor.model.last_range_time == stamp
