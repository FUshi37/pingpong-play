"""Short camera gaps keep causal state, not synthetic measurement freshness."""
import numpy as np
import pytest

from pingpong_controller.ball_observation_model import (
    BallObservationModel, BallObservationProcessor, query_observation_snapshot,
)


def supported_model():
    model = BallObservationModel()
    model.reacquisition_filter_enabled = True
    velocity = np.array([.02, -.01, .7])
    for t in np.arange(10)/90.:
        position = np.array([0., -.49, 1.05]) + t*velocity
        result = model.step(position, velocity, True, t, t,
                            supported=True, velocity_valid=True)
    assert result['valid']
    return model, position, velocity, t


@pytest.mark.parametrize('misses', [1, 2])
def test_support_missing_frames_preserve_state_segment_and_real_range_time(misses):
    model, position, velocity, stamp = supported_model()
    count, segment, real_stamp = len(model.history), model.segment, model.last_range_time
    for index in range(1, misses + 1):
        dt = index/90.
        result = model.step([0., 0., 0.], [0., 0., 0.], False, stamp+dt, stamp+dt)
        assert result['valid'] and result['position_valid']
        assert result['range_age_s'] == pytest.approx(dt)
        assert result['source'] == 'prediction'
        np.testing.assert_allclose(result['position'], position+dt*velocity)
        assert len(model.history) == count and model.segment == segment
        assert model.last_range_time == real_stamp


@pytest.mark.parametrize('misses', [1, 2])
def test_recovery_to_support_does_not_reset_or_mix_predictions_into_history(misses):
    model, position, velocity, stamp = supported_model()
    for index in range(1, misses+1):
        model.step([0., 0., 0.], [0., 0., 0.], False, stamp+index/90., stamp+index/90.)
    now = stamp+(misses+1)/90.
    result = model.step(position+(now-stamp)*velocity, velocity, True, now, now,
                        supported=True, velocity_valid=True)
    assert result['valid'] and result['range_age_s'] == 0.
    assert result['segment'] == 0 and model.history[-1][0] == now
    assert all(t <= stamp or t == now for t, _, _ in model.history)


def test_real_release_after_two_misses_has_a_bounded_velocity_prior():
    model, position, velocity, stamp = supported_model()
    for index in [1, 2]:
        model.step([0., 0., 0.], [0., 0., 0.], False, stamp+index/90., stamp+index/90.)
    dt = 3/90.
    real = position+dt*velocity+.5*dt*dt*model.gravity
    result = model.step(real, np.zeros(3), True, stamp+dt, stamp+dt, supported=False)
    assert result['valid'] and result['position_valid']
    assert not model.supported and result['segment'] == 1
    assert result['range_age_s'] == 0.
    assert len(model.history) == 1
    np.testing.assert_array_equal(model.history[0][1], real)
    assert np.linalg.norm(result['position']-real) < .010


@pytest.mark.parametrize('explicit', ['contact_pending', 'segment_reset'])
def test_explicit_collision_still_clears_support_prior_during_a_miss(explicit):
    model, _, _, stamp = supported_model()
    result = model.step([0., 0., 0.], [0., 0., 0.], False, stamp+1/90., stamp+1/90.,
                        **{explicit: True})
    assert not result['valid'] and model.state is None
    assert model.last_range_time is None


def test_old_support_motion_expires_without_refreshing_real_range_age():
    model, _, _, stamp = supported_model()
    for index in range(1, 7):
        result = model.step([0., 0., 0.], [0., 0., 0.], False, stamp+index/90., stamp+index/90.)
    assert not result['valid'] and not result['position_valid']
    assert result['reason'] == 'supported_prediction_timeout'
    assert model.last_range_time == stamp


@pytest.mark.parametrize('rate', [90, 120, 200])
@pytest.mark.parametrize('misses', [1, 2])
def test_control_queries_during_missing_frames_remain_valid_at_all_state_rates(rate, misses):
    processor = BallObservationProcessor()
    processor.model, _, _, stamp = supported_model()
    for index in range(1, misses+1):
        now = stamp+index/90.
        observation = dict(detected=False, position=np.zeros(3), velocity=np.zeros(3),
            raw_measurement=dict(ball_observation_version=1))
        processor.process(observation, now)
    raw = observation['raw_measurement']
    last_range = processor.model.last_range_time
    for elapsed in np.arange(0., 1/90.+.008, 1/rate):
        result = query_observation_snapshot(raw, now, now+elapsed, [0., 0., 0.])
        assert result['valid']
        assert result['range_age_s'] == pytest.approx(now+elapsed-stamp)
    expired = query_observation_snapshot(raw, now, stamp+.051, [0., 0., 0.])
    assert not expired['valid']
    assert processor.model.last_range_time == last_range


def test_missing_frames_without_an_acquired_ball_cannot_create_a_state():
    model = BallObservationModel()
    for t in np.arange(3)/90.:
        result = model.step(np.zeros(3), np.zeros(3), False, t, t)
        assert not result['valid'] and not result['position_valid']
        assert model.last_range_time is None
