import numpy as np

from pingpong_controller.ball_observation_model import (
    BallObservationModel, BallObservationProcessor, query_observation_snapshot,
)


def trajectory(t):
    return np.array([.1+.2*t, -.4+.1*t, 1.35+1.8*t-4.9*t*t])


def seeded():
    model = BallObservationModel()
    for t in np.arange(10)/90.:
        result = model.step(trajectory(t), np.zeros(3), True, t, t)
    assert result['valid']
    return model


def test_bearing_updates_recover_lateral_motion_without_inventing_range_samples():
    model = seeded()
    count, last_range = len(model.history), model.last_range_time
    origin = np.array([0., -1., 1.8])
    for t in np.arange(10, 20)/90.:
        ray = trajectory(t)-origin
        result = model.step(np.zeros(3), np.zeros(3), False, t, t,
                            bearing_origin=origin, bearing_direction=ray/np.linalg.norm(ray))
        assert result['valid'] and result['bearing_accepted']
        np.testing.assert_allclose(result['position'], trajectory(t), atol=1e-8)
    assert len(model.history) == count
    assert model.last_range_time == last_range


def test_repeated_control_queries_do_not_reinsert_or_shrink_covariance():
    model = seeded()
    timestamp = model.last_frame_time
    covariance = model.covariance.copy()
    history = list(model.history)
    for dt in [.001, .003, .005, .007]:
        result = model.step(trajectory(timestamp), np.zeros(3), True, timestamp, timestamp+dt)
        np.testing.assert_allclose(result['position'], trajectory(timestamp+dt), atol=1e-8)
    np.testing.assert_array_equal(model.covariance, covariance)
    assert [r[0] for r in model.history] == [r[0] for r in history]


def test_bearing_frames_and_duplicate_queries_cannot_extend_prediction_deadline():
    model = seeded()
    last_range = model.last_range_time
    for t in np.arange(10, 35)/90.:
        ray = trajectory(t)/np.linalg.norm(trajectory(t))
        result = model.step(np.zeros(3), np.zeros(3), False, t, t,
                            bearing_origin=np.zeros(3), bearing_direction=ray)
    assert t-last_range > .2
    assert not result['valid'] and not result['position_valid']
    assert result['reason'] == 'prediction_timeout'


def test_single_large_range_outlier_is_not_published_or_added_to_fit():
    model = seeded()
    count = len(model.history)
    t = 10/90.
    bad = trajectory(t)+np.array([0., 0., .15])
    result = model.step(bad, np.zeros(3), True, t, t)
    assert not result['position_valid']
    assert result['range_rejected'] and len(model.history) == count
    t = 11/90.
    result = model.step(trajectory(t), np.zeros(3), True, t, t)
    assert result['valid']
    np.testing.assert_allclose(result['position'], trajectory(t))


def test_confirmed_new_segment_never_blends_old_position_or_velocity():
    model = seeded()
    t = .12
    first = np.array([.2, -.4, 1.1])
    result = model.step(first, np.array([0., 0., -3.]), True, t, t,
                        segment_reset=True)
    np.testing.assert_allclose(result['position'], first)
    assert result['position_valid'] and not result['valid']
    second = first+np.array([0., 0., .02])
    result = model.step(second, np.array([0., 0., -3.]), True, t+.01, t+.01)
    assert result['valid'] and result['velocity'][2] > 1.8
    np.testing.assert_allclose(result['position'], second)


def test_two_fresh_ranges_can_reacquire_when_impact_event_was_missed():
    model = seeded()
    t = .12
    first = np.array([.2, -.4, 1.1])
    result = model.step(first, np.zeros(3), True, t, t)
    assert not result['valid']
    second = first+np.array([0., 0., .02])
    result = model.step(second, np.zeros(3), True, t+.01, t+.01)
    assert result['valid'] and result['velocity'][2] > 1.8
    assert result['segment'] == 1
    np.testing.assert_allclose(result['position'], second)


def test_contact_pending_invalidates_old_flight_even_with_a_good_position():
    model = seeded()
    t = .12
    result = model.step(trajectory(t), np.array([0., 0., -3.]), True, t, t,
                        contact_pending=True)
    assert result['position_valid'] and not result['valid']
    np.testing.assert_allclose(result['position'], trajectory(t))
    assert model.state is None


def test_long_unobserved_low_fall_cannot_predict_through_unknown_contact():
    model = BallObservationModel(prediction_floor_m=1.24)
    for t in np.arange(10)/90.:
        model.step(np.array([0., 0., 1.3-1.5*t-4.9*t*t]), np.zeros(3), True, t, t)
    t = .18
    result = model.step(np.zeros(3), np.zeros(3), False, t, t)
    assert not result['position_valid']
    assert result['reason'] == 'prediction_below_contact_band'


def test_snapshot_consumer_is_stateless_and_never_runs_linear_algebra(monkeypatch):
    processor = BallObservationProcessor()
    for t in np.arange(10)/90.:
        observation = dict(detected=True, position=1000*trajectory(t), velocity=np.zeros(3),
            raw_measurement=dict(ball_observation_version=1, velocity_ready=True, range_sigma_mm=3.))
        processor.process(observation, t)
    raw = observation['raw_measurement'].copy()
    def forbidden(*args, **kwargs):
        raise AssertionError('fitting/matrix work reached the control reader')
    for method in ['lstsq', 'solve', 'inv', 'eigvalsh', 'svd']:
        monkeypatch.setattr(np.linalg, method, forbidden)
    for dt in [.0, .005, .010, .015]:
        result = query_observation_snapshot(raw, t, t+dt, [0., 0., 0.])
        assert result['valid']
        np.testing.assert_allclose(result['position'], trajectory(t+dt), atol=1e-8)
    assert raw == observation['raw_measurement']


def test_snapshot_can_expire_without_any_new_camera_frame():
    processor = BallObservationProcessor()
    for t in np.arange(10)/90.:
        observation = dict(detected=True, position=1000*trajectory(t), velocity=np.zeros(3),
            raw_measurement=dict(ball_observation_version=1, velocity_ready=True, range_sigma_mm=3.))
        processor.process(observation, t)
    result = query_observation_snapshot(observation['raw_measurement'], t, t+.201, [0., 0., 0.])
    assert not result['valid'] and not result['position_valid']
    assert result['reason'] == 'prediction_timeout'


def test_control_query_cannot_extrapolate_preimpact_state_through_racket():
    processor = BallObservationProcessor()
    for t in np.arange(10)/90.:
        position = np.array([0., 0., 1.3-2*t-4.9*t*t])
        observation = dict(detected=True, position=1000*position, velocity=np.zeros(3),
            raw_measurement=dict(ball_observation_version=1, velocity_ready=True, range_sigma_mm=3.))
        processor.process(observation, t, racket_state=(t, [0., 0., 1000.], [0., 0., 0.]))
    raw = observation['raw_measurement']
    assert 0 < raw['observation_contact_horizon_s'] < .01
    current = query_observation_snapshot(raw, t, t, [0., 0., 0.])
    assert current['position_valid']
    future = query_observation_snapshot(raw, t, t+.015, [0., 0., 0.])
    assert not future['valid'] and not future['position_valid']
    assert future['reason'] == 'contact_prediction_limit'


def test_position_without_velocity_is_not_reported_at_a_later_control_time():
    processor = BallObservationProcessor()
    observation = dict(detected=True, position=1000*trajectory(0.), velocity=np.zeros(3),
        raw_measurement=dict(ball_observation_version=1, velocity_ready=False, range_sigma_mm=3.))
    processor.process(observation, 0.)
    raw = observation['raw_measurement']
    assert query_observation_snapshot(raw, 0., 0., [0., 0., 0.])['position_valid']
    later = query_observation_snapshot(raw, 0., .02, [0., 0., 0.])
    assert not later['valid'] and not later['position_valid']
