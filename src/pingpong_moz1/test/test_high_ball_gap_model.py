"""Missing-apex prediction, provenance, expiry and live-chain contracts."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pingpong_controller.high_ball_gap_model import HighBallGapConfig, HighBallGapModel
from pingpong_controller.high_ball_occlusion_predictor import HighBallOcclusionConfig, HighBallOcclusionPredictor


def flight(t):
    return np.array([.08*t, -.49+.04*t, 1.35+1.4*t-4.9*t*t])


def velocity(t):
    return np.array([.08, .04, 1.4-9.8*t])


def seed(model, times=np.arange(10)/90.):
    for t in times:
        model.observe(t, flight(t))


def runtime():
    model = HighBallOcclusionPredictor(HighBallOcclusionConfig(
        enable_high_ball_gap_prediction=True,
        enable_optimized_initial_state=True, enable_reacquisition_smoothing=True))
    for t in np.arange(10)/90.:
        model.step(flight(t), velocity(t), True, t, t)
    return model


def test_missing_apex_matches_physics_without_future_measurements():
    model = HighBallGapModel()
    seed(model)
    for t in np.arange(.11, .281, .01):
        p, v, info = model.predict(t)
        np.testing.assert_allclose(p, flight(t), atol=1e-12)
        np.testing.assert_allclose(v, velocity(t), atol=1e-12)
        assert info['elapsed_s'] == pytest.approx(t-.1)
    assert len(model.history) == 10


def test_descending_near_apex_is_eligible():
    model = HighBallGapModel()
    seed(model, np.arange(.10, .201, .01))
    assert velocity(.20)[2] < 0
    np.testing.assert_allclose(model.predict(.23)[0], flight(.23), atol=1e-12)


def test_no_static_high_ball_or_low_ball_prediction():
    for height in [1.1, 1.45]:
        model = HighBallGapModel()
        for t in np.arange(10)/90.:
            model.observe(t, [0, -.49, height])
        assert model.predict(.12) is None
    model = HighBallGapModel()
    for t in np.arange(10)/90.:
        model.observe(t, flight(t)-[0, 0, .25])
    assert model.predict(.12) is None


def test_predictions_and_duplicate_stamps_cannot_renew_budget():
    model = HighBallGapModel()
    seed(model)
    for t in np.arange(.11, .29, .005):
        model.observe(.1+2e-7, flight(.1))
        model.predict(t)
    assert len(model.history) == 10
    assert model.predict(.301) is None
    assert model.reason == 'prediction_timeout'
    assert model.predict(.32) is None


def test_noise_and_uncertainty_stop_unreliable_extrapolation():
    model = HighBallGapModel()
    for i, t in enumerate(np.arange(10)/90.):
        model.observe(t, flight(t)+[.012*(-1)**i, 0, 0])
    assert model.predict(.12) is None
    assert model.reason == 'trajectory_fit_residual_too_large'
    model = HighBallGapModel(HighBallGapConfig(max_uncertainty_m=.005))
    seed(model)
    assert model.predict(.11) is not None
    assert model.predict(.25) is None
    assert model.reason == 'prediction_uncertainty_limit'


def test_reset_and_long_real_sample_gap_require_new_evidence():
    model = HighBallGapModel()
    seed(model)
    model.observe(.18, flight(.18))
    assert len(model.history) == 1
    assert model.predict(.19) is None
    model.reset()
    assert len(model.history) == 0
    assert model.predict(.20) is None


def test_live_chain_fills_high_loss_and_labels_model_state():
    model = runtime()
    for t in [.12, .16, .20, .24]:
        result = model.step(np.zeros(3), np.zeros(3), False, t, t)
        assert result.valid and result.predicted
        assert result.diagnostics['activeTrigger'] == 'high_ball_gap'
        assert result.diagnostics['positionPredicted']
        assert result.diagnostics['highGapUncertaintyMm'] > 0
        np.testing.assert_allclose(result.position_m, flight(t), atol=1e-12)


def test_live_high_depth_fault_uses_bounded_high_model():
    model = runtime()
    result = model.step([100, 100, 100], [0, 0, 0], False, .12, .12,
                        force_prediction_reason='sensor_depth_anomaly')
    assert result.diagnostics['activeTrigger'] == 'high_ball_gap'
    assert len(model._high_gap_model.history) == 10
    assert not model.step([0]*3, [0]*3, False, .301, .301).valid
    assert not model.step([0]*3, [0]*3, False, .32, .32).valid


def test_reacquisition_uses_fresh_position_and_bounded_velocity_bridge():
    model = runtime()
    model.step([0]*3, [0]*3, False, .12, .12)
    fresh = flight(.22)+[.002, 0, 0]
    result = model.step(fresh, [0]*3, True, .22, .22,
                        measurement_velocity_valid=False)
    np.testing.assert_allclose(result.position_m, fresh, rtol=0, atol=1e-7)
    assert result.valid and result.diagnostics['velocityPredicted']
    assert not result.diagnostics['positionPredicted']
    # Repeated position-only input cannot keep the old model velocity alive.
    stale = model.step(fresh, [0]*3, True, .22, .31,
                       measurement_velocity_valid=False)
    assert not stale.valid


def test_rejected_reacquisition_cannot_pollute_history_or_bypass_expiry():
    model = runtime()
    model.step([0]*3, [0]*3, False, .12, .12)
    size = len(model._high_gap_model.history)
    bad = model.step(flight(.16)+[.1, 0, 0], velocity(.16), True, .16, .16)
    assert bad.diagnostics['reason'] == 'reacquisition_outside_gate'
    assert bad.diagnostics['positionPredicted']
    assert len(model._high_gap_model.history) == size
    assert not model.step([0]*3, [0]*3, False, .31, .31).valid


def test_invalid_configuration_rejected():
    for kwargs in [dict(max_prediction_s=0), dict(max_fit_error_m=float('nan')),
                   dict(min_samples=2), dict(min_span_s=.2)]:
        with pytest.raises(ValueError):
            HighBallGapConfig(**kwargs)
