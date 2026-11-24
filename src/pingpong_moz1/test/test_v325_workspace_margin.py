"""Five-centimetre real envelope preserves the authenticated actor projection."""
import numpy as np
import pytest

from pingpong_controller.done_condition_monitor import DoneConditionMonitor
from pingpong_controller.fplc_deployment import FPLC_CONTRACTS, fplc_real_workspace_limits
from pingpong_controller.fplc_policy import FPLCPolicyController

PROFILE = 'fplc_gpu0_v325'
TRAINING_BOX = ((-.348, .348), (-.7, -.28), (1., 1.15))


@pytest.fixture(scope='module')
def controller():
    return FPLCPolicyController(observation_profile=PROFILE)


@pytest.mark.parametrize('axis,position', [(0, -.37), (0, .37), (1, -.73),
                                          (1, -.25), (2, .98), (2, 1.18)])
def test_tracking_margin_does_not_expand_actor_projection(controller, monkeypatch, axis, position):
    point = np.array([0., -.49, 1.075])
    point[axis] = position
    original = controller._pose

    def pose(q, **kwargs):
        parts = list(original(q, **kwargs))
        parts[2] = point.copy()
        parts[3] = np.zeros(3)
        return tuple(parts)

    monkeypatch.setattr(controller, '_pose', pose)
    assert not controller.pose_is_safe(controller.native_reset_arm_q)
    assert controller.pose_is_safe(controller.native_reset_arm_q, fplc_real_workspace_limits(PROFILE))
    np.testing.assert_array_equal(controller.position_lower,
                                  np.array([-.348, -.7, 1.], dtype=np.float32))
    np.testing.assert_array_equal(controller.position_upper,
                                  np.array([.348, -.28, 1.15], dtype=np.float32))


@pytest.mark.parametrize('margin', [-.05, float('nan'), float('inf')])
def test_invalid_model_margin_is_rejected(monkeypatch, margin):
    monkeypatch.setitem(FPLC_CONTRACTS[PROFILE], 'real_workspace_margin_m', margin)
    with pytest.raises(ValueError, match='margin'):
        fplc_real_workspace_limits(PROFILE)


def test_recorded_224830_z_crossing_no_longer_trips_outer_done():
    racket = np.array([-.028239, -.480462, .999944])
    anchor = np.array([0., -.49, 1.025])
    conditions = []
    for box in (TRAINING_BOX, fplc_real_workspace_limits(PROFILE)):
        monitor = DoneConditionMonitor(ee_x_limits=box[0], ee_y_limits=box[1],
                                       ee_z_limits=box[2], enable_predictive_stop=False)
        _, _, flags = monitor.check_done(np.array([-.079499, -.538215, 1.30477]),
                                         np.zeros(3), racket, anchor, check_ball_conditions=False)
        conditions.append(flags['ee_z_out_of_workspace'])
    assert conditions == [True, False]
