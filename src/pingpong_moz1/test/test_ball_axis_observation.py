import numpy as np
from types import SimpleNamespace

from pingpong_controller.ball_axis_observation import retain_horizontal_edge, update_horizontal_edge
from pingpong_controller.ball_observation_model import BallObservationModel
from pingpong_controller.ball_axis_observation import apply_contact_timing_mean
from pingpong_controller.ball_observation_model import query_observation_snapshot


def model_with_edge(candidates=((4., 0., 20.),)):
    model = BallObservationModel()
    model.state = np.array([0., 0., .5, 0., 0., 0.])
    model.covariance = np.diag([.004**2] * 3 + [.05**2] * 3)
    model.covariance[0, 3] = model.covariance[3, 0] = .00005
    model.state_time = model.last_frame_time = model.last_range_time = 10.
    model.velocity_valid = True
    model.axis_refinement_enabled = True
    geometry = dict(inverse=np.eye(3), origin=np.zeros(3), fx=400., fy=400., cx=0., cy=0.)
    model.current_raw = {'_horizontal_edge_observation': (geometry, candidates)}
    return model


def test_edge_changes_only_horizontal_state_and_preserves_range_age():
    model = model_with_edge()
    prior = model._propagated(10.01)
    assert update_horizontal_edge(model, 10.01)
    assert model.state[0] > prior[0][0]
    np.testing.assert_array_equal(model.state[[1, 2, 4, 5]], prior[0][[1, 2, 4, 5]])
    assert model.last_range_time == 10.
    assert len(model.history) == 0
    assert model.last_radius_accepted_time is None
    assert np.linalg.eigvalsh(model.covariance).min() >= -1e-12


def test_missing_prior_ambiguous_and_outside_association_are_rejected():
    model = model_with_edge()
    model.state = None
    assert not update_horizontal_edge(model, 10.01)
    for candidates in (((4., 0., 20.), (10., 0., 20.)), ((25., 0., 20.),), ((4., 100., 20.),)):
        model = model_with_edge(candidates)
        old = model.state.copy()
        assert not update_horizontal_edge(model, 10.01)
        np.testing.assert_array_equal(model.state, old)


def test_edge_cannot_extend_expired_range_or_duplicate_frame_updates():
    model = model_with_edge()
    result = model.step(np.zeros(3), np.zeros(3), False, 10.3, 10.3)
    assert not result['position_valid']
    assert not result['bearing_accepted']
    model = model_with_edge()
    model.step(np.zeros(3), np.zeros(3), False, 10.01, 10.01)
    state, cov = model.state.copy(), model.covariance.copy()
    model.step(np.zeros(3), np.zeros(3), False, 10.01, 10.01)
    np.testing.assert_array_equal(model.state, state)
    np.testing.assert_array_equal(model.covariance, cov)


def test_retention_is_bounded_and_excludes_horizontal_clipping():
    detector = SimpleNamespace(_axis_refinement_enabled=True, _horizontal_edges=[], _horizontal_edges_ambiguous=False)
    retain_horizontal_edge(detector, 10., 0., 20., (480, 848))
    assert not detector._horizontal_edges
    retain_horizontal_edge(detector, 100., 0., 20., (480, 848))
    retain_horizontal_edge(detector, 101., 1., 20., (480, 848))
    assert len(detector._horizontal_edges) == 1
    for x in range(130, 700, 30):
        retain_horizontal_edge(detector, x, 0., 20., (480, 848))
    assert len(detector._horizontal_edges) == 8
    assert detector._horizontal_edges_ambiguous


def contact_snapshot(enabled=True):
    raw = dict(ball_axis_refinement=enabled, observation_velocity_valid=True,
        observation_position_valid=True, observation_range_age_s=.01,
        observation_position_variance=.002**2, observation_velocity_variance=.02**2,
        observation_cross_covariance=0., observation_prediction_budget_s=.32,
        observation_bounce_horizon_s=.01, observation_source='measured',
        observation_contact_model_active=True, observation_contact_model_until_s=.05)
    for k,p,v,bp,bv in zip('xyz',[0.,0.,1.2],[.1,.2,-2.],[.001,.002,1.17951],[.1,.2,2.]):
        raw['observation_position_'+k+'_m']=p;raw['observation_velocity_'+k+'_m_s']=v
        raw['observation_bounce_position_'+k+'_m']=bp;raw['observation_bounce_velocity_'+k+'_m_s']=bv
    return raw


def test_contact_mean_is_continuous_and_velocity_matches_derivative():
    raw=contact_snapshot()
    for t in (.008,.009999,.010001,.012):
        q=query_observation_snapshot(raw,0.,t,np.zeros(3),prediction_floor_m=1.)
        dt=1e-7
        a=query_observation_snapshot(raw,0.,t-dt,np.zeros(3),prediction_floor_m=1.)
        b=query_observation_snapshot(raw,0.,t+dt,np.zeros(3),prediction_floor_m=1.)
        np.testing.assert_allclose((b['position']-a['position'])/(2*dt),q['velocity'],atol=1e-6)
        old=query_observation_snapshot(contact_snapshot(False),0.,t,np.zeros(3),prediction_floor_m=1.)
        np.testing.assert_array_equal(q['position'][:2],old['position'][:2])
        assert q['position'][2]>=old['position'][2]
        assert q['position'][2]-old['position'][2]<.006
    a=query_observation_snapshot(raw,0.,.01-1e-8,np.zeros(3),prediction_floor_m=1.)
    b=query_observation_snapshot(raw,0.,.01+1e-8,np.zeros(3),prediction_floor_m=1.)
    np.testing.assert_allclose(a['position'],b['position'],atol=1e-7)
    np.testing.assert_allclose(a['velocity'],b['velocity'],atol=2e-5)


def test_contact_mean_never_revalidates_expired_or_future_snapshots():
    for t in (-.01,.052,.4):
        old=query_observation_snapshot(contact_snapshot(False),0.,t,np.zeros(3),prediction_floor_m=1.)
        new=query_observation_snapshot(contact_snapshot(),0.,t,np.zeros(3),prediction_floor_m=1.)
        assert new['position_valid']==old['position_valid']==False
        assert new['reason']==old['reason']
        np.testing.assert_array_equal(new['position'],old['position'])
