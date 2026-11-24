"""Contact evidence, uncertainty and expiry; predictions never become samples."""
from types import SimpleNamespace
import numpy as np
from pingpong_controller.ball_contact_refinement import ContactRefinement
from pingpong_controller.ball_contact_observation import BallContactObservation
from pingpong_controller.ball_observation_model import BallObservationModel


def setup():
    g=np.array([0.,0.,-9.8]);actual=.1;planned=.106
    pc=np.array([0.,0.,1.]);vin=np.array([0.,0.,-2.]);vout=np.array([0.,0.,2.])
    def incoming(t):
        dt=t-actual;return np.r_[pc+dt*vin+.5*dt*dt*g,vin+dt*g]
    model=SimpleNamespace(gravity=g,_propagated=lambda t:(incoming(t),np.eye(6)))
    plan=dict(time=planned,position=incoming(planned)[:3],velocity=vout)
    contact=SimpleNamespace(plan=plan,seeded=None)
    def observed(t):
        dt=t-actual;return pc+dt*vout+.5*dt*dt*g
    return ContactRefinement(),contact,model,observed


def test_contact_time_and_position_are_reestimated_together_from_real_evidence():
    refine,c,m,observed=setup();raw={};old=c.plan.copy()
    refine.refine(c,m,raw,observed(.113),True,.113)
    assert raw['observation_contact_refined']
    assert c.plan['time'] < old['time']-.002
    assert abs(c.plan['time']-.1)<.004
    np.testing.assert_allclose(c.plan['position'],m._propagated(c.plan['time'])[0][:3])
    assert c.plan['velocity'][2]>.25
    assert len(refine.samples)==1
    refine.refine(c,m,{},np.zeros(3),False,.124)
    assert len(refine.samples)==1 # missing/predicted frames cannot enter the fit


def test_consistent_plan_and_future_contact_are_not_rewritten():
    refine,c,m,_=setup();original=c.plan
    t=.105;refine.refine(c,m,{},m._propagated(t)[0][:3],True,t)
    assert c.plan is original
    t=.11;dt=t-c.plan['time'];p=c.plan['position']+dt*c.plan['velocity']+.5*dt*dt*m.gravity
    refine.refine(c,m,{},p,True,t)
    assert c.plan is original


def test_same_event_refinement_does_not_seed_a_second_segment():
    refine,c,m,observed=setup();c.seeded=c.plan['time']
    refine.refine(c,m,{},observed(.113),True,.113)
    assert c.seeded==c.plan['time']
    assert m.contact_velocity_prior[0]==c.seeded


def test_two_exact_points_still_have_nonzero_velocity_uncertainty():
    m=BallObservationModel();m.fast_segment=True;m.contact_velocity_prior=(0.,np.array([0.,0.,2.]))
    for t in (0.,1./90.):
        m.history.append((t,np.array([0.,0.,1.+2*t-4.9*t*t]),.003))
    assert m._fit(1./90.)
    assert m.fit_rmse<1e-9
    assert .01<m.covariance[5,5]<.1
    assert np.linalg.eigvalsh(m.covariance).min()>0


def test_tilted_measured_plane_changes_reflected_direction_and_stale_feedback_expires():
    m=BallObservationModel();m.state=np.array([0.,0.,1.1,0.,0.,-2.]);m.state_time=1.
    m.covariance=np.eye(6)*1e-6;m.velocity_valid=True
    c=BallContactObservation();n=np.array([.1,0.,1.]);n/=np.linalg.norm(n)
    c.context=dict(normal=n,stamp=1.,velocity_valid=True)
    c.prepare(m,{},m.state[:3],True,1.,(1.,[0,0,1000],[0,0,0]))
    assert c.plan is not None and c.plan['velocity'][0]>.1
    assert c.racket_motion((1.,[0,0,1000],[0,0,0]),1.061) is None
