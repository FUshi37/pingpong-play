"""Weak silhouette-range evidence; never renew the stereo measurement age."""
import numpy as np


def update_radius_observation(model, raw, now, origin, direction):
    measured=raw.get('monocular_range_m')
    sigma=raw.get('monocular_range_sigma_m')
    if (measured is None or sigma is None or model.supported
            or raw.get('range_quality_reason')!='sparse_stereo'
            or not model.velocity_valid or origin is None or direction is None):
        return False
    if not np.isfinite([measured,sigma]).all() or not .15<measured<1. or not .005<=sigma<=.05:
        return False
    predicted=model._propagated(now)
    if predicted is None:return False
    state,covariance=predicted
    origin=np.asarray(origin,float);direction=np.asarray(direction,float)
    if (origin.shape!=(3,) or direction.shape!=(3,)
            or not np.isfinite([*origin,*direction]).all() or np.linalg.norm(direction)<.5):
        return False
    direction=direction/np.linalg.norm(direction)
    innovation=float(measured-(state[:3]-origin)@direction)
    if abs(innovation)>.025:return False
    previous=model.last_radius_time;model.last_radius_time=now
    if previous is None or not .005<=now-previous<=.030:return False
    h=np.r_[direction,np.zeros(3)];noise=float(sigma)**2
    gain=covariance@h/(h@covariance@h+noise)
    residual=np.eye(6)-np.outer(gain,h)
    model.state=state+gain*innovation
    model.covariance=residual@covariance@residual.T+noise*np.outer(gain,gain)
    model.state_time=now;model.last_radius_accepted_time=now
    raw['observation_radius_range_accepted']=True
    return True
