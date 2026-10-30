"""Fixed observation calibration, loaded once and bound to camera geometry."""
import hashlib
import json
from pathlib import Path
import numpy as np


def geometry_fingerprint(core):
    if core.T_base_camera is None:
        return None
    values=np.r_[np.asarray(core.T_base_camera).ravel(),np.asarray(core.T_ir_to_rgb).ravel(),
        core.ir_fx,core.ir_fy,core.ir_cx,core.ir_cy,core.depth_scale,core.ball_diameter_mm]
    return hashlib.sha256(np.round(values,8).astype('<f8').tobytes()).hexdigest()


def load_calibration(core, path=None):
    path=Path(path) if path is not None else Path(__file__).with_name('ball_observation_calibration.json')
    if not path.is_file():
        return None
    value=json.loads(path.read_text())
    delta=np.asarray(value.get('translation_m'),float)
    shift=float(value.get('relative_time_s',0.))
    if (value.get('schema_version')!=1 or delta.shape!=(3,) or not np.isfinite(delta).all()
            or np.linalg.norm(delta)>.1 or not np.isfinite(shift) or abs(shift)>.020):
        raise ValueError('Invalid ball observation calibration')
    if value.get('geometry_fingerprint')!=geometry_fingerprint(core):
        return None
    if 'rotation' in value:
        rotation=np.asarray(value['rotation'],float)
        if (rotation.shape!=(3,3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
            raise ValueError('Invalid observation rotation')
    profile=value.get('silhouette_range')
    if profile is not None:
        coefficients=np.asarray([profile.get(k,np.nan) for k in ('a','b','sigma_m')],float)
        if (not np.isfinite(coefficients).all() or not 0<coefficients[0]<100
                or not -10<coefficients[1]<10 or not .005<=coefficients[2]<=.05
                or not isinstance(profile.get('ir_params'),dict) or not profile['ir_params']):
            raise ValueError('Invalid silhouette range calibration')
    return value


def silhouette_range(raw, calibration, ir_params, optical_cosine, width, height, camera_settings=None):
    """A camera/IR-profile-bound weak distance, not a stereo depth replacement."""
    profile=calibration.get('silhouette_range')
    if (not profile or not raw.get('geometry_quality_valid')
            or raw.get('range_quality_reason')!='sparse_stereo'
            or any(ir_params.get(k)!=v for k,v in profile['ir_params'].items())):
        return None
    if any((camera_settings or {}).get(k)!=v for k,v in profile.get('camera_settings',{}).items()):
        return None
    radius=raw.get('geometry_radius_px');u=raw.get('cx');v=raw.get('cy')
    if radius is None or u is None or v is None:return None
    if (not np.isfinite([radius,u,v,optical_cosine]).all() or not 10<radius<70
            or min(u,width-1-u,v,height-1-v)<radius+3 or optical_cosine<.5):
        return None
    distance=float(profile['a'])/(radius-float(profile['b']))/optical_cosine
    return (distance,float(profile['sigma_m'])) if .15<distance<1. else None


def depth_pair_quality(ir_time_ms, depth_time_ms):
    """No waiting or stream changes: an asynchronous pair supplies bearing only."""
    if ir_time_ms is None or depth_time_ms is None:
        return None,None
    delta=float(depth_time_ms)-float(ir_time_ms)
    return bool(np.isfinite(delta) and abs(delta)<=2.),delta
