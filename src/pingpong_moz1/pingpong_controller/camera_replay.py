"""Capture detector configuration and rebuild the same core for offline replay."""
import inspect
from pathlib import Path

import numpy as np

from pingpong_controller.raw_camera_recording import json_value


IR_KEYS = ('ir_threshold', 'use_adaptive', 'dilate_iterations', 'close_kernel_size',
           'denoise_kernel_size', 'min_radius', 'max_radius', 'circularity_min',
           'brightness_percentile')
TUNABLE_KEYS = IR_KEYS + ('depth_neighborhood_px', 'depth_foreground_percentile',
    'synchronized_depth_enabled', 'depth_assisted_detection_enabled', 'observation_quality_enabled',
    'partial_surface_detection_enabled', 'local_surface_detection_enabled', 'observation_calibration_enabled', 'candidate_filter_enabled',
    'candidate_min_brightness', 'candidate_max_depth_disagreement_mm',
    'candidate_motion_scale_px')


def snapshot_detector(core, cfg):
    """Called at camera startup, before capture/detection threads start."""
    signature = inspect.signature(type(core).__init__)
    kwargs = {}
    excluded = {'self','calibration_data','offline_camera_params','load_yolo_model'}
    for name, parameter in signature.parameters.items():
        if name in excluded:
            continue
        if name == 'ir_threshold':
            value = core.ir_params['threshold']
        elif name in core.ir_params:
            value = core.ir_params[name]
        elif name == 'width':
            value = core.image_width
        elif name == 'height':
            value = core.image_height
        elif name == 'exposure':
            value = core.ir_exposure
        elif name == 'gain':
            value = core.ir_gain
        elif name == 'laser_power':
            value = getattr(core, 'ir_laser_power', getattr(cfg, 'ir_laser_power', parameter.default))
        elif name in ('impact_min_downward_velocity_m_s','impact_min_upward_velocity_m_s',
                       'impact_min_velocity_jump_m_s'):
            value = getattr(core, name.replace('_m_s','_mm_s'))/1000.
        else:
            value = getattr(core, name, parameter.default)
        if value is inspect.Parameter.empty:
            raise ValueError(f'cannot snapshot detector option {name}')
        kwargs[name] = json_value(value)
    calibration = core.calibration_data
    calib_json = None
    if calibration is not None:
        calib_json = dict(convention=calibration.convention,
            intrinsics={key:json_value(getattr(calibration,key)) for key in (
                'camera_matrix','dist_coeffs','original_camera_matrix','original_dist_coeffs',
                'image_width','image_height','image_model','frame_id')},
            T_base_camera=json_value(calibration.T_base_camera))
    # Preserve constants as well as constructor parameters; the current source
    # may otherwise change between capture and a later tuning session.
    constants = {key:json_value(getattr(core,key)) for key in dir(type(core))
                 if key.isupper() and isinstance(getattr(core,key), (int,float,bool))}
    return dict(schemaVersion=1, format='raw-ir-depth-npz-v1',
        camera=core.export_camera_params(), detectorKwargs=kwargs, constants=constants,
        calibration=calib_json, yoloEnabled=bool(core.yolo_enabled),
        impactRacketGateRequired=bool(getattr(core,'impact_racket_gate_required',False)),
        useDepthSensor=bool(getattr(cfg,'use_depth_sensor',True)),
        simOffsetM=[float(getattr(cfg,'sim_cartesian_x_offset_m',0.)),
                    float(getattr(cfg,'sim_cartesian_y_offset_m',0.)),
                    float(getattr(cfg,'sim_base_height_m',0.))],
        observationConfig={'predictionFloorM':float(getattr(cfg,'high_ball_prediction_stop_z_m',1.24))},
        observationCalibration=json_value(getattr(core,'observation_calibration',None)),
        captureScope='Frames consumed by the production detector, including rejected/missing detections; capture frames replaced before detection are not recorded.',
        rgbPolicy='RGB retained only when YOLO is enabled; IR-only replay uses an IR-derived display canvas.',
        recordingPolicy='Bounded best-effort copy. Never block camera reception or control. Recording drops remain visible in manifest sequence/frame numbers and summary.')


def validate_overrides(overrides):
    unknown = set(overrides)-set(TUNABLE_KEYS)
    if unknown:
        raise ValueError('Not offline-tunable: '+', '.join(sorted(unknown)))
    boolean_keys = {'use_adaptive','candidate_filter_enabled','synchronized_depth_enabled',
                    'depth_assisted_detection_enabled', 'observation_quality_enabled',
                    'partial_surface_detection_enabled', 'local_surface_detection_enabled', 'observation_calibration_enabled'}
    for key, value in overrides.items():
        if not isinstance(value,(int,float,bool)) or not np.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
        if key in boolean_keys and value not in (True,False):
            raise ValueError(f'{key} must be boolean')
        if key not in boolean_keys and value < 0:
            raise ValueError(f'{key} must be non-negative')
        if key in {'ir_threshold','dilate_iterations','close_kernel_size','denoise_kernel_size',
                   'depth_neighborhood_px'} and int(value)!=value:
            raise ValueError(f'{key} must be an integer')
    if not 0 <= overrides.get('ir_threshold',170) <= 255:
        raise ValueError('ir_threshold must be in [0,255]')
    for key in ('close_kernel_size','denoise_kernel_size'):
        if key in overrides and overrides[key] not in (0,1) and (int(overrides[key]) != overrides[key] or int(overrides[key])%2 == 0):
            raise ValueError(f'{key} must be 0, 1 or a positive odd integer')
    if 'circularity_min' in overrides and overrides['circularity_min'] > 1:
        raise ValueError('circularity_min must be <= 1')
    if overrides.get('brightness_percentile', 30) > 100:
        raise ValueError('brightness_percentile must be in [0,100]')
    if overrides.get('depth_foreground_percentile', 25) > 50:
        raise ValueError('depth_foreground_percentile must be in [0,50]')


def make_detector(metadata, overrides=None, runtime_options=None, recording_dir=None):
    from pingpong_controller.hybrid_ball_detector import HybridBallDetector
    from pingpong_controller.calibration_loader import CalibrationData
    overrides = overrides or {}
    validate_overrides(overrides)
    kwargs = dict(metadata['detectorKwargs'])
    options = runtime_options or {}
    for key,value in options.items():
        if key in kwargs:
            kwargs[key] = value
    kwargs.update(overrides)
    if kwargs['min_radius'] >= kwargs['max_radius']:
        raise ValueError('min_radius must be less than max_radius')
    yolo_enabled=bool(options.get('yolo_enabled',metadata.get('yoloEnabled',False)))
    if recording_dir is not None:
        for key in ('depth_correction_model_path','yolo_model_path'):
            asset=metadata.get('assets',{}).get(key)
            if asset:
                kwargs[key]=str(Path(recording_dir)/asset['path'])
    kwargs.update(offline_camera_params=metadata['camera'],
        calibration_data=CalibrationData(metadata['calibration']) if metadata.get('calibration') else None,
        print_every=10**12, load_yolo_model=yolo_enabled)
    detector = HybridBallDetector(**kwargs)
    if 'observationCalibration' in metadata and kwargs.get('observation_calibration_enabled',True):
        from pingpong_controller.ball_observation_calibration import geometry_fingerprint
        recorded_calibration=metadata['observationCalibration']
        detector.observation_calibration=(recorded_calibration if recorded_calibration
            and recorded_calibration.get('geometry_fingerprint')==geometry_fingerprint(detector) else None)
    for key, value in metadata.get('constants',{}).items():
        if key.isupper() and hasattr(detector,key):
            setattr(detector,key,value)
    detector.configure_yolo(yolo_enabled)
    detector.impact_racket_gate_required=bool(metadata.get('impactRacketGateRequired',False))
    return detector
