import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pingpong_controller.camera_replay import make_detector, snapshot_detector, validate_overrides
from pingpong_controller.tools.offline_ir_tuner import ReplaySession, MocapTruth


def metadata():
    matrix=[[250,0,80],[0,250,60],[0,0,1]]
    intr=dict(width=160,height=120,ppx=80,ppy=60,fx=250,fy=250,model=0,coeffs=[0]*5)
    camera=dict(width=160,height=120,fps=90,depth_scale=.001,
        ir_intrinsics=intr,color_intrinsics=intr,
        depth_to_color_extrinsics=dict(rotation=np.eye(3).reshape(-1).tolist(),translation=[0,0,0]))
    calib=dict(intrinsics=dict(camera_matrix=matrix,original_camera_matrix=matrix,
        dist_coeffs=[0]*5,original_dist_coeffs=[0]*5,image_width=160,image_height=120),
        T_base_camera=np.eye(4).tolist())
    return dict(format='raw-ir-depth-npz-v1',camera=camera,calibration=calib,
        detectorKwargs=dict(yolo_model_path='__not_needed__',width=160,height=120,fps=90,
            ir_threshold=100,use_adaptive=False,dilate_iterations=0,close_kernel_size=0,
            denoise_kernel_size=0,min_radius=5,max_radius=20,circularity_min=.6,
            brightness_percentile=25,depth_neighborhood_px=2,depth_mode='sensor',
            pixel_kalman_enabled=False,candidate_filter_enabled=False,
            candidate_min_brightness=23000,candidate_max_depth_disagreement_mm=180,
            candidate_motion_scale_px=130,output_frame='base_link',depth_correction_model_path='__none__'),
        yoloEnabled=False,simOffsetM=[0,0,0])


def fixture_recording(tmp_path):
    path=tmp_path/'raw_camera';(path/'frames').mkdir(parents=True)
    (path/'metadata.json').write_text(json.dumps(metadata()))
    (path/'summary.json').write_text('{"state":"complete"}')
    rows=[]
    for i in range(8):
        ir=np.zeros((120,160),np.uint8)
        if i!=3:cv2.circle(ir,(80+i,60),10,220,-1)
        depth=np.full((120,160),500,np.uint16)
        filename=f'frames/{i:09d}.npz'
        np.savez(path/filename,ir=ir,depth=depth)
        rows.append(dict(sequence=i+1,path=filename,timestampS=10+i/90.,
            sourceFrameNumber=i,clockSync=dict(ready=True,generation=0,epoch=0),
            runtimeOptions=dict(yolo_enabled=False,use_depth_sensor=True)))
    (path/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path


def test_core_without_camera_or_yolo_and_snapshot_roundtrip():
    core=make_detector(metadata())
    assert core.pipeline is None and core.yolo_model is None and not core.yolo_enabled
    snap=snapshot_detector(core,SimpleNamespace(use_depth_sensor=True))
    restored=make_detector(snap)
    assert restored.ir_params==core.ir_params
    assert restored.pixel_kalman_enabled==core.pixel_kalman_enabled
    assert restored.depth_mode==core.depth_mode
    assert snap['detectorKwargs']['width']==160


def test_opt_in_depth_assistance_recovers_faint_ball_in_shared_core():
    ir=np.zeros((120,160),np.uint8)
    depth=np.full(ir.shape,2000,np.uint16)
    cv2.circle(ir,(80,60),10,120,-1)
    cv2.circle(depth,(80,60),10,500,-1)
    cfg=metadata();cfg['detectorKwargs']['ir_threshold']=150
    original=make_detector(cfg)
    improved=make_detector(cfg,{'depth_assisted_detection_enabled':True})
    rgb=cv2.cvtColor(ir,cv2.COLOR_GRAY2BGR)
    assert not original.get_observation_offline(rgb,ir,depth,1.)['detected']
    obs=improved.get_observation_offline(rgb,ir,depth,1.)
    assert obs['detected']
    assert obs['raw_measurement']['detection_method']=='depth_supported_ir'
    np.testing.assert_allclose(obs['position'],[0,0,500],atol=2)
    snap=snapshot_detector(improved,SimpleNamespace(use_depth_sensor=True))
    assert make_detector(snap).depth_assisted_detection_enabled


def test_bad_depth_on_bright_primary_candidate_retries_instead_of_losing_ball():
    ir=np.zeros((120,160),np.uint8);depth=np.full(ir.shape,2000,np.uint16)
    cv2.circle(ir,(35,60),10,240,-1)  # Brighter but not at ball depth.
    cv2.circle(ir,(100,60),10,120,-1)
    cv2.circle(depth,(100,60),10,500,-1)
    cfg=metadata();cfg['detectorKwargs']['ir_threshold']=150
    core=make_detector(cfg,{'depth_assisted_detection_enabled':True})
    obs=core.get_observation_offline(cv2.cvtColor(ir,cv2.COLOR_GRAY2BGR),ir,depth,1.)
    assert obs['detected']
    assert obs['raw_measurement']['detection_method']=='depth_supported_ir'
    np.testing.assert_allclose(obs['position'],[40,0,500],atol=2)


def test_replay_matches_direct_production_core_including_failed_frames(tmp_path):
    path=fixture_recording(tmp_path);session=ReplaySession(path)
    direct=make_detector(metadata())
    for i,row in enumerate(session.frames):
        rgb,ir,depth=session.load_arrays(row)
        expected=direct.get_observation_offline(rgb,ir,depth,row['timestampS'])
        actual=session.frame(i,{})[1]
        assert actual['detected']==expected['detected']
        np.testing.assert_allclose(actual.get('position',[0]*3),expected.get('position',[0]*3))
    assert not session.frame(3,{})[1]['detected']


def test_freeze_and_parameter_change_replay_prefix_without_double_filter_updates(tmp_path):
    session=ReplaySession(fixture_recording(tmp_path))
    first=session.frame(5,{})[1]
    count=session.detector.frame_count
    second=session.frame(5,{})[1]
    assert first is second and session.detector.frame_count==count
    high=session.frame(5,{'ir_threshold':250})[1]
    assert not high['detected']
    original=session.frame(5,{})[1]
    np.testing.assert_allclose(original['position'],first['position'])
    preview=session.preview(5,{})
    assert preview['images']['binary'].startswith('data:image/png;base64,')


def test_truth_is_evaluation_only_and_no_truth_does_not_claim_accuracy(tmp_path):
    session=ReplaySession(fixture_recording(tmp_path))
    result=session.evaluate({})
    assert result['gtEligible']==0 and result['score'] is None
    original=session.frame(4,{})[1]['position']
    session.truth.segments=[(np.array([10.,10.1]),np.array([[20,30,40],[20,30,40]]))]
    np.testing.assert_array_equal(session.frame(4,{})[1]['position'],original)
    assert (session.output/'evaluation.csv').exists()


def test_mocap_scoring_counts_missed_detections_in_denominator(tmp_path):
    path=fixture_recording(tmp_path)
    mocap=[]
    for t in np.arange(9.99,10.11,.005):
        mocap.append(dict(receiverTimeS=t,mocap=dict(valid=True,frameAligned=True,
            clockSync=dict(ready=True,epoch=0),measurementTimestampS=t,
            position=[.18*(t-10),0,.5])))
    (tmp_path/'ball_sensor_samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mocap))
    session=ReplaySession(path)
    baseline=session.evaluate({})
    missed=session.evaluate({'ir_threshold':250},tag='missed')
    assert baseline['gtEligible']==missed['gtEligible']==8
    assert baseline['correct50mm']==7
    assert missed['missed']==8 and missed['score']==0
    assert baseline['rmsMm']<2
    with (session.output/'evaluation.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert float(rows[0]['gt_z_m'])==.5 and rows[3]['detected']=='False'


def test_mocap_interpolation_never_bridges_missing_or_ambiguous_epochs(tmp_path):
    truth=MocapTruth(tmp_path/'absent')
    truth.segments=[(np.array([1.,1.01]),np.array([[0,0,1.],[0,0,1.01]]))]
    np.testing.assert_allclose(truth.at(1.005),[0,0,1.005])
    assert truth.at(1.02) is None and truth.at(1.005,ready=False) is None
    truth.segments*=2
    assert truth.at(1.005) is None


def test_hardware_parameters_are_not_faked_offline_and_export_does_not_apply(tmp_path):
    for params in [dict(exposure=1000),dict(gain=16),dict(ir_threshold=300),
                   dict(close_kernel_size=4),dict(dilate_iterations=2.5),
                   dict(brightness_percentile=101),
                   dict(depth_foreground_percentile=51),
                   dict(depth_foreground_percentile=-1)]:
        with pytest.raises(ValueError):validate_overrides(params)
    session=ReplaySession(fixture_recording(tmp_path))
    result=session.export({'ir_threshold':180})
    assert result['ir_params']['ir_threshold']==180
    assert not result['automaticLiveApply']
    assert not session.output.exists()
    with pytest.raises(ValueError,match='min_radius'):
        session.export({'min_radius':30,'max_radius':20})


def test_reject_live_or_interrupted_recording(tmp_path):
    path=fixture_recording(tmp_path)
    (path/'summary.json').write_text('{"state":"recording"}')
    with pytest.raises(ValueError,match='interrupted'):ReplaySession(path)


def test_foreground_percentile_override_survives_recording_snapshot():
    core=make_detector(metadata(),{'depth_foreground_percentile':10})
    snap=snapshot_detector(core,SimpleNamespace(use_depth_sensor=True))
    assert snap['detectorKwargs']['depth_foreground_percentile']==10
    assert make_detector(snap).depth_foreground_percentile==10
    assert make_detector(metadata()).depth_foreground_percentile==25


def test_same_frame_override_wins_over_recorded_runtime_median():
    core=make_detector(metadata(),{'synchronized_depth_enabled':True},
                       {'synchronized_depth_enabled':False})
    core.depth_history.extend([600.,600.,600.])
    assert core._depth_for_reconstruction(550.,commit=False)==550.
    validate_overrides({'synchronized_depth_enabled':True})
    with pytest.raises(ValueError):
        validate_overrides({'synchronized_depth_enabled':2})
