#!/usr/bin/env python3
"""Offline IR tuning against raw_camera recordings, using the production core.

python3 offline_ir_tuner.py --recording /path/probe_recording_... --serve
python3 offline_ir_tuner.py --recording /path/probe_recording_... --grid grid.json
No camera/ROS initialization, actuator commands or live configuration writes.
"""
import argparse
import base64
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
from pathlib import Path
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pingpong_controller.camera_replay import IR_KEYS, TUNABLE_KEYS, make_detector, validate_overrides
from pingpong_controller.raw_camera_recording import atomic_json, json_value
from pingpong_controller.ball_sensor_comparison import collect_observations


def read_jsonl(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


class MocapTruth:
    def __init__(self, path):
        self.segments=[]
        if path.is_file():
            observations,_=collect_observations(read_jsonl(path))
            groups={}
            for p in observations['mocap']:
                groups.setdefault(p['segment'],[]).append(p)
            for group in groups.values():
                group=sorted(group,key=lambda p:p['stamp'])
                self.segments.append((np.array([p['stamp'] for p in group]),
                                      np.array([p['position'] for p in group])))

    def at(self, timestamp, ready=True, max_gap=.025):
        if not ready:return None
        matches=[]
        for times,positions in self.segments:
            if not len(times) or timestamp<times[0] or timestamp>times[-1]:continue
            j=int(np.searchsorted(times,timestamp))
            if j<len(times) and abs(times[j]-timestamp)<1e-8:
                matches.append(positions[j]);continue
            if not 0<j<len(times):continue
            dt=times[j]-times[j-1]
            if not 0<dt<=max_gap:continue
            if np.linalg.norm(positions[j]-positions[j-1])/dt>10:continue
            matches.append(positions[j-1]+(positions[j]-positions[j-1])*(timestamp-times[j-1])/dt)
        # Repeated timestamps in distinct clock epochs are ambiguous. Refuse
        # them instead of silently matching an old trajectory.
        return matches[0] if len(matches)==1 else None


class ReplaySession:
    def __init__(self, path, output=None):
        root=Path(path).expanduser().resolve()
        self.directory=root if (root/'metadata.json').is_file() and (root/'frames').is_dir() and root.name=='raw_camera' else root/'raw_camera'
        # Also accept a standalone raw session with a custom directory name.
        if not self.directory.is_dir() and (root/'metadata.json').is_file():self.directory=root
        if not (self.directory/'metadata.json').is_file():
            raise ValueError('No raw_camera/metadata.json. Annotated JPGs and scalar logs cannot replay IR/depth detection.')
        self.metadata=json.loads((self.directory/'metadata.json').read_text())
        if self.metadata.get('format')!='raw-ir-depth-npz-v1':
            raise ValueError('Unsupported raw camera recording format')
        summary_path=self.directory/'summary.json'
        self.recording_summary=json.loads(summary_path.read_text()) if summary_path.exists() else {'state':'unknown'}
        if self.recording_summary.get('state')=='recording':
            raise ValueError('Raw session is recording or was interrupted. Stop recording first; recover interrupted sessions explicitly before tuning.')
        self.frames=list(read_jsonl(self.directory/'frames.jsonl'))
        if not self.frames:raise ValueError('Raw recording has no frames')
        self.truth=MocapTruth(self.directory.parent/'ball_sensor_samples.jsonl')
        self.output=Path(output).resolve() if output else self.directory.parent/'offline_ir_tuning'
        self.lock=threading.Lock()
        self.detector=None
        self.index=-1
        self.key=None
        self.context_key=None
        self.last=None

    def defaults(self):
        values={k:self.metadata['detectorKwargs'][k] for k in TUNABLE_KEYS
                if k in self.metadata['detectorKwargs']}
        # Older raw sessions used the hard-coded 25th percentile. Expose that
        # exact default when replaying them with the newly tunable sampler.
        values.setdefault('depth_foreground_percentile',25.0)
        values.setdefault('synchronized_depth_enabled',False)
        values.setdefault('depth_assisted_detection_enabled',False)
        values.setdefault('observation_quality_enabled',True)
        values.setdefault('partial_surface_detection_enabled',True)
        values.setdefault('observation_calibration_enabled',True)
        return values

    def reset(self):
        self.detector=None;self.index=-1;self.context_key=None;self.last=None

    def load_arrays(self, row):
        path=(self.directory/row['path']).resolve()
        if self.directory not in path.parents:raise ValueError('frame path escapes recording')
        with np.load(path,allow_pickle=False) as data:
            arrays={k:data[k] for k in data.files}
        ir=arrays['ir']
        if ir.dtype!=np.uint8 or ir.ndim!=2:raise ValueError('Expected raw uint8 IR image')
        depth=arrays.get('depth')
        if depth is not None and (depth.dtype!=np.uint16 or depth.shape!=ir.shape):
            raise ValueError('Expected raw IR-aligned uint16 depth image')
        rgb=arrays.get('rgb')
        if rgb is None:rgb=cv2.cvtColor(ir,cv2.COLOR_GRAY2BGR)
        return rgb,ir,depth

    def step(self, index, overrides):
        row=self.frames[index]
        options=row.get('runtimeOptions',{})
        sync=row.get('clockSync',{})
        effective={k:v for k,v in options.items() if k in self.metadata['detectorKwargs'] or k in {'yolo_enabled','use_depth_sensor'}}
        context=(sync.get('generation'),sync.get('epoch'),json.dumps(effective,sort_keys=True))
        backwards=index>0 and row['timestampS']<=self.frames[index-1]['timestampS']
        reset=self.detector is None or context!=self.context_key or backwards
        if reset:
            self.detector=make_detector(self.metadata,overrides,options,self.directory)
            self.detector.offline_debug_images={}
            self.context_key=context
            from pingpong_controller.ball_observation_model import BallObservationProcessor
            self.observation_processor=BallObservationProcessor(
                self.metadata.get('simOffsetM', (0., 0., 0.)),
                self.metadata.get('observationConfig', {}).get('predictionFloorM', 1.24))
        detector=self.detector
        state=row.get('racketState')
        detector.latest_racket_state=(float(state[0]),np.asarray(state[1]),np.asarray(state[2])) if state else None
        rgb,ir,depth=self.load_arrays(row)
        detector.offline_debug_images.clear()
        if options.get('detect',True):
            observation=detector.get_observation_offline(rgb,ir,
                depth if options.get('use_depth_sensor',self.metadata.get('useDepthSensor',True)) else None,
                float(row['timestampS']), frame_metadata=row)
        else:
            observation={'detected':False,'raw_measurement':{'observation_reason':'recorded_detection_disabled'}}
        self.observation_processor.process(observation, float(row['timestampS']),
                                           racket_state=detector.latest_racket_state,
                                           racket_context=row.get('racketContext'))
        self.index=index
        self.last=(row,observation,rgb,ir,depth,reset)
        return self.last

    def frame(self, index, overrides):
        validate_overrides(overrides)
        index=max(0,min(len(self.frames)-1,int(index)))
        key=json.dumps(overrides,sort_keys=True)
        # Every alternative starts from the recorded prefix, including all
        # missing detections. No repeated update of one frozen frame.
        if key!=self.key or index<self.index:
            self.reset();self.key=key
        while self.index<index:self.step(self.index+1,overrides)
        return self.last

    def truth_position(self,row):
        sync=row.get('clockSync',{})
        return self.truth.at(float(row['timestampS']),bool(sync.get('ready',False)))

    def project_truth(self, point):
        """Evaluation/overlay only: never feed a mocap point into detection."""
        if point is None or self.metadata.get('calibration') is None:return None
        import pyrealsense2 as rs
        offset=np.asarray(self.metadata.get('simOffsetM',[0,0,0]))
        base=point-offset
        T=np.asarray(self.metadata['calibration']['T_base_camera'])
        color=np.linalg.solve(T,np.r_[base,1.])[:3]
        ir=np.linalg.solve(self.detector.T_ir_to_rgb,np.r_[color,1.])[:3]
        if ir[2]<=0:return None
        uv=np.array(rs.rs2_project_point_to_pixel(self.detector.ir_intrinsics,ir.tolist()))
        if not np.isfinite(uv).all():return None
        return uv

    def preview(self,index,overrides):
        row,obs,rgb,ir,depth,reset=self.frame(index,overrides)
        raw=obs.get('raw_measurement') or {}
        annotated=cv2.cvtColor(ir,cv2.COLOR_GRAY2BGR)
        cx,cy=raw.get('cx'),raw.get('cy')
        if cx is not None and cy is not None and np.isfinite([cx,cy]).all():
            cv2.circle(annotated,(int(cx),int(cy)),max(2,int(raw.get('radius') or 3)),(0,200,0),2)
        truth=self.truth_position(row);uv=self.project_truth(truth)
        if uv is not None:
            cv2.drawMarker(annotated,tuple(np.rint(uv).astype(int)),(0,0,255),cv2.MARKER_CROSS,16,2)
        debug=self.detector.offline_debug_images
        binary=debug.get('binary',np.zeros_like(ir))
        def encode(image):
            ok,data=cv2.imencode('.png',image)
            if not ok:raise ValueError('PNG encoding failed')
            return 'data:image/png;base64,'+base64.b64encode(data).decode()
        return json_value(dict(index=self.index,count=len(self.frames),timestampS=row['timestampS'],
            sourceFrameNumber=row.get('sourceFrameNumber'),sequence=row['sequence'],
            detected=bool(obs.get('detected')),measurement=raw,truthPositionM=truth,
            truthPixel=uv,contextReset=reset,binaryRoi=debug.get('roi'),
            images={'ir':encode(annotated),'binary':encode(binary)},
            replayNote='Changing parameters/backward stepping replays the recorded prefix. Runtime config or clock changes reset the detector. Red cross=mocap projection, green=camera candidate.'))

    def evaluate(self,overrides,limit=0,tag='evaluation'):
        validate_overrides(overrides)
        self.reset();self.key=json.dumps(overrides,sort_keys=True)
        rows=[];errors=[];eligible=correct=wrong=miss=ir_present=detected=0
        longest_gap=current_gap=0.;previous_time=None
        count=min(len(self.frames),limit) if limit else len(self.frames)
        for i in range(count):
            row,obs,rgb,ir,depth,reset=self.step(i,overrides)
            raw=obs.get('raw_measurement') or {};valid=bool(obs.get('detected'))
            detected+=int(valid)
            ir_present+=int(raw.get('cx') is not None and raw.get('cy') is not None)
            point=np.asarray(obs.get('position',[np.nan]*3),float)/1000 if valid else np.full(3,np.nan)
            if self.metadata['detectorKwargs'].get('output_frame')=='base_link':
                point+=np.asarray(self.metadata.get('simOffsetM',[0,0,0]))
            truth=self.truth_position(row);uv=self.project_truth(truth)
            truth_ok=uv is not None and 0<=uv[0]<ir.shape[1] and 0<=uv[1]<ir.shape[0]
            error=float(np.linalg.norm(point-truth)*1000) if truth_ok and valid else None
            good=error is not None and np.isfinite(error) and error<=50
            if truth_ok:
                eligible+=1;correct+=int(good);wrong+=int(valid and not good);miss+=int(not valid)
                if error is not None and np.isfinite(error):errors.append(error)
            dt=float(row['timestampS'])-previous_time if previous_time is not None else 0.
            if truth_ok and not good and 0<dt<=.035 and not reset:current_gap+=dt
            else:current_gap=0.
            longest_gap=max(longest_gap,current_gap);previous_time=float(row['timestampS'])
            rows.append(dict(index=i,sequence=row['sequence'],timestamp_s=row['timestampS'],
                source_frame=row.get('sourceFrameNumber'),detected=valid,gt_eligible=truth_ok,
                correct_50mm=good,error_mm=error,x_m=point[0],y_m=point[1],z_m=point[2],
                gt_x_m=truth[0] if truth is not None else None,
                gt_y_m=truth[1] if truth is not None else None,
                gt_z_m=truth[2] if truth is not None else None,
                context_reset=reset,
                reason=raw.get('observation_reason',raw.get('depth_rejection_reason')),
                ir_contours=raw.get('ir_contours_examined'),depth_mm=raw.get('depth_mm')))
        report=dict(frames=count,detected=detected,detectedRate=detected/count,irCandidateRate=ir_present/count,
            gtEligible=eligible,correct50mm=correct,wrong50mm=wrong,missed=miss,
            correctCoverage=correct/eligible if eligible else None,
            wrongRate=wrong/eligible if eligible else None,
            rmsMm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            p95Mm=float(np.percentile(errors,95)) if errors else None,
            longestObservedBadIntervalS=longest_gap,
            score=(correct-wrong)/eligible if eligible else None,
            parameters=overrides,recordingSummary=self.recording_summary,
            scope='Recorded frames only. Recording losses are separate from detector misses. No high-ball prediction; mocap is used only for evaluation. 50 mm is a fixed evaluation tolerance, not a ground-truth detection label.',
            runtimeChanges='Detector resets at configuration or camera clock changes.',
            truthAvailable=bool(self.truth.segments))
        self.output.mkdir(parents=True,exist_ok=True)
        atomic_json(self.output/(tag+'.json'),report)
        with (self.output/(tag+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return report

    def export(self,overrides):
        validate_overrides(overrides)
        values=dict(self.metadata['detectorKwargs']);values.update(overrides)
        if values['min_radius'] >= values['max_radius']:
            raise ValueError('min_radius must be less than max_radius')
        # This file is a candidate. The tool never overwrites live ir_params.json.
        ir={key:values[key] for key in IR_KEYS}
        for key in ('laser_power','exposure','gain'):ir[key]=values.get(key)
        return dict(ir_params=ir,detector_overrides=overrides,
                    sourceRecording=str(self.directory),automaticLiveApply=False)


HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>离线 IR 调参</title>
<style>body{font:16px sans-serif;margin:24px;background:#f5f6f8;color:#17212a}button,input,select{font:inherit;padding:6px;margin:4px}button{cursor:pointer}#controls{display:flex;flex-wrap:wrap;gap:8px}label{display:flex;flex-direction:column;font-size:13px}input[type=number]{width:110px}img{width:48%;object-fit:contain;background:#111}pre{white-space:pre-wrap;background:white;padding:12px}#status{padding:10px;background:#e9eef5}#seek{width:65%}</style>
<h2>离线球检测调参</h2><p>原始 IR / 深度逐帧回放 · 使用正式检测核心 · 红十字：动捕投影；绿圈：相机候选</p>
<div id="status">载入中…</div><p><button id="prev">上一帧</button><input id="seek" type="range" min="0" value="0"><button id="next">下一帧</button><input id="idx" type="number" min="0" value="0"></p>
<div id="controls"></div><p><button id="apply">重算当前帧</button><button id="reset">恢复录制参数</button><button id="evaluate">评估整段</button><button id="export">下载候选配置</button></p>
<p>IR 候选与动捕位置　　　　　　　　　　　　　　　二值图（ROI 检测时显示对应局部）</p>
<div><img id="ir"><img id="binary"></div><pre id="info"></pre>
<script>
let defaults={},busy=false,count=0;const $=id=>document.getElementById(id);
function params(){let p={};for(const k in defaults){const e=$('p_'+k);p[k]=e.type==='checkbox'?e.checked:Number(e.value);}return p;}
async function request(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});let d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d;}
async function run(fn){if(busy)return;busy=true;$('status').textContent='正在离线回放；参数变化或向前跳转时需要处理历史帧…';try{await fn();}catch(e){$('status').textContent=e.message;}finally{busy=false;}}
async function show(){await run(async()=>{const d=await request('/frame',{index:Number($('idx').value),parameters:params()});$('idx').value=$('seek').value=d.index;$('ir').src=d.images.ir;$('binary').src=d.images.binary;delete d.images;$('info').textContent=JSON.stringify(d,null,2);$('status').textContent=`帧 ${d.index+1}/${count} · ${d.detected?'检测有效':'检测失败'} · 已处理历史帧，重复查看不更新滤波器`;});}
$('prev').onclick=()=>{$('idx').value=Math.max(0,Number($('idx').value)-1);show();};$('next').onclick=()=>{$('idx').value=Math.min(count-1,Number($('idx').value)+1);show();};$('seek').onchange=()=>{$('idx').value=$('seek').value;show();};$('idx').onchange=show;$('apply').onclick=show;
$('reset').onclick=()=>{for(const[k,v]of Object.entries(defaults)){let e=$('p_'+k);if(e.type==='checkbox')e.checked=v;else e.value=v;}show();};
$('evaluate').onclick=()=>run(async()=>{let d=await request('/evaluate',{parameters:params()});$('info').textContent=JSON.stringify(d,null,2);$('status').textContent=d.gtEligible?`整段评估完成：真值有效帧 ${d.gtEligible}；正确位置覆盖率 ${(d.correctCoverage*100).toFixed(1)}%`:'整段评估完成；没有可用动捕配对，不能依据检出率判断准确性';});
$('export').onclick=()=>run(async()=>{let d=await request('/export',{parameters:params()});for(const[name,value]of Object.entries({'ir_params_candidate.json':d.ir_params,'detector_overrides.json':d.detector_overrides})){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.download=name;a.click();URL.revokeObjectURL(a.href);}$('status').textContent='已下载候选参数，尚未应用到实时相机';});
fetch('/session').then(r=>r.json()).then(d=>{defaults=d.parameters;count=d.frames;$('seek').max=$('idx').max=count-1;for(const[k,v]of Object.entries(defaults)){let l=document.createElement('label');l.textContent=k;let e=document.createElement('input');e.id='p_'+k;e.type=typeof v==='boolean'?'checkbox':'number';e.step='any';if(e.type==='checkbox')e.checked=v;else e.value=v;l.append(e);$('controls').append(l);}show();}).catch(e=>$('status').textContent=e.message);
</script></html>'''


def serve(session,port):
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,status=200,content_type='application/json'):
            body=data.encode() if isinstance(data,str) else json.dumps(json_value(data),ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type+'; charset=utf-8')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/':self.send(HTML,content_type='text/html')
            elif self.path=='/session':self.send(dict(frames=len(session.frames),parameters=session.defaults(),summary=session.recording_summary))
            else:self.send({'error':'not found'},404)
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=65536:raise ValueError('invalid request size')
                body=json.loads(self.rfile.read(size));params=body.get('parameters',{})
                with session.lock:
                    if self.path=='/frame':result=session.preview(body.get('index',0),params)
                    elif self.path=='/evaluate':result=session.evaluate(params)
                    elif self.path=='/export':result=session.export(params)
                    else:raise ValueError('unknown operation')
                self.send(result)
            except Exception as exc:self.send({'error':str(exc)},400)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'Offline tuner: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',required=True)
    parser.add_argument('--serve',action='store_true')
    parser.add_argument('--port',type=int,default=8092)
    parser.add_argument('--params',type=Path,help='Flat offline software parameter overrides JSON')
    parser.add_argument('--grid',type=Path,help='JSON object mapping parameter names to value lists')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--limit-frames',type=int,default=0)
    args=parser.parse_args()
    if args.limit_frames<0:parser.error('--limit-frames must be non-negative')
    session=ReplaySession(args.recording,args.output)
    if args.serve:
        serve(session,args.port);return
    params=json.loads(args.params.read_text()) if args.params else {}
    if args.grid:
        grid=json.loads(args.grid.read_text())
        if not grid or any(not isinstance(v,list) or not v for v in grid.values()):
            parser.error('grid values must be nonempty lists')
        if np.prod([len(v) for v in grid.values()])>512:parser.error('grid limited to 512 combinations')
        reports=[]
        for index,values in enumerate(itertools.product(*grid.values())):
            candidate={**params,**dict(zip(grid,values))}
            report=session.evaluate(candidate,args.limit_frames,f'candidate_{index:03d}')
            reports.append(report)
            print(json.dumps(json_value(dict(index=index,score=report['score'],rmsMm=report['rmsMm'],parameters=candidate))),flush=True)
        reports.sort(key=lambda r:r['score'] if r['score'] is not None else -float('inf'),reverse=True)
        atomic_json(session.output/'grid_results.json',dict(candidates=reports,
            best=reports[0]['parameters'] if reports and reports[0]['gtEligible'] else None,
            note='Development-set ranking only; validate on separate recordings before applying. No truth => no automatic recommendation.'))
    else:
        print(json.dumps(json_value(session.evaluate(params,args.limit_frames)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
