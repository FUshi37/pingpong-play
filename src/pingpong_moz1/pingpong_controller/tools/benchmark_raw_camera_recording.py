#!/usr/bin/env python3
"""No-hardware 200 Hz control/receive + 90 Hz raw recording contention test.

Uses the actual recorder, full-sized arrays and actual disk writes. Timers and
pipe packets stand in for the robot/camera, so this is not hardware rate proof.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pingpong_controller.raw_camera_recording import RawCameraRecorder, atomic_json


def ticks(rate,duration):
    start=time.perf_counter()
    for index in range(int(rate*duration)):
        remaining=start+index/rate-time.perf_counter()
        if remaining>0:time.sleep(remaining)
        yield index,time.perf_counter()


def summarize(times):
    times=np.asarray(times)
    dt=np.diff(times)*1000
    return dict(count=len(times),hz=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0,
        intervalP99Ms=float(np.percentile(dt,99)) if len(dt) else None,
        intervalMaxMs=float(dt.max()) if len(dt) else None)


def unpack(packet):return packet


def camera_sim(mode,path,seconds,start,ready,results):
    arrays={'ir':np.full((480,848),150,np.uint8),'depth':np.full((480,848),500,np.uint16),
            'rgb':np.zeros((480,848,3),np.uint8)}
    recorder=None
    if mode!='off':
        recorder=RawCameraRecorder(slot_bytes=sum(x.nbytes for x in arrays.values()),slots=4,
            unpack=unpack,write_delay_s=.100 if mode=='slow_disk' else 0.,max_duration_s=seconds+5.)
        recorder.set_session(path,{'benchmark':True})
        deadline=time.monotonic()+10
        while not recorder.accepting and time.monotonic()<deadline:time.sleep(.005)
        if not recorder.accepting:raise RuntimeError('recorder did not start')
    ready.put('camera');start.wait()
    times=[];costs=[]
    for i,t in ticks(90,seconds):
        times.append(t)
        begin=time.perf_counter_ns()
        if recorder is not None:
            recorder.offer((arrays,dict(timestampS=t,sourceFrameNumber=i)))
        costs.append((time.perf_counter_ns()-begin)/1000)
    stats=summarize(times)
    stats.update(offerP99Us=float(np.percentile(costs,99)),offerMaxUs=float(max(costs)))
    if recorder:
        recorder.close(timeout_s=3.)
        stats['raw']=json.loads((Path(path)/'summary.json').read_text())
    results.put(stats)


def feedback_sender(connection,seconds,start,ready):
    ready.put('feedback');start.wait()
    for i,t in ticks(200,seconds):connection.send((i,t))
    connection.send(None);connection.close()


def trial(mode,path,seconds):
    ctx=mp.get_context('spawn');start=ctx.Event();ready=ctx.Queue();results=ctx.Queue()
    rx,tx=ctx.Pipe(duplex=False)
    camera=ctx.Process(target=camera_sim,args=(mode,str(path),seconds,start,ready,results))
    sender=ctx.Process(target=feedback_sender,args=(tx,seconds,start,ready))
    camera.start();sender.start();tx.close()
    received=[];latencies=[]
    def receive():
        while True:
            packet=rx.recv()
            if packet is None:break
            now=time.perf_counter();received.append(now);latencies.append((now-packet[1])*1000)
    thread=threading.Thread(target=receive);thread.start()
    ready.get(timeout=20);ready.get(timeout=20)
    start.set();control=[t for _,t in ticks(200,seconds)]
    thread.join(timeout=seconds+5)
    sender.join(timeout=5);camera.join(timeout=10)
    if camera.is_alive() or sender.is_alive() or thread.is_alive():
        camera.terminate();sender.terminate();raise RuntimeError('benchmark subprocess timeout')
    if camera.exitcode or sender.exitcode:raise RuntimeError('benchmark worker failed')
    result=dict(mode=mode,control=summarize(control),receiver=summarize(received),camera=results.get(timeout=2))
    result['receiver']['latencyP99Ms']=float(np.percentile(latencies,99))
    rx.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=4.)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(scope='Synthetic rate/load test; actual recorder and storage; no camera, ROS or robot commands.',
                resolution=[848,480],nominalHz=dict(control=200,receiver=200,camera=90),trials=[])
    with tempfile.TemporaryDirectory(prefix='raw-recording-benchmark-',dir=args.output.parent) as temp:
        for mode in ('off','normal_disk','slow_disk'):
            result=trial(mode,Path(temp)/mode,args.seconds)
            report['trials'].append(result)
            print(json.dumps(result),flush=True)
    atomic_json(args.output,report)


if __name__=='__main__':main()
