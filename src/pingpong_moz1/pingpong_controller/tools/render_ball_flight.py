#!/usr/bin/env python3
"""Build a self-contained, offline trajectory film from a probe recording.

Only reads recorder files. Does not import ROS or change the robot controller.
Optional video rendering needs Playwright/Chromium and ffmpeg; HTML needs neither.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import subprocess


def finite_vector(value):
    return (isinstance(value, list) and len(value) == 3
            and all(isinstance(x, (float, int)) and math.isfinite(x) for x in value))


def build(recording: Path, output: Path, start: float, duration: float,
          playback_rate: float = 0.6):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((recording / 'metadata.json').read_text())
    summary = json.loads((recording / 'summary.json').read_text())
    rows, counts = [], dict(control=0, invalid=0, recorded_prediction=0, predicted_flag=0)
    first_time = None
    previous_time = -math.inf
    active_start, active_end = None, None
    source_counts = {}
    for line in (recording / 'control_samples.jsonl').open():
        if not line.strip():
            continue
        row = json.loads(line)
        first_time = float(row['time']) if first_time is None else first_time
        time = round(float(row['time']) - first_time, 6)
        counts['control'] += 1
        counts['recorded_prediction'] += int(row.get('predictedBall') is not None)
        counts['predicted_flag'] += int(bool(row.get('predictedBallIsPredicted')))
        source = row.get('ballSource', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
        position, velocity, racket = (row.get(k) for k in
                                      ('measuredBall', 'measuredVelocity', 'racket'))
        valid = bool(row.get('valid') and row.get('rawValid') and finite_vector(position)
                     and finite_vector(velocity))
        mode = {'release_gate': 0, 'policy': 1, 'done_failsafe': 2}.get(row['mode'], 3)
        counts['invalid'] += int(not valid)
        if mode == 1:
            active_start = time if active_start is None else active_start
            active_end = time
        # Retain every 200 Hz control query, including validity transitions.
        # The latest query at/before playback time is the ONLY prediction seed.
        rows.append([time, *(position if finite_vector(position) else [None]*3),
                     *(velocity if finite_vector(velocity) else [None]*3),
                     *(racket if finite_vector(racket) else [None]*3), int(valid), mode])
        if time < previous_time:
            raise ValueError('Control time must be monotonic')
        previous_time = time
    if not rows or active_start is None:
        raise ValueError('Recording has no policy samples')
    events = []
    for line in (recording / 'events.jsonl').open():
        event = json.loads(line)
        if event.get('event') == 'juggle_bounce':
            events.append([round(event['sourceTimeS'] - first_time, 6), event['count']])
    events.sort()
    if start < 0 or duration <= 0 or start + duration > rows[-1][0]:
        raise ValueError(f'Playback interval must be within recording 0–{rows[-1][0]:.3f} s')
    if not math.isfinite(playback_rate) or playback_rate <= 0:
        raise ValueError('Playback rate must be finite and positive')
    selected = [r for r in rows if start <= r[0] <= start + duration]
    positions = [r[1:4] for r in rows if r[10] and r[11] == 1]
    center = [sum(p[i] for p in positions)/len(positions) for i in range(3)]
    center[2] = 1.29
    payload = dict(
        recording=recording.name, recordedAt=metadata['startedLocalTime'],
        duration=rows[-1][0], active=[active_start, active_end], highlight=[start, start+duration],
        playbackRate=playback_rate,
        center=center, rows=rows, events=events, sourceCounts=source_counts,
        bounceCount=summary['juggle']['count'], counts=counts,
        model=dict(gravity=9.8, drag=0.13545, step=0.005, horizon=0.4, ballRadius=0.02),
    )
    template = Path(__file__).with_name('ball_flight_viewer.html').read_text()
    html = template.replace('/*__RECORDING_DATA__*/', json.dumps(payload, separators=(',', ':'), allow_nan=False))
    (output / 'ball_flight.html').write_text(html)
    audit = dict(
        recording=str(recording.resolve()), source='measuredBall / measuredVelocity (mocap)',
        counts=counts, sources=source_counts, recording_duration_s=rows[-1][0],
        policy_interval_s=[active_start, active_end], highlight_interval_s=payload['highlight'],
        highlight_samples=len(selected), highlight_invalid=sum(not r[10] for r in selected),
        playback_rate=playback_rate, film_duration_s=duration/playback_rate,
        bounce_events=len(events), playback_samples=len(rows), playback_sampling='all control queries, no decimation',
        prediction='offline causal free-flight rollout; NOT a recorded online prediction',
        prediction_seed='latest valid measured state at or before playback time; no future fit',
        model=payload['model'],
        termination='400 ms or downward intersection with frozen horizontal racket height + 20 mm',
        orientation='recording has racket XYZ only; horizontal schematic disc, not measured attitude',
        path='measured positions; linear display interpolation <= 40 ms within same valid mode; no interpolation across impacts',
        gap_policy='invalid state / >40 ms timestamp gap / >120 mm position jump breaks paths',
        display='physical XYZ scale preserved; 20 mm ball radius; world/control sim_frame; height is absolute Z',
        camera='orthographic projection; quintic eased camera transitions',
    )
    (output / 'data_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    report = f'''球运动轨迹动画 / BALL FLIGHT

打开 ball_flight.html 即可离线播放；全部数据内嵌，不依赖 CDN、网络或本地服务器。
ball_flight.mp4 为所选片段的视频（如已导出），播放速度 {playback_rate:g}×；默认导出 1920×1080 / 60 fps。
本段覆盖录制 {start:.3f}–{start+duration:.3f} s，视频时长 {duration/playback_rate:g} s。

操作
  空格：播放 / 暂停；左右方向键：前后 1 秒。
  1：自动运镜；2–5：斜侧、正面、侧面、俯视。
  鼠标拖动：自由旋转；滚轮：缩放；双击：恢复自动运镜。
  时间轴：定位；播放速度：0.25×–2×；全屏按钮 / F：沉浸查看。
  “精选片段 / 整段录制”切换播放范围；整段默认保留等待和结束状态。
  底部时间轴显示的是相对录制起点的控制时间（秒）。

数据与预测
  录制：{metadata['startedLocalTime']}；输入：动捕 mocap。
  共 {counts['control']:,} 条控制记录、{summary['juggle']['count']} 个记录器判定的颠球事件。
  控制策略区间：{active_start:.3f}–{active_end:.3f} s。
  默认精选：{start:.3f}–{start+duration:.3f} s；该区间无效样本：{audit['highlight_invalid']}。
  整段保留全部 {len(rows):,} 条记录；每条约 5 ms。实测轨迹不做平滑拟合。

  青色：实测历史轨迹（最近 2.6 秒）；白色：当前球；半透明球：过去 0.4 秒的历史状态。
  金色虚线：从当前时刻及以前的实测位置/速度离线前推，最长 400 ms。
  模型：dp/dt=v；dv/dt=[0,0,-9.8]−0.13545|v|v；前向 Euler，步长 5 ms。
  这些参数复用 high_ball_occlusion_predictor.py 的默认飞行参数，不代表已为本次动捕轨迹标定。
  当前 tick 的预测以最近一条已记录状态为种子，不使用后续轨迹点，也不预知下一次触拍时刻。
  到达“当前拍面 Z + 球半径”时停止，不绘制未经测量支持的碰撞反弹；拍面采用固定水平面近似。
  标注的预测终点不是已知真实落点；预测误差未在此展示中量化。
  灰色圆盘只表示实测球拍中心位置；原录制没有拍面朝向，不代表真实姿态。
  球半径 20 mm、XYZ 比例一致；网格为 50 mm；高度值为控制坐标系绝对 Z。
  失效数据断线、隐藏当前球及预测，不使用填补数据冒充测量。

当前实物代码的预测能力
  tools/interactive_policy_safety_probe.py 使用 HighBallOcclusionPredictor，为相机遮挡/高球等情况提供短时预测。
  _resolve_high_ball_occlusion_state 中 options.ball_source != "camera" 会以 camera_source_required 禁用此路径。
  ball_tracker/impact_predictor.py 另有解析触拍预测；其存在不代表本次动捕运行启用了它。
  本次 predictedBall 非空 {counts['recorded_prediction']} / {counts['control']}，predictedBallIsPredicted=True 为 {counts['predicted_flag']}。
  因此不能将本动画的金色线称为“实物运行时已经记录的预测轨迹”。

复现
  python3 src/pingpong_moz1/pingpong_controller/tools/render_ball_flight.py \\
    {recording} --start {start:g} --duration {duration:g} --playback-rate {playback_rate:g}
  加 --video 生成视频（额外需要 playwright、Chromium、ffmpeg）。
  Python 数据导出本身仅使用标准库；不会调用 ROS 或修改控制器。
  data_audit.json 记录数据来源、计数和显示规则。
'''
    (output / '使用说明.txt').write_text(report)
    print(json.dumps(dict(html=str(output/'ball_flight.html'), **counts), ensure_ascii=False), flush=True)
    return output / 'ball_flight.html'


async def video(html: Path, seconds: float, fps: int, width: int, height: int,
                start_seconds: float = 0.0, filename: str = 'ball_flight.mp4'):
    from playwright.async_api import async_playwright
    output = html.parent
    async with async_playwright() as runtime:
        browser = await runtime.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        page = await browser.new_page(viewport=dict(width=width, height=height), device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        await page.goto(html.resolve().as_uri()+'?export=1')
        await page.wait_for_function('window.ballFlightReady === true')
        await page.evaluate('document.fonts.ready')
        command = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'image2pipe', '-vcodec', 'mjpeg',
                   '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output/filename)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in range(round(seconds*fps)):
                await page.evaluate('(s) => window.renderFilmFrame(s)', start_seconds+frame/fps)
                data = await page.screenshot(type='jpeg', quality=94)
                process.stdin.write(data)
                if frame in (0, int(seconds*fps/3), int(seconds*fps*2/3)):
                    await page.screenshot(path=str(output/f'view_{round(start_seconds*fps)+frame:04d}.png'))
                if frame % (fps*5) == 0:
                    print(f'Rendered {frame}/{round(seconds*fps)} frames', flush=True)
        finally:
            process.stdin.close()
            code = process.wait()
            await browser.close()
        if code or errors:
            raise RuntimeError(f'Video export failed: ffmpeg={code}, browser={errors}')
    print(f'Video: {output / filename}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--start', type=float, default=60.0, help='seconds relative to recording start')
    parser.add_argument('--duration', type=float, default=24.0)
    parser.add_argument('--playback-rate', type=float, default=0.6)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--film-seconds', type=float, help='defaults to selected duration / playback rate')
    parser.add_argument('--film-start', type=float, default=0.0, help='film time offset for segmented export')
    parser.add_argument('--video-name', default='ball_flight.mp4')
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    args = parser.parse_args()
    if (not math.isfinite(args.playback_rate) or args.playback_rate <= 0
            or args.duration <= 0 or args.fps <= 0 or args.film_start < 0):
        parser.error('duration, playback-rate and fps must be positive; film-start must be nonnegative')
    film_seconds = (args.film_seconds if args.film_seconds is not None
                    else args.duration / args.playback_rate - args.film_start)
    if film_seconds <= 0 or (args.film_start + film_seconds) * args.playback_rate > args.duration + 1e-6:
        parser.error('film interval must fit within the selected data interval')
    output = args.output or args.recording / 'analysis' / 'ball_flight'
    html = build(args.recording, output, args.start, args.duration, args.playback_rate)
    if args.video:
        asyncio.run(video(html, film_seconds, args.fps, args.width, args.height,
                          args.film_start, args.video_name))


if __name__ == '__main__':
    main()
