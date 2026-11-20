#!/usr/bin/env python3
"""Read-only source inspection and a separate core-topic derivative of this run."""
import csv
import json
import os
from pathlib import Path
import re
import struct
from collections import Counter, defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quaddock_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG = ROOT / "src/quaddock_logging/config/recording.yaml"
RUNS = [Path('/home/pc/perching_bag/20261117_173156_vicon_1a115d'),
        Path('/home/pc/perching_bag/20261117_173448_vicon_e852e7')]
cfg = yaml.safe_load(CONFIG.read_text())
selected = set(cfg['topics'])


def keep(topic):
    return topic in selected or any(re.fullmatch(p, topic) for p in cfg['topic_regex'])


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def intervals(times, max_gap):
    if not len(times):
        return []
    times = np.asarray(times)
    breaks = np.flatnonzero(np.diff(times) > max_gap)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(times) - 1]
    return [(float(times[a]), float(times[b])) for a, b in zip(starts, ends)]


summaries = []
topic_rows = []
for run in RUNS:
    latest = run == RUNS[-1]
    source = run / 'data_0.bag'
    stats = defaultdict(lambda: dict(times=[], stamps=[], ages=[], bytes=0, type=''))
    poses = defaultdict(list)
    diagnostic = defaultdict(list)
    status = []
    events, logs, svo = [], [], []
    frames, latency = [], []
    writer = rosbag.Bag(str(OUT / 'data_core.bag'), 'w') if latest else None
    try:
        with rosbag.Bag(str(source)) as bag:
            start, end = bag.get_start_time(), bag.get_end_time()
            for entry in bag.read_messages(raw=True, return_connection_header=True):
                topic, raw, stamp, header = entry
                datatype, data = raw[:2]
                cls = raw[-1]
                t = stamp.to_sec()
                row = stats[topic]
                row['times'].append(t - start)
                row['bytes'] += len(data)
                row['type'] = datatype
                if writer is not None and keep(topic):
                    writer.write(topic, raw, stamp, raw=True, connection_header=header)
                if cls._has_header:
                    sec, nsec = struct.unpack_from('<II', data, 4)
                    h = sec + nsec * 1e-9
                    row['stamps'].append(h)
                    row['ages'].append(t - h)
                if datatype in ('nav_msgs/Odometry', 'geometry_msgs/PoseStamped',
                                'geometry_msgs/PoseWithCovarianceStamped'):
                    m = cls().deserialize(data)
                    p = m.pose if datatype == 'geometry_msgs/PoseStamped' else m.pose.pose
                    poses[topic].append([t - start, p.position.x, p.position.y, p.position.z,
                                         p.orientation.x, p.orientation.y, p.orientation.z,
                                         p.orientation.w, m.header.stamp.to_sec(), m.header.frame_id])
                elif topic == '/evaluation/vicon/diagnostics':
                    m = cls().deserialize(data)
                    for s in m.status:
                        kv = {v.key: v.value for v in s.values}
                        if s.name == 'vicon/stream':
                            frames.append((t-start, int(kv.get('frame_number', -1))))
                            if 'sdk_total_latency_s' in kv:
                                latency.append(float(kv['sdk_total_latency_s']))
                        else:
                            diagnostic[s.name].append((t-start, kv.get('raw_pose_valid') == 'true'))
                elif topic == '/quaddock/ours_sw3/status':
                    status.append((t-start, json.loads(cls().deserialize(data).data)))
                elif topic == '/quaddock/experiment/events':
                    events.append((t-start, cls().deserialize(data).data))
                elif topic == '/svo/info':
                    m = cls().deserialize(data)
                    svo.append((t-start, int(m.stage), int(m.tracking_quality), m.processing_time))
                elif topic == '/rosout':
                    m = cls().deserialize(data)
                    if m.level >= 4:
                        logs.append((t-start, int(m.level), m.name, m.msg))
    finally:
        if writer is not None:
            writer.close()
    for topic, r in stats.items():
        times = np.array(r['times'])
        diff = np.diff(r['stamps'])
        row = dict(run=run.name, topic=topic, msg_type=r['type'], messages=len(times),
                   first_s=float(times[0]), last_s=float(times[-1]),
                   median_hz=1 / float(np.median(np.diff(times))) if len(times)>1 else None,
                   max_receipt_gap_s=float(np.max(np.diff(times))) if len(times)>1 else None,
                   serialized_bytes=r['bytes'], keep_core=keep(topic),
                   backwards_header_stamps=int(np.sum(diff<0)), duplicate_header_stamps=int(np.sum(diff==0)),
                   zero_header_stamps=int(np.sum(np.array(r['stamps'])==0)),
                   header_age_p50_s=percentile(r['ages'], 50), header_age_p95_s=percentile(r['ages'], 95))
        topic_rows.append(row)
    vicon = {}
    for subject in ['drone00', 'quadruped00']:
        topic = '/evaluation/vicon/' + subject + '/pose'
        series = poses.get(topic, [])
        valid = diagnostic.get('vicon/'+subject, [])
        vicon[subject] = dict(valid_diagnostic_fraction=float(np.mean([v[1] for v in valid])) if valid else None,
                              pose_intervals_gap_0p1s=intervals(stats[topic]['times'], .1))
        if series:
            xyz = np.array([r[1:4] for r in series], dtype=float)
            ts = np.array([r[0] for r in series])
            dt = np.diff(ts)
            jump = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            candidates = np.flatnonzero((dt < .1) & (jump > .1))
            vicon[subject].update(position_range_m=np.ptp(xyz, axis=0).tolist(),
                finite_positions=bool(np.all(np.isfinite(xyz))),
                large_adjacent_steps_over_0p1m=[dict(time_s=float(ts[i+1]), step_m=float(jump[i]),
                                                   dt_s=float(dt[i])) for i in candidates])
    states = []
    for t, payload in status:
        name = payload['status']
        if not states or states[-1]['status'] != name:
            states.append(dict(start_s=t, end_s=t, status=name))
        states[-1]['end_s'] = t
    costs = [p['solve_ms'] for _, p in status if p.get('solve_ms') is not None]
    errors = Counter(p.get('last_error') for _,p in status if p.get('last_error'))
    position_steps = {}
    for topic, series in poses.items():
        a = np.array([r[:4] for r in series], dtype=float)
        dt = np.diff(a[:,0])
        steps = np.linalg.norm(np.diff(a[:,1:],axis=0),axis=1)
        near = (dt > 0) & (dt < .1)
        idx = np.flatnonzero(near & (steps > .5))
        position_steps[topic] = dict(max_step_under_0p1s_m=float(max(steps[near])) if near.any() else None,
            steps_over_0p5m=[dict(time_s=float(a[i+1,0]), step_m=float(steps[i]), dt_s=float(dt[i])) for i in idx])
    summary = dict(run=run.name, source=str(source), size_bytes=source.stat().st_size,
        duration_s=end-start, start_ros_time=start, vicon=vicon, status_intervals=states,
        last_status=status[-1][1] if status else None, estimator_errors=dict(errors), position_steps=position_steps,
        solve_ms_p50=percentile(costs,50), solve_ms_p95=percentile(costs,95),
        svo_states=dict(Counter(str((v[1],v[2])) for v in svo)),
        events=events, warning_error_messages=logs,
        sdk_latency_p50_s=percentile(latency,50), sdk_latency_p95_s=percentile(latency,95),
        frame_number_duplicates=sum(b[1]==a[1] for a,b in zip(frames,frames[1:])),
        frame_number_backwards=sum(b[1]<a[1] for a,b in zip(frames,frames[1:])),
        frame_number_skipped=sum(max(0,b[1]-a[1]-1) for a,b in zip(frames,frames[1:])),
        source_metadata=json.loads((run/'run.json').read_text()),
        vision_message_intervals_gap_0p5s=intervals(stats['/dog_pos_processor/sw3/vision']['times'],.5),
        target_output_intervals_gap_0p1s=intervals(stats['/quaddock/target_odom']['times'],.1),
        selected_serialized_bytes=sum(v['bytes'] for k,v in stats.items() if keep(k)),
        image_serialized_bytes=sum(v['bytes'] for k,v in stats.items() if v['type']=='sensor_msgs/Image'))
    # Metadata inside a filtered bag still describes the original acquisition.
    if latest:
        manifest = dict(source=str(source), derivative=str(OUT/'data_core.bag'),
                        derivative_size_bytes=(OUT/'data_core.bag').stat().st_size,
                        topics=sorted(k for k in stats if keep(k)),
                        note='Filtered copy; timestamps, payloads and connection headers preserved. Embedded acquisition metadata describes the original full recording.',
                        config=cfg)
        (OUT/'core_filter_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    summaries.append(summary)

    fig, axes = plt.subplots(3,1,figsize=(13,10),gridspec_kw={'height_ratios':[1,1,1.4]})
    for subject in ['drone00','quadruped00']:
        arr = poses.get('/evaluation/vicon/'+subject+'/pose', [])
        if arr:
            xyz = np.array([r[:4] for r in arr], dtype=float)
            axes[0].plot(xyz[:,1],xyz[:,2],label=subject,lw=.8)
    axes[0].set(xlabel='mocap X (m)',ylabel='mocap Y (m)',title=run.name+' | Raw Vicon trajectories (no alignment fitted)')
    if axes[0].lines:
        axes[0].legend()
    axes[0].axis('equal')
    target=poses.get('/quaddock/target_odom',[])
    uav=poses.get('/vins_fusion/imu_propagate',[])
    if target and uav:
        a=np.array([r[:4] for r in target],float)
        b=np.array([r[:4] for r in uav],float)
        indices=np.searchsorted(b[:,0],a[:,0]).clip(1,len(b)-1)
        indices=np.where(abs(b[indices-1,0]-a[:,0])<abs(b[indices,0]-a[:,0]),indices-1,indices)
        good=abs(b[indices,0]-a[:,0])<.05
        for i,name in enumerate('XYZ'):
            axes[1].plot(a[good,0],a[good,i+1]-b[indices[good],i+1],label=name,lw=.8)
        axes[1].legend()
        for event in position_steps['/quaddock/target_odom']['steps_over_0p5m']:
            axes[1].axvline(event['time_s'],color='r',alpha=.25,lw=.6)
    axes[1].set(xlabel='time since bag start (s)',ylabel='relative coordinate (m)',
                title='Ours target minus UAV in world frame; red: target steps >0.5 m / <0.1 s; no Vicon calibration')
    timeline=[('/evaluation/vicon/drone00/pose',.1,'Vicon drone00'),
              ('/evaluation/vicon/quadruped00/pose',.1,'Vicon quadruped00'),
              ('/vins_fusion/imu_propagate',.5,'UAV odometry'),
              ('/dog_pos',.5,'Dog odometry'),
              ('/svo/info',.5,'SchurVINS info'),
              ('/dog_pos_processor/sw3/vision',.5,'Vision -> SW3'),
              ('/AOA_Tag_data',1.,'UWB AOA'),
              ('/quaddock/target_odom',.1,'Ours output')]
    for i,(topic,gap,label) in enumerate(timeline):
        spans=intervals(stats[topic]['times'],gap)
        axes[2].broken_barh([(s,max(e-s,.03)) for s,e in spans],(i-.3,.6),facecolors='C'+str(i))
    axes[2].set(yticks=range(len(timeline)),yticklabels=[t[2] for t in timeline],xlim=(0,end-start),
                xlabel='time since bag start (s)',title='Message coverage; gaps split at topic-specific thresholds (see report)')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT/(run.name+'_overview.png'),dpi=150)
    plt.close(fig)

(OUT/'inspection.json').write_text(json.dumps(summaries,indent=2,ensure_ascii=False))
with (OUT/'topic_statistics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(topic_rows[0]))
    writer.writeheader()
    writer.writerows(topic_rows)
run_rows=[]
for s in summaries:
    common=s['target_output_intervals_gap_0p1s']
    for subject in ('drone00','quadruped00'):
        other=s['vicon'][subject]['pose_intervals_gap_0p1s']
        common=[(max(a,c),min(b,d)) for a,b in common for c,d in other if min(b,d)>max(a,c)]
    last=s['last_status'] or {}
    run_rows.append(dict(run=s['run'], bag_path=s['source'], duration_s=s['duration_s'],
        size_bytes=s['size_bytes'], common_message_coverage_s=sum(b-a for a,b in common),
        normal_position_rmse_m='', blackout_position_rmse_m='', position_p95_m='',
        accuracy_note='Pending clock synchronization and subject/task-point extrinsics',
        fused_vision=last.get('counters',{}).get('fused_vision',0),
        fused_uwb_range=last.get('counters',{}).get('fused_uwb_range',0),
        fused_uwb_bearing=last.get('counters',{}).get('fused_uwb_bearing',0),
        solve_ms_p95=s['solve_ms_p95']))
with (OUT/'run_statistics.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(run_rows[0]))
    writer.writeheader()
    writer.writerows(run_rows)
print(json.dumps([dict(run=s['run'],duration_s=s['duration_s'],size_bytes=s['size_bytes'],
                      selected_serialized_bytes=s['selected_serialized_bytes'],
                      image_serialized_bytes=s['image_serialized_bytes'],vicon=s['vicon'],
                      status_intervals=s['status_intervals'],last_status=s['last_status'])
                  for s in summaries],indent=2,ensure_ascii=False))
