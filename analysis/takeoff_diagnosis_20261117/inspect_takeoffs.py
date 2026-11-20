#!/usr/bin/env python3
"""Offline takeoff inspection. Reads bags only; never connects to ROS or a vehicle."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/quaddock_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TZ=datetime.timezone(datetime.timedelta(hours=8))
RUNS=['20261117_180135_vicon_181620','20261117_180844_vicon_efc9be']
POSES=['/svo/pose_imu','/vins_fusion/imu_propagate','/evaluation/vicon/drone00/pose',
       '/dog_pos_processed','/quaddock/target_odom','/flow_data']
reports=[]
csv_rows=[]

def clock(t):
    return datetime.datetime.fromtimestamp(t,TZ).strftime('%H:%M:%S.%f')[:-3]

def nearest(series,t):
    a=np.asarray(series,dtype=float)
    idx=int(np.argmin(abs(a[:,0]-t)))
    return a[idx] if abs(a[idx,0]-t)<.25 else None

for name in RUNS:
    run=Path('/home/pc/perching_bag')/name
    poses={t:[] for t in POSES}
    commands=[]
    logs=[]
    states=[]
    publishers=set()
    for bagpath in sorted(run.glob('*.bag')):
        with rosbag.Bag(str(bagpath)) as bag:
            for c in bag._get_connections():
                if c.topic in POSES:
                    publishers.add((c.topic,str(c.header.get('callerid')),c.datatype))
            topics=POSES+['/rosout','/px4ctrl/takeoff_land','/mavros/state']
            for topic,m,t in bag.read_messages(topics=topics):
                time=t.to_sec()
                if topic in poses:
                    p=m.pose if topic.endswith('/pose') else m.pose.pose
                    v=m.twist.twist.linear if hasattr(m,'twist') else None
                    poses[topic].append([time,m.header.stamp.to_sec(),p.position.x,p.position.y,p.position.z,
                                        v.x if v else 0.,v.y if v else 0.,v.z if v else 0.])
                elif topic=='/rosout':
                    logs.append(dict(time=time,clock=clock(time),node=m.name,level=m.level,message=m.msg))
                elif topic=='/px4ctrl/takeoff_land' and m.takeoff_land_cmd==1:
                    commands.append(time)
                elif topic=='/mavros/state':
                    states.append(dict(time=time,mode=m.mode,armed=m.armed))
    launches=[r for r in logs if r['node']=='/px4ctrl' and 'MANUAL_CTRL(L1) --> AUTO_TAKEOFF' in r['message']]
    matched_commands=[min(commands,key=lambda t:abs(t-r['time'])) for r in launches]
    snapshot=run/'source_snapshot/src/read/src/read.cpp'
    source_matches=(hashlib.sha256(snapshot.read_bytes()).digest()==
                    hashlib.sha256((ROOT/'src/read/src/read.cpp').read_bytes()).digest())
    for index,(event,launch) in enumerate(zip(matched_commands,launches),1):
        event_logs=[dict(relative_s=r['time']-event,**r) for r in logs
                    if event-10<=r['time']<=event+25 and r['node']=='/px4ctrl']
        event_states=[]
        for s in states:
            if event-1<=s['time']<=event+25:
                if not event_states or (s['mode'],s['armed'])!=(event_states[-1]['mode'],event_states[-1]['armed']):
                    event_states.append(dict(relative_s=s['time']-event,**s))
        metrics=[]
        for rel in [0.,.5,1.,1.5,2.,2.5,3.,4.,5.]:
            row=dict(run=name,attempt=index,command_clock=clock(event),sample_relative_s=rel)
            for topic,label in [('/svo/pose_imu','svo'),('/vins_fusion/imu_propagate','control_odom'),
                                ('/evaluation/vicon/drone00/pose','vicon')]:
                value=nearest(poses[topic],event+rel)
                base=nearest(poses[topic],event)
                if value is None or base is None:
                    continue
                row.update({label+'_sample_relative_s':float(value[0]-event),
                            label+'_header_age_s':float(value[0]-value[1]),
                            label+'_z_m':float(value[4]), label+'_delta_z_m':float(value[4]-base[4]),
                            label+'_delta_xy_m':float(np.linalg.norm(value[2:4]-base[2:4])),
                            label+'_reported_vz_mps':float(value[7]) if label=='control_odom' else None})
            metrics.append(row)
            csv_rows.append(row)
        relays=[r for r in poses['/vins_fusion/imu_propagate'] if event<=r[0]<=event+5]
        raw=[r for r in poses['/svo/pose_imu'] if event<=r[0]<=event+5]
        report=dict(run=name,attempt=index,command_time=event,command_clock=clock(event),
                    transition_clock=launch['clock'],px4ctrl_logs=event_logs,state_changes=event_states,
                    samples=metrics,publishers=sorted(publishers),read_source_snapshot_matches_current=source_matches,
                    raw_svo_age_p50_s=float(np.median([r[0]-r[1] for r in raw])),
                    raw_svo_age_max_s=float(max(r[0]-r[1] for r in raw)),
                    relay_age_p95_s=float(np.percentile([r[0]-r[1] for r in relays],95)),
                    any_cmd_ctrl_log=any('CMD_CTRL' in r['message'] for r in event_logs))
        reports.append(report)
        fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for topic,label in [('/evaluation/vicon/drone00/pose','Vicon body height change'),
                            ('/svo/pose_imu','Raw SchurVINS (receipt time)'),
                            ('/vins_fusion/imu_propagate','Odometry consumed by controller')]:
            a=np.array(poses[topic],float)
            a=a[(a[:,0]>=event-2)&(a[:,0]<=event+8)]
            base=nearest(poses[topic],event)
            axs[0].plot(a[:,0]-event,a[:,4]-base[4],label=label,lw=1.)
            if topic!='/evaluation/vicon/drone00/pose':
                axs[1].plot(a[:,0]-event,a[:,0]-a[:,1],label=label,lw=1.)
            if topic=='/vins_fusion/imu_propagate':
                axs[2].plot(a[:,0]-event,a[:,7],label='Reported vertical velocity',lw=1.)
        a=np.array(poses['/evaluation/vicon/drone00/pose'],float)
        # Differences over 100 ms suppress Vicon sample-level noise.
        grid=np.arange(event-2,event+8,.1)
        z=np.interp(grid,a[:,0],a[:,4])
        axs[2].plot(grid-event,np.gradient(z,.1),label='Vicon vertical velocity (0.1 s differences)',lw=1.)
        for r in event_logs:
            if 'AUTO_TAKEOFF --> AUTO_HOVER' in r['message']:
                for ax in axs:
                    ax.axvline(r['relative_s'],color='k',ls='--',alpha=.5)
        for ax in axs:
            ax.axvline(0,color='gray',ls=':')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8,loc='upper right')
        axs[0].set(ylabel='height change (m)',title=name+' | takeoff '+clock(event)+' CST')
        axs[1].set(ylabel='receipt minus header (s)')
        axs[2].set(ylabel='vertical velocity (m/s)',xlabel='seconds since takeoff command')
        fig.tight_layout()
        fig.savefig(OUT/(name+'_attempt'+str(index)+'.png'),dpi=150)
        plt.close(fig)
        print(json.dumps({k:report[k] for k in ['run','attempt','command_clock','raw_svo_age_p50_s','raw_svo_age_max_s','relay_age_p95_s','any_cmd_ctrl_log']},ensure_ascii=False))

(OUT/'evidence.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False))
fields=list(dict.fromkeys(k for row in csv_rows for k in row))
with (OUT/'takeoff_samples.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(csv_rows)
