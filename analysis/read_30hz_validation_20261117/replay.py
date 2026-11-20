#!/usr/bin/env python3
"""Offline comparison of the actual compiled resampler against recorded flights."""
from pathlib import Path
from collections import Counter
import csv
import io
import json
import os
import subprocess
os.environ.setdefault('MPLCONFIGDIR','/tmp/quaddock_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
EXECUTABLE=ROOT/'devel/lib/read/read_odometry_replay'
events=json.loads((ROOT/'analysis/takeoff_diagnosis_20261117/evidence.json').read_text())
summary=[]
for name in sorted({r['run'] for r in events}):
    run=Path('/home/pc/perching_bag')/name
    inputs=[]
    old=[]
    truth=[]
    for p in sorted(run.glob('*.bag')):
        with rosbag.Bag(str(p)) as bag:
            for topic,m,t in bag.read_messages(topics=['/svo/pose_imu','/vins_fusion/imu_propagate',
                                                       '/evaluation/vicon/drone00/pose']):
                if topic=='/svo/pose_imu':
                    p=m.pose.pose.position
                    q=m.pose.pose.orientation
                    inputs.append([t.to_sec(),m.header.stamp.to_sec(),p.x,p.y,p.z,q.x,q.y,q.z,q.w])
                elif topic=='/vins_fusion/imu_propagate':
                    old.append([t.to_sec(),m.pose.pose.position.z,m.twist.twist.linear.z])
                else:
                    truth.append([t.to_sec(),m.pose.position.z])
    output=subprocess.run([str(EXECUTABLE)],input='\n'.join(','.join(map(str,r)) for r in inputs),
                           text=True,capture_output=True,check=True).stdout
    (OUT/(name+'_replay.csv')).write_text(output)
    rows=list(csv.DictReader(io.StringIO(output)))
    a=np.array([[float(r[k]) for k in ['receipt','valid','z','vz']] for r in rows])
    b=np.array(old)
    c=np.array(truth)
    for event in [e for e in events if e['run']==name]:
        t0=event['command_time']
        win=(a[:,0]>=t0)&(a[:,0]<=t0+5)
        sub=[r for r,keep in zip(rows,win) if keep]
        valid=a[win&(a[:,1]==1)]
        result=dict(run=name,attempt=event['attempt'],command_clock=event['command_clock'],
                    snapshots=len(sub),valid_snapshots=len(valid),
                    input_results=dict(Counter(r['input_result'] for r in sub)),samples=[])
        for dt in [1.0,2.5,3.5]:
            t=t0+dt
            sample=dict(relative_s=dt,old_vz_mps=float(np.interp(t,b[:,0],b[:,2])),
                        vicon_vz_mps=float((np.interp(t+.05,c[:,0],c[:,1])-np.interp(t-.05,c[:,0],c[:,1]))/.1))
            if len(valid):
                r=valid[np.argmin(abs(valid[:,0]-t))]
                if abs(r[0]-t)<.1:
                    sample['new_vz_mps']=float(r[3])
                    sample['new_sample_dt_s']=float(r[0]-t0)
            result['samples'].append(sample)
        summary.append(result)
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        good=a[(a[:,0]>=t0-1)&(a[:,0]<=t0+6)&(a[:,1]==1)]
        ob=b[(b[:,0]>=t0-1)&(b[:,0]<=t0+6)]
        grid=np.arange(t0-1,t0+6,.1)
        z=np.interp(grid,c[:,0],c[:,1])
        axes[0].plot(grid-t0,np.gradient(z,.1),label='Vicon vertical differences (0.1 s)',lw=1.)
        axes[0].plot(ob[:,0]-t0,ob[:,2],label='Old recorded feedback',lw=1.)
        axes[0].plot(good[:,0]-t0,good[:,3],label='New offline causal feedback',lw=1.)
        ages=np.array([[r[0]-t0,r[0]-r[1]] for r in inputs if t0-1<=r[0]<=t0+6])
        axes[1].plot(ages[:,0],ages[:,1],label='Original SVO source age')
        axes[1].axhline(.1,color='r',ls='--',label='Source age limit (0.1 s)')
        for ax in axes:
            ax.grid(alpha=.25)
            ax.axvline(0,color='gray',ls=':')
            ax.legend(fontsize=8)
        axes[0].set(ylabel='vertical velocity (m/s)',title=name+' | '+event['command_clock']+' CST')
        axes[1].set(ylabel='source age (s)',xlabel='seconds since recorded takeoff command')
        fig.tight_layout()
        fig.savefig(OUT/(name+'_attempt'+str(event['attempt'])+'.png'),dpi=150)
        plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps(summary,indent=2,ensure_ascii=False))
