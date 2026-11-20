#!/usr/bin/python3
"""Read-only comparison of flight phases; no ground-truth alignment fit.

Uses analyzer CSV for reconstructed I and raw messages for gaps/readiness.
Phase durations come from mode events, not interpolation over command gaps.
All times displayed in Asia/Shanghai (UTC+8).
"""
import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag

ROOT = Path(__file__).resolve().parent
ZONE = dt.timezone(dt.timedelta(hours=8))
RUNS = ['20261120_181526_vicon_05b7fc', '20261120_183726_vicon_0f0610']


def clock(t):
    return dt.datetime.fromtimestamp(float(t), ZONE).strftime('%H:%M:%S.%f')[:-3]


def analyze(run):
    folder = Path('/home/pc/perching_bag') / run
    output = ROOT / ('integral_diagnosis_' + run[:15])
    records = list(csv.DictReader((output/'integral_timeline.csv').open()))
    parameters = json.loads((output/'summary.json').read_text())['parameters']
    columns = [k for k in records[0] if k != 'clock']
    data = {k:np.array([float(r[k]) for r in records]) for k in columns}
    modes, takeoffs = [], []
    streams = {t:[] for t in ['/target_ekf_odom','/dog_pos_processed','/vins_fusion/imu_propagate',
                            '/position_cmd','/AOA_Tag_data','/read/odometry_diagnostics',
                            '/quaddock/ours_sw3/status','/svo/info']}
    with rosbag.Bag(str(folder/'data_0.bag')) as b:
        end = b.get_end_time()
        for topic, m, receipt in b.read_messages(topics=list(streams)+['/mode_manager','/px4ctrl/takeoff_land']):
            t = receipt.to_sec()
            if topic == '/mode_manager':
                modes.append((t,int(m.pose.orientation.w)))
            elif topic == '/px4ctrl/takeoff_land':
                takeoffs.append((t,m.takeoff_land_cmd))
            elif topic == '/dog_pos_processed':
                p=m.pose.pose
                streams[topic].append([t,m.header.stamp.to_sec(),p.orientation.w,p.orientation.x,
                                      p.position.x,p.position.y,p.position.z,
                                      m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.angular.x])
            elif topic == '/read/odometry_diagnostics':
                v={v.key:v.value for s in m.status for v in s.values}
                try:age=float(v.get('source_age_s','nan'))
                except ValueError:age=float('nan')
                streams[topic].append([t,int(v.get('output_valid')=='true'),age])
            elif topic == '/quaddock/ours_sw3/status':
                d=json.loads(m.data)
                streams[topic].append((t,d))
            elif topic == '/target_ekf_odom':
                p=m.pose.pose
                streams[topic].append([t,m.header.stamp.to_sec(),p.position.x,p.position.y,p.position.z,p.orientation.w])
            elif topic == '/vins_fusion/imu_propagate':
                p=m.pose.pose
                streams[topic].append([t,m.header.stamp.to_sec(),p.position.x,p.position.y,p.position.z,
                                      m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.linear.z])
            else:
                streams[topic].append([t,m.header.stamp.to_sec()])
    takeoff = next(t for t,m in takeoffs if m==1)
    ret = next(t for t,m in modes if m==0 and t>takeoff)
    precise = next(t for t,m in modes if m==1 and t>ret)
    stop = next(t for t,m in modes if m==-1 and t>precise)
    phases = []
    relevant = [(t,m) for t,m in modes if ret<=t<=stop]
    for (a,m),(b,_) in zip(relevant[:-1],relevant[1:]):
        phases.append(dict(start=a,end=b,start_clock=clock(a),end_clock=clock(b),mode=m,duration=b-a))
    def scalar_metrics(values):
        return dict(p50=float(np.median(values)),p95=float(np.percentile(values,95)),
                    max=float(np.max(values)),rmse=float(np.sqrt(np.mean(values**2))))
    stats={}
    for mode,label in [(0,'return'),(1,'precise'),(2,'descent')]:
        mask=(data['time']>=ret)&(data['time']<stop)&(data['mode']==mode)
        s=dict(duration=sum(p['duration'] for p in phases if p['mode']==mode),
               samples=int(mask.sum()))
        for k in ['truth_xy','est_xy','truth_rel_speed','vision_age']:
            s[k]=scalar_metrics(data[k][mask])
        s['distance_discrepancy_lower_bound']=scalar_metrics(np.abs(data['truth_xy'][mask]-data['est_xy'][mask]))
        for axis_index,ax in enumerate(('x','y')):
            a=np.abs(data['I_'+ax][mask])
            limit=parameters['integral_limits'][axis_index]
            s['integral_'+ax]=dict(abs_max=float(a.max()),near_limit_fraction=float(np.mean(a>=.99*limit)),
                                   Ivel_abs_p95=float(np.percentile(np.abs(data['Ivel_'+ax][mask]),95)))
        stats[label]=s
    def gaps(topic, threshold, start=ret, finish=stop):
        stamps=np.array([r[0] for r in streams[topic]])
        result=[]
        for a,b in zip(stamps[:-1],stamps[1:]):
            if b-a>=threshold and b>start and a<finish:
                result.append(dict(start=a,end=b,start_clock=clock(a),end_clock=clock(b),duration=b-a,
                                   overlap_with_flight=max(0,min(b,finish)-max(a,start))))
        return result
    dog=np.array(streams['/dog_pos_processed'])
    readiness=[]
    for a,b in zip(dog[:-1],dog[1:]):
        if b[0]>ret and a[0]<stop and (a[2]!=b[2] or a[3]!=b[3]):
            readiness.append(dict(clock=clock(b[0]),time=b[0],pos_ready=b[2],yaw_ready=b[3]))
    def intervals(column, value):
        result=[];begin=None
        for a,b in zip(dog[:-1],dog[1:]):
            if b[0]<=ret or a[0]>=stop:continue
            if a[column]==value:
                if begin is None:begin=max(a[0],ret)
            elif begin is not None:
                result.append(dict(start_clock=clock(begin),end_clock=clock(a[0]),duration=a[0]-begin));begin=None
        if begin is not None:result.append(dict(start_clock=clock(begin),end_clock=clock(stop),duration=stop-begin))
        return result
    events={}
    for label,when in [('return_start',ret),('precise_start',precise),('stop',stop)]:
        index=int(np.argmin(np.abs(data['time']-when)))
        events[label]={k:float(v[index]) for k,v in data.items()}
        events[label]['clock']=clock(data['time'][index])
    peak=int(np.argmax(np.where((data['time']>=ret)&(data['time']<=stop),data['truth_xy'],-1)))
    events['peak']={k:float(v[peak]) for k,v in data.items()};events['peak']['clock']=clock(data['time'][peak])
    read=np.array(streams['/read/odometry_diagnostics'])
    rm=(read[:,0]>=ret)&(read[:,0]<=stop)
    status=[dict(time=t,clock=clock(t),data=d) for t,d in streams['/quaddock/ours_sw3/status'] if ret<=t<=stop]
    # Keep compact status samples at roughly one second; counters remain inspectable.
    status_sample=[];last=-float('inf')
    for row in status:
        if row['time']-last>=1:status_sample.append(row);last=row['time']
    result=dict(run=run,parameters=parameters,phases=phases,return_to_precise=precise-ret,return_to_stop=stop-ret,
                stats=stats,readiness_changes=readiness,
                pos_not_ready_intervals=intervals(2,0),yaw_not_ready_intervals=intervals(3,0),
                vision_gaps=gaps('/target_ekf_odom',.5),cmd_gaps=gaps('/position_cmd',.05),
                odom_gaps=gaps('/vins_fusion/imu_propagate',.05),uwb_gaps=gaps('/AOA_Tag_data',.6),
                read_invalid_count=int(np.sum(read[rm,1]==0)),
                vio_source_age=scalar_metrics(read[rm,2]),events=events,status_sample=status_sample)
    (output/'flight_comparison_details.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    # Numerical raw streams allow later offline inspection without rereading the bag.
    np.savez_compressed(output/'streams.npz',**{k.strip('/').replace('/','__'):np.asarray(v) for k,v in streams.items() if k!='/quaddock/ours_sw3/status'})
    return result,data,streams


def main():
    fig,axes=plt.subplots(4,2,figsize=(14,11))
    all_results=[]
    for col,run in enumerate(RUNS):
        result,d,streams=analyze(run);all_results.append(result)
        ret=result['phases'][0]['start'];stop=result['phases'][-1]['end'];precise=ret+result['return_to_precise']
        mask=(d['time']>=ret)&(d['time']<=stop);x=d['time'][mask]-ret
        axes[0,col].plot(x,d['truth_xy'][mask],label='Vicon horizontal distance')
        axes[0,col].plot(x,d['est_xy'][mask],label='Estimated distance',alpha=.8)
        axes[0,col].set_title(run[:15]+'; X limit '+('0.7' if col==0 else '0.4'))
        axes[0,col].set_ylabel('Distance (m)')
        axes[1,col].plot(x,d['vision_age'][mask],label='ArUco age')
        dog=np.array(streams['/dog_pos_processed']);dm=(dog[:,0]>=ret)&(dog[:,0]<=stop)
        axes[1,col].step(dog[dm,0]-ret,dog[dm,2],where='post',label='Position ready')
        axes[1,col].step(dog[dm,0]-ret,dog[dm,3],where='post',label='Yaw ready',alpha=.6)
        axes[1,col].set_ylabel('Age (s) / readiness')
        axes[2,col].plot(x,d['Ivel_x'][mask],label='Reconstructed X I velocity')
        axes[2,col].plot(x,d['Ivel_y'][mask],label='Reconstructed Y I velocity')
        axes[2,col].set_ylabel('I velocity (m/s)')
        axes[3,col].plot(x,d['truth_rel_speed'][mask],label='Vicon relative XY speed')
        axes[3,col].plot(x,d['dog_speed'][mask],label='Estimated dog speed')
        axes[3,col].set_ylabel('Speed (m/s)');axes[3,col].set_xlabel('Seconds after return '+clock(ret))
        for ax in axes[:,col]:
            for phase in result['phases']:
                if phase['mode']==2:ax.axvspan(phase['start']-ret,phase['end']-ret,color='green',alpha=.10)
            ax.axvline(precise-ret,color='black',ls='--',lw=1,label='First precise mode')
            ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right');ax.set_xlim(0,stop-ret)
    fig.suptitle('Two separate flights: motion and visual availability differ; not a controlled gain comparison')
    fig.subplots_adjust(left=.065,right=.99,top=.94,bottom=.07,hspace=.25,wspace=.20)
    output=ROOT/'integral_diagnosis_20261120_183726'
    fig.savefig(output/'comparison_overview.png',dpi=145)
    slim=[{k:v for k,v in r.items() if k!='status_sample'} for r in all_results]
    (output/'comparison.json').write_text(json.dumps(slim,ensure_ascii=False,indent=2))
    print(json.dumps(slim,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
