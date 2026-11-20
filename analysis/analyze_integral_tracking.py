#!/usr/bin/python3
"""Read-only bag diagnosis. Reconstruct integral state, never replay ROS topics.

Input association uses latest message published before each command timestamp.
This approximates subscriber callback ordering; integral is NOT directly logged.
Vicon interpolation is evaluation only. No alignment or ground-truth fit is used.
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rosbag

ODOM = '/vins_fusion/imu_propagate'
DOG = '/dog_pos_processed'
CMD = '/position_cmd'
VU = '/evaluation/vicon/drone00/pose'
VD = '/evaluation/vicon/quadruped00/pose'
VISION = '/target_ekf_odom'
ZONE = dt.timezone(dt.timedelta(hours=8))


def xyz(v):
    return [v.x, v.y, v.z]


def clock(t):
    return dt.datetime.fromtimestamp(float(t), ZONE).strftime('%H:%M:%S.%f')[:-3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    src = (args.run / 'source_snapshot/src/planning/src/traj_server.cpp').read_text()
    def param(name):
        return float(re.search(r'(?:const )?double ' + name + r'\s*=\s*([-\d.]+)', src)[1])
    gain = np.array([param('x_i'), param('y_i')])
    limits = np.array([param('integral_limit_x'), param('integral_limit_y')])
    step = param('accel_dt')
    tables = {t: [] for t in (ODOM, DOG, CMD, VU, VD, VISION)}
    modes, diagnostics, read_diag = [], [], []
    with rosbag.Bag(str(args.run / 'data_0.bag')) as b:
        start, end = b.get_start_time(), b.get_end_time()
        topics = list(tables) + ['/mode_manager', '/evaluation/vicon/diagnostics', '/read/odometry_diagnostics']
        for topic, m, receipt in b.read_messages(topics=topics):
            t = receipt.to_sec()
            if topic == '/mode_manager':
                modes.append([t, int(m.pose.orientation.w)])
            elif topic == '/evaluation/vicon/diagnostics':
                valid = {s.name: dict((v.key, v.value) for v in s.values).get('raw_pose_valid') == 'true' for s in m.status}
                diagnostics.append([m.header.stamp.to_sec(), int(valid.get('vicon/drone00', False)), int(valid.get('vicon/quadruped00', False))])
            elif topic == '/read/odometry_diagnostics':
                v = dict((v.key, v.value) for s in m.status for v in s.values)
                read_diag.append([m.header.stamp.to_sec(), v.get('output_valid'), v.get('source_age_s')])
            elif topic == CMD:
                tables[topic].append([m.header.stamp.to_sec(), t] + xyz(m.position) + xyz(m.velocity))
            elif topic in (VU, VD):
                tables[topic].append([m.header.stamp.to_sec(), t] + xyz(m.pose.position))
            else:
                p = m.pose.pose
                tables[topic].append([m.header.stamp.to_sec(), t] + xyz(p.position) + xyz(m.twist.twist.linear) + [m.twist.twist.angular.x, p.orientation.w, p.orientation.x])
    for topic in tables:
        tables[topic] = np.array(tables[topic], dtype=float)
        tables[topic] = tables[topic][np.argsort(tables[topic][:, 0], kind='stable')]
    def previous(topic, t):
        a = tables[topic]
        i = np.searchsorted(a[:, 0], t, side='right') - 1
        return a[max(0, i)]
    def interp(topic, t):
        a = tables[topic]
        return np.array([np.interp(t, a[:, 0], a[:, j]) for j in range(2, 5)])
    def rotate(v, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c*v[0]+s*v[1], -s*v[0]+c*v[1]])
    mi, mode = 0, -1
    world_i, body_i = np.zeros(2), np.zeros(2)
    rows = []
    last_cmd = None
    max_accel, max_jerk = param('max_accel'), param('max_jerk')
    last_acc = np.zeros(3)
    for cmd in tables[CMD]:
        t = cmd[0]
        dog, uav = previous(DOG, t), previous(ODOM, t)
        yaw = dog[8]
        while mi < len(modes) and modes[mi][0] <= t:
            new = modes[mi][1]
            if mode in (0, -2) and new in (1, 2):
                body_i = rotate(world_i, yaw)
            elif mode in (1, 2) and new in (0, -2):
                world_i = rotate(body_i, -yaw)
            if new == -1:
                world_i[:], body_i[:] = 0, 0
            mode = new
            mi += 1
        if mode not in (0, -2, 1, 2):
            continue
        # cmdCallback explicitly uses cos=1/sin=0 in MPC modes.
        yaw = 0.0 if mode in (0, -2) else yaw
        used_i = world_i.copy() if mode in (0, -2) else body_i.copy()
        ki = np.array([param('mpc_x_i'), param('mpc_y_i')]) if mode in (0, -2) else gain
        err_body = rotate(dog[2:4] - uav[2:4], yaw)
        update_error = rotate(cmd[2:4] - uav[2:4], yaw)
        i_term = ki * used_i
        raw_v = dog[5:7] + rotate(0.3*err_body + i_term, -yaw)
        model_residual = np.nan
        if last_cmd is not None:
            # Same full 3-D limiter, using recorded z command to isolate XY model.
            pred = np.r_[raw_v, cmd[7]]
            pred = np.clip(pred, [-1.5, -1.5, -.8], [1.5, 1.5, .8])
            desired = (pred-last_cmd[5:8])/step
            da = desired-last_acc
            if np.linalg.norm(da) > max_jerk*step:
                da *= max_jerk*step/np.linalg.norm(da)
            acc = last_acc+da
            if np.linalg.norm(acc) > max_accel:
                acc *= max_accel/np.linalg.norm(acc)
            pred = last_cmd[5:8]+acc*step
            # Raw MPC trajectory is not recorded; only assess this model in 1/2.
            if mode in (1, 2):
                model_residual = np.linalg.norm(pred[:2]-cmd[5:7])
            # The true limiter acceleration is exactly output command difference.
            last_acc = (cmd[5:8]-last_cmd[5:8])/step
        last_cmd = cmd
        true_rel = interp(VD, t)-interp(VU, t)
        # Derivative over 0.2 s, diagnostic only; not used by estimation/control.
        true_vel = ((interp(VD, t+.1)-interp(VU, t+.1)) - (interp(VD, t-.1)-interp(VU, t-.1)))/.2
        cmd_rel = rotate(cmd[5:7]-dog[5:7], yaw)
        row = dict(time=t, elapsed=t-start, clock=clock(t), mode=mode,
                   truth_xy=float(np.linalg.norm(true_rel[:2])), truth_z=float(true_rel[2]),
                   est_xy=float(np.linalg.norm(dog[2:4]-uav[2:4])),
                   truth_rel_speed=float(np.linalg.norm(true_vel[:2])),
                   odom_age=t-uav[0], dog_age=t-dog[0], vision_age=t-previous(VISION,t)[0],
                   dog_speed=float(np.linalg.norm(dog[5:7])),
                   cmd_speed=float(np.linalg.norm(cmd[5:7])), model_residual=float(model_residual),
                   pos_ready=dog[9], yaw_ready=dog[10])
        for k, axis in enumerate(('x', 'y')):
            row.update({f'error_{axis}':err_body[k], f'I_{axis}':used_i[k], f'Ivel_{axis}':i_term[k],
                        f'Pvel_{axis}':.3*err_body[k], f'cmd_relative_{axis}':cmd_rel[k],
                        f'update_error_{axis}':update_error[k]})
        rows.append(row)
        if mode in (0, -2):
            world_i = np.clip(world_i + step*(cmd[2:4]-uav[2:4]), -limits, limits)
        else:
            body_i = np.clip(body_i + step*update_error, -limits, limits)
    data = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != 'clock'}
    times = data['time']
    weights = np.r_[np.diff(times), .015]
    def metrics(mask):
        out = {'seconds':float(weights[mask].sum()), 'samples':int(mask.sum())}
        if not mask.any():
            return out
        for key in ('truth_xy', 'est_xy', 'truth_rel_speed', 'dog_speed', 'model_residual', 'odom_age', 'vision_age'):
            v = data[key][mask]
            if np.isfinite(v).any():
                out[key] = dict(p50=float(np.nanpercentile(v, 50)), p95=float(np.nanpercentile(v, 95)), max=float(np.nanmax(v)))
        out['truth_xy']['rmse'] = float(np.sqrt(np.mean(data['truth_xy'][mask]**2)))
        for k, axis in enumerate(('x','y')):
            integ, err = data['I_'+axis], data['error_'+axis]
            out['integral_'+axis] = dict(min=float(integ[mask].min()), max=float(integ[mask].max()),
                saturated_seconds=float(weights[mask & (np.abs(integ)>=.99*limits[k])].sum()),
                opposes_P_seconds=float(weights[mask & (integ*err < 0) & (np.abs(data['Ivel_'+axis])>.05)].sum()),
                Ivel_abs_p95=float(np.percentile(np.abs(data['Ivel_'+axis][mask]),95)))
        return out
    first_precise = next(t for t, m in modes if m == 1)
    descend = next((t for t, m in modes if m == 2), end)
    stop = next((t for t, m in modes if m == -1 and t>first_precise), end)
    masks = {'mpc':data['mode']==0, 'precise_all':data['mode']==1,
             'precise_first_3s':(times>=first_precise)&(times<first_precise+3),
             'precise_last_10s':(times>=max(first_precise,descend-10))&(times<descend),
             'descent':data['mode']==2}
    summary = dict(run=str(args.run), start=clock(start), duration=end-start,
                   parameters=dict(ki=gain.tolist(), integral_limits=limits.tolist(), dt=step),
                   caveat='Integral reconstructed from recorded feedback and final commands; callback timing is approximate. Parameters from source snapshot, not binary proof. Mode 0 axes are world, 1/2 dog. Mode 0 P/relative velocity uses dog target for context, not the unlogged MPC trajectory. Vicon used only for evaluation, no alignment fit.',
                   modes=[dict(time=t, clock=clock(t), mode=m) for t,m in modes],
                   phases={k:metrics(v) for k,v in masks.items()})
    events = {}
    for label, mask in masks.items():
        if not mask.any(): continue
        indices = np.where(mask)[0]
        for what, index in [('max_error',indices[np.argmax(data['truth_xy'][mask])]), ('last',indices[-1])]:
            events[label+'_'+what] = rows[index]
    summary['events'] = events
    valid = np.array(diagnostics)
    vm = (valid[:,0]>=first_precise)&(valid[:,0]<=stop)
    summary['vicon_invalid_follow_frames'] = int(np.sum((valid[vm,1]==0)|(valid[vm,2]==0)))
    od = tables[ODOM][:,0]
    gaps = [(a,b,b-a) for a,b in zip(od[:-1],od[1:]) if b>first_precise and a<stop and b-a>.05]
    summary['odom_gaps_over_50ms'] = gaps
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    with (args.output/'integral_timeline.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    fig, axes = plt.subplots(4,1,figsize=(13,12),sharex=True)
    x=times-first_precise
    axes[0].plot(x,data['truth_xy'],label='Vicon center distance')
    axes[0].plot(x,data['est_xy'],label='Estimated center distance',alpha=.8)
    axes[0].set_ylabel('Horizontal distance (m)');axes[0].legend()
    for ax,axis,k in [(axes[1],'x',0),(axes[2],'y',1)]:
        ax.plot(x,data['Ivel_'+axis],label='Reconstructed I velocity')
        ax.plot(x,data['Pvel_'+axis],label='P velocity')
        ax.plot(x,data['cmd_relative_'+axis],label='Final command - dog velocity',alpha=.6)
        ax.axhline(gain[k]*limits[k],color='grey',ls=':')
        ax.axhline(-gain[k]*limits[k],color='grey',ls=':')
        ax.set_ylabel('Dog '+axis+' velocity (m/s)');ax.legend(loc='upper right')
    axes[3].plot(x,data['truth_rel_speed'],label='Vicon relative XY speed (0.2s derivative)')
    axes[3].plot(x,data['dog_speed'],label='Estimated dog speed')
    axes[3].set_ylabel('Speed (m/s)');axes[3].legend()
    for ax in axes:
        ax.set_xlim(-.1, stop-first_precise+.1)
        for t,m in modes:
            ax.axvline(t-first_precise,color='black',lw=.7,alpha=.4)
        for a,b in zip(tables[VISION][:-1,0],tables[VISION][1:,0]):
            if b-a>=.5: ax.axvspan(a-first_precise,b-first_precise,color='orange',alpha=.12)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Seconds from mode 1 ('+clock(first_precise)+' CST); orange = ArUco gap >=0.5s')
    fig.suptitle(args.run.name+'; Ki='+str(gain.tolist())+'; integral is reconstructed')
    fig.tight_layout();fig.savefig(args.output/'integral_overview.png',dpi=145);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='events'},ensure_ascii=False,indent=2))
    print('KEY EVENTS',json.dumps(events,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
