"""Shared gravity-aligned geometry and small dense Gaussian operations.

All angles are radians, all arrays are float64.  Dog velocity in ``local``
means the axes of its local odometry, rather than its body heading.
"""
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def drz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])


def uav_rotation(uav):
    """UAV body-to-world rotation; real roll/pitch must not be discarded."""
    if uav.get('quaternion_xyzw') is None:
        return rz(float(uav['yaw']))
    q=np.asarray(uav['quaternion_xyzw'],float)
    norm=float(q @ q)
    if q.shape!=(4,) or norm<1e-12 or not np.isfinite(norm):
        raise ValueError('invalid UAV quaternion_xyzw')
    x,y,z,w=q/math.sqrt(norm)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def symmetric(a):
    return (np.asarray(a, dtype=float) + np.asarray(a, dtype=float).T) * 0.5


def psd(a, floor=1e-14):
    a = symmetric(a)
    vals, vecs = np.linalg.eigh(a)
    return symmetric((vecs * np.maximum(vals, floor)) @ vecs.T)


def solve_spd(a, b):
    a = symmetric(a)
    try:
        return cho_solve(cho_factor(a, lower=True, check_finite=False), b,
                         check_finite=False)
    except np.linalg.LinAlgError:
        return np.linalg.solve(psd(a, 1e-10), b)


def marginalize_gaussian(information, information_vector, keep):
    """Exact Schur elimination in canonical coordinates (no diagonalization).

    Returns the marginal information matrix/vector on ``keep`` in that order.
    Useful independently to verify fixed-lag prior maintenance.
    """
    j = np.asarray(information, dtype=float)
    h = np.asarray(information_vector, dtype=float)
    keep = np.asarray(keep, dtype=int)
    leave = np.asarray([i for i in range(len(h)) if i not in set(keep)], dtype=int)
    if not len(leave):
        return j[np.ix_(keep, keep)].copy(), h[keep].copy()
    jkl = j[np.ix_(keep, leave)]
    jll = j[np.ix_(leave, leave)]
    return (symmetric(j[np.ix_(keep, keep)] - jkl @ solve_spd(jll, jkl.T)),
            h[keep] - jkl @ solve_spd(jll, h[leave]))


def robust_weight(residual, covariance, transition=2.5):
    """One Huber weight for the norm of a complete whitened sensor residual."""
    residual = np.asarray(residual, dtype=float)
    norm = math.sqrt(max(0.0, float(residual @ solve_spd(covariance, residual))))
    return min(1.0, float(transition) / max(norm, 1e-15))


def alignment_relative(a, dog, uav, q_reference, u_reference):
    """r = b + R(psi)(q-q_s) - (u-u_s), Jacobian wrt [psi,b]."""
    dq = np.asarray(dog['position'], float) - np.asarray(q_reference, float)
    du = np.asarray(uav['position'], float) - np.asarray(u_reference, float)
    r = np.asarray(a[1:4]) + rz(float(a[0])) @ dq - du
    jac = np.column_stack((drz(float(a[0])) @ dq, np.eye(3)))
    return r, jac


def physical_velocity(psi, dog, frame='local', alignment_rate=None, q_reference=None):
    """Mapped target world velocity without differencing correction jumps.

    The backward-compatible three-argument call returns transported odometry
    velocity only. With an estimated alignment rate, differentiate the mapping
    analytically: R*qdot + psidot*R'*(q-qs) + bdot. In the protocol, qdot itself
    contains local-frame drift, so the rate terms are needed for physical speed.
    """
    velocity = np.asarray(dog.get('velocity', [0.0, 0.0, 0.0]), float)
    if frame == 'body':
        velocity = rz(float(dog['yaw'])) @ velocity
    elif frame not in ('local', 'world'):
        raise ValueError('dog_velocity_frame must be local, world or body')
    mapped=rz(float(psi)) @ velocity
    if alignment_rate is not None:
        if q_reference is None:
            raise ValueError('q_reference required for analytic mapped velocity')
        rate=np.asarray(alignment_rate,float)
        dq=np.asarray(dog['position'],float)-np.asarray(q_reference,float)
        mapped += rate[0]*(drz(float(psi)) @ dq)+rate[1:4]
    return mapped


def sensor_prediction(sensor, r, body_yaw, uav, config):
    """Measurement and Jacobian wrt [relative world position, body yaw].

    Lever arms are metres in the relevant body frame: camera/receiver on UAV,
    vision board/UWB tag on dog. Vision positions are expressed in UAV body
    axes (camera rotation must already have been calibrated by the adapter).
    UWB bearing is in UAV body axes by default; set uwb_bearing_frame='world'
    for an already transformed frontend. Vision yaw is a *body* yaw.
    """
    u_yaw = float(uav['yaw'])
    u_rotation = uav_rotation(uav)
    rot = u_rotation.T
    if sensor == 'vision':
        dog_arm = np.asarray(config.get('vision_dog_lever_arm_m', [0, 0, 0]), float)
        uav_arm = np.asarray(config.get('vision_uav_lever_arm_m', [0, 0, 0]), float)
    elif sensor in ('uwb', 'uwb_range', 'uwb_bearing'):
        dog_arm = np.asarray(config.get('uwb_dog_lever_arm_m', [0, 0, 0]), float)
        uav_arm = np.asarray(config.get('uwb_uav_lever_arm_m', [0, 0, 0]), float)
    else:
        raise ValueError('unknown relative sensor: ' + str(sensor))
    delta = np.asarray(r) + rz(body_yaw) @ dog_arm - u_rotation @ uav_arm
    arm_jac = drz(body_yaw) @ dog_arm
    if sensor == 'vision':
        h = np.r_[rot @ delta, float(wrap(body_yaw - u_yaw))]
        jac = np.zeros((4, 4))
        jac[:3, :3] = rot
        jac[:3, 3] = rot @ arm_jac
        jac[3, 3] = 1.0
        return h, jac, [3]
    bearing_rot = np.eye(3) if config.get('uwb_bearing_frame') == 'world' else rot
    d = bearing_rot @ delta
    norm = max(float(np.linalg.norm(d)), 1e-12)
    horizontal_sq = max(float(d[0]**2 + d[1]**2), 1e-12)
    jd = np.array([d / norm, [-d[1] / horizontal_sq, d[0] / horizontal_sq, 0]])
    jposition = jd @ bearing_rot
    if sensor == 'uwb_range':
        # Alignment state is [psi, bx, by, bz].  The range row therefore has
        # three position derivatives plus one yaw derivative from the tag arm.
        jac = np.r_[jposition[0], float(jposition[0] @ arm_jac)]
        return np.array([norm]), jac.reshape(1, 4), []
    if sensor == 'uwb_bearing':
        jac = np.r_[jposition[1], float(jposition[1] @ arm_jac)]
        return np.array([math.atan2(d[1], d[0])]), jac.reshape(1, 4), [0]
    return np.array([norm, math.atan2(d[1], d[0])]), np.column_stack(
        (jposition, jposition @ arm_jac)), [1]


def observation(event):
    value = event['value']
    if event['sensor'] == 'vision':
        if value.get('yaw') is None:
            return np.asarray(value['position'], float), []
        return np.r_[np.asarray(value['position'], float), float(value['yaw'])], [3]
    if event['sensor'] == 'uwb_range':
        return np.array([float(value['range'])]), []
    if event['sensor'] == 'uwb_bearing':
        return np.array([float(value['bearing'])]), [0]
    return np.array([float(value['range']), float(value['bearing'])]), [1]


def measurement_covariance(event, config, size=None):
    sensor = event['sensor']
    if sensor == 'vision':
        sig = [config.get('vision_position_sigma_m', 0.02)] * 3 + [
            math.radians(config.get('vision_yaw_sigma_deg', 2.0))]
    elif sensor == 'uwb_range':
        sig = [config.get('uwb_range_sigma_m', 0.05)]
    elif sensor == 'uwb_bearing':
        sig = [math.radians(config.get('uwb_bearing_sigma_deg', 3.0))]
    else:
        sig = [config.get('uwb_range_sigma_m', 0.05),
               math.radians(config.get('uwb_bearing_sigma_deg', 3.0))]
    covariance = event.get('covariance')
    cov = np.diag(np.square(sig)) if covariance is None else np.asarray(covariance, float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    if size is not None:
        cov = cov[:size, :size]
    return psd(cov, 1e-12)


def alignment_measurement(a, event, q_reference, u_reference, config):
    """Return innovation, full alignment Jacobian, shared noise covariance."""
    dog, uav = event['odom']['dog'], event['odom']['uav']
    r, jr = alignment_relative(a, dog, uav, q_reference, u_reference)
    body_yaw = float(dog['yaw']) + float(a[0])
    pred, js, angles = sensor_prediction(event['sensor'], r, body_yaw, uav, config)
    observed, unused = observation(event)
    n = len(observed)
    pred, js = pred[:n], js[:n]
    inn = observed - pred
    for k in angles:
        if k < n:
            inn[k] = wrap(inn[k])
    jstate = np.zeros((4, 4))
    jstate[:3] = jr
    jstate[3, 0] = 1.0
    jac = js @ jstate
    cov = measurement_covariance(event, config, n)
    # Shared odometry uncertainty, not fabricated independent position truth.
    odom_var = (config.get('dog_odom_position_sigma_m', 0.005)**2 +
                config.get('uav_odom_position_sigma_m', 0.005)**2)
    cov += odom_var * (js[:, :3] @ js[:, :3].T)
    return inn, jac, cov


def integrated_diffusion(dt, sigma):
    """8D [alignment, alignment rate] integrated white rate diffusion."""
    dt = max(0.0, float(dt))
    d = np.diag(np.square(np.asarray(sigma, float)))
    f = np.eye(8)
    f[:4, 4:] = dt * np.eye(4)
    q = np.block([[d * dt**3 / 3.0, d * dt**2 / 2.0],
                  [d * dt**2 / 2.0, d * dt]])
    return f, q


def interpolation_matrix(times, time):
    """Piecewise linear 4D interpolation, with endpoint rate extrapolation."""
    times = np.asarray(times, float)
    i = min(max(int(np.searchsorted(times, time, side='right')) - 1, 0), 1)
    fraction = (float(time) - times[i]) / (times[i + 1] - times[i])
    matrix = np.zeros((4, 12))
    matrix[:, 4*i:4*i+4] = (1 - fraction) * np.eye(4)
    matrix[:, 4*i+4:4*i+8] = fraction * np.eye(4)
    return matrix


def endpoint_matrix(times, now):
    """Joint endpoint alignment/rate map, preserving cross covariance."""
    dt = float(times[2] - times[1])
    derivative = np.zeros((4, 12))
    derivative[:, 4:8] = -np.eye(4) / dt
    derivative[:, 8:12] = np.eye(4) / dt
    return np.vstack((interpolation_matrix(times, now), derivative))


def prepare_alignment_factors(events,times,q_reference,u_reference,config,dimension=12):
    """Cache iteration-invariant geometry once per observation/grid update."""
    prepared=[]
    for sensor, size in (('vision',4),('vision',3),('uwb',2),
                         ('uwb_range',1),('uwb_bearing',1)):
        group = [e for e in events if e['sensor']==sensor and
                 (sensor!='vision' or (4 if e['value'].get('yaw') is not None else 3)==size)]
        if not group:
            continue
        for event in group:
            if '_geometry' not in event:
                dog,uav=event['odom']['dog'],event['odom']['uav']
                event['_geometry']=(np.asarray(dog['position'])-np.asarray(q_reference),
                    np.asarray(uav['position'])-np.asarray(u_reference),float(dog['yaw']),
                    float(uav['yaw']),observation(event)[0],measurement_covariance(event,config,size),
                    uav_rotation(uav))
        dq=np.array([e['_geometry'][0] for e in group])
        du=np.array([e['_geometry'][1] for e in group])
        dyaw=np.array([e['_geometry'][2] for e in group])
        uyaw=np.array([e['_geometry'][3] for e in group])
        observed=np.array([e['_geometry'][4] for e in group])
        covariance=np.array([e['_geometry'][5] for e in group])
        if dimension==12:
            times=np.asarray(times)
            ts=np.array([e['sample_time'] for e in group])
            index=(ts>=times[1]).astype(int)
            fraction=(ts-times[index])/(times[index+1]-times[index])
            weights=np.zeros((len(group),3))
            weights[np.arange(len(group)),index]=1-fraction
            weights[np.arange(len(group)),index+1]=fraction
            mapping=(weights[:,None,:,None]*np.eye(4)[None,:,None,:]).reshape(len(group),4,12)
        else:
            mapping=np.tile(np.eye(4),(len(group),1,1))
        arm_sensor = 'uwb' if sensor in ('uwb_range', 'uwb_bearing') else sensor
        dog_arm=np.asarray(config.get(arm_sensor+'_dog_lever_arm_m',[0,0,0]))
        uav_arm=np.asarray(config.get(arm_sensor+'_uav_lever_arm_m',[0,0,0]))
        u_rotation=np.array([e['_geometry'][6] for e in group])
        uav_lever=np.einsum('nij,j->ni',u_rotation,uav_arm)
        rotation=np.transpose(u_rotation,(0,2,1))
        if sensor in ('uwb', 'uwb_bearing') and config.get('uwb_bearing_frame')=='world':
            rotation=np.tile(np.eye(3),(len(group),1,1))
        odom_var=config.get('dog_odom_position_sigma_m',.005)**2+config.get('uav_odom_position_sigma_m',.005)**2
        fixed_precision=None
        if sensor=='vision':
            covariance[:,:3,:3]+=np.eye(3)[None,:,:]*odom_var
            fixed_precision=np.linalg.inv(covariance)
        prepared.append((sensor,size,dq,du,dyaw,uyaw,observed,covariance,mapping,dog_arm,uav_lever,rotation,fixed_precision))
    return prepared


def alignment_batch_normal(nodes, events, times, q_reference, u_reference, config,prepared=None):
    """Vectorized retained factors, exactly the scalar shared observation model.

    This avoids Python/SciPy call overhead for every retained observation in
    every nonlinear iteration. General (non-diagonal) covariance is supported.
    """
    dimension=len(nodes)
    information,gradient=np.zeros((dimension,dimension)),np.zeros(dimension)
    if prepared is None:
        prepared=prepare_alignment_factors(events,times,q_reference,u_reference,config,dimension)
    odom_var=config.get('dog_odom_position_sigma_m',.005)**2+config.get('uav_odom_position_sigma_m',.005)**2
    for sensor,size,dq,du,dyaw,uyaw,observed,covariance,mapping,dog_arm,uav_lever,rotation,fixed_precision in prepared:
        n=len(dq)
        a=np.einsum('nij,j->ni',mapping,nodes)
        psi=a[:,0]; c,s=np.cos(psi),np.sin(psi)
        rotated=np.column_stack((c*dq[:,0]-s*dq[:,1],s*dq[:,0]+c*dq[:,1],dq[:,2]))
        dpsi=np.column_stack((-rotated[:,1],rotated[:,0],np.zeros(n)))
        cy,sy=np.cos(psi+dyaw),np.sin(psi+dyaw)
        dog_lever=np.column_stack((cy*dog_arm[0]-sy*dog_arm[1],sy*dog_arm[0]+cy*dog_arm[1],
                                  np.full(n,dog_arm[2])))
        dpsi+=np.column_stack((-dog_lever[:,1],dog_lever[:,0],np.zeros(n)))
        delta=a[:,1:]+rotated-du+dog_lever-uav_lever
        body=np.einsum('nij,nj->ni',rotation,delta)
        ja=np.zeros((n,3,4))
        ja[:,:,0]=np.einsum('nij,nj->ni',rotation,dpsi)
        ja[:,:,1:]=rotation
        if sensor=='vision':
            predicted=np.column_stack((body,wrap(psi+dyaw-uyaw)))[:,:size]
            j=np.zeros((n,size,4)); j[:,:3]=ja
            if size==4:
                j[:,3,0]=1
            precision=fixed_precision.copy()
        elif sensor in ('uwb', 'uwb_range', 'uwb_bearing'):
            norm=np.maximum(np.linalg.norm(body,axis=1),1e-12)
            rho=np.maximum(body[:,0]**2+body[:,1]**2,1e-12)
            geometry=np.zeros((n,2,3));geometry[:,0]=body/norm[:,None]
            geometry[:,1,0]=-body[:,1]/rho;geometry[:,1,1]=body[:,0]/rho
            if sensor == 'uwb_range':
                predicted=norm[:,None]
                j=np.einsum('nij,njk->nik',geometry[:,0:1],ja)
            elif sensor == 'uwb_bearing':
                predicted=np.arctan2(body[:,1],body[:,0])[:,None]
                j=np.einsum('nij,njk->nik',geometry[:,1:2],ja)
            else:
                predicted=np.column_stack((norm,np.arctan2(body[:,1],body[:,0])))
                j=np.einsum('nij,njk->nik',geometry,ja)
            varying_covariance=covariance.copy()
            if sensor == 'uwb_range':
                varying_covariance[:,0,0]+=odom_var
            elif sensor == 'uwb_bearing':
                varying_covariance[:,0,0]+=odom_var/rho
            else:
                varying_covariance[:,0,0]+=odom_var;varying_covariance[:,1,1]+=odom_var/rho
            precision=np.linalg.inv(varying_covariance)
        residual=observed-predicted
        if sensor=='uwb':
            residual[:,1]=wrap(residual[:,1])
        elif sensor=='uwb_bearing':
            residual[:,0]=wrap(residual[:,0])
        elif size==4:
            residual[:,3]=wrap(residual[:,3])
        norms=np.sqrt(np.maximum(0,np.einsum('ni,nij,nj->n',residual,precision,residual)))
        weights=np.minimum(1,config.get('huber_transition_whitened_norm',2.5)/np.maximum(norms,1e-15))
        precision*=weights[:,None,None]
        h=np.einsum('nij,njk->nik',j,mapping)
        ph=np.einsum('nij,njk->nik',precision,h)
        information+=np.einsum('nki,nkj->ij',h,ph)
        gradient+=np.einsum('nki,nkl,nl->i',h,precision,residual)
    return information,gradient
