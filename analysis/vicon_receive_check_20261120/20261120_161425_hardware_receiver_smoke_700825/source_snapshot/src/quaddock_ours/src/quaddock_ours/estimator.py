"""Self-contained double-precision Ours-SW3 estimator.

The ROS adapter supplies matched, causally available odometry with every
relative observation. This module has no ROS, evaluation, or truth imports.
"""
import copy
import math
import time
import numpy as np

from .models import (alignment_batch_normal, alignment_measurement, alignment_relative, endpoint_matrix,
                     integrated_diffusion, interpolation_matrix, measurement_covariance,
                     observation, physical_velocity, prepare_alignment_factors, psd, robust_weight, rz,
                     sensor_prediction, solve_spd, symmetric, uav_rotation, wrap)



DEFAULTS = dict(window_s=2.0, process_scale=1.0, yaw_process_ratio=1.0,
    yaw_curvature_scale=1.0, translation_curvature_scale=1.0,
    yaw_bound_confidence=1.0, translation_bound_confidence=1.0,
    initial_yaw_sigma_deg=10.0, initial_translation_sigma_m=0.20,
    initial_yaw_rate_sigma_degps=0.5, initial_translation_rate_sigma_mps=0.03,
    nominal_yaw_rate_diffusion_degps_per_sqrt_s=0.05,
    nominal_translation_rate_diffusion_mps_per_sqrt_s=0.003,
    soft_bound_yaw_rate_degps=0.5, soft_bound_translation_rate_mps=0.03,
    soft_bound_yaw_sigma_degps=0.2, soft_bound_translation_sigma_mps=0.01,
    huber_transition_whitened_norm=2.5, max_solver_iterations=8,
    update_budget_ms=40.0, dog_velocity_frame='local',
    maximum_relative_age_s=0.05)


def _initial_covariance(config, initialization):
    if initialization.get('covariance') is not None:
        p = np.asarray(initialization['covariance'], float)
        if p.shape != (8, 8):
            raise ValueError('initial alignment covariance must be 8x8')
        return psd(p)
    sig = [math.radians(config['initial_yaw_sigma_deg'])] + [config['initial_translation_sigma_m']]*3
    sig += [math.radians(config['initial_yaw_rate_sigma_degps'])] + [config['initial_translation_rate_sigma_mps']]*3
    return np.diag(np.square(sig))


def _sigma(config):
    return np.array([math.radians(config['nominal_yaw_rate_diffusion_degps_per_sqrt_s']) *
                     config.get('yaw_process_ratio', 1.0)] +
                    [config['nominal_translation_rate_diffusion_mps_per_sqrt_s']]*3) * config['process_scale']


def _bounds(config):
    return (np.array([math.radians(config['soft_bound_yaw_rate_degps'])] +
                     [config['soft_bound_translation_rate_mps']]*3),
            np.array([math.radians(config['soft_bound_yaw_sigma_degps'])] +
                     [config['soft_bound_translation_sigma_mps']]*3))


def _bound_confidence(config):
    return np.array([config['yaw_bound_confidence']] +
                    [config['translation_bound_confidence']]*3, dtype=float)


def _kalman_update(x, p, innovation, jac, covariance, transition):
    # Same whitened *measurement* residual used by window factors, so robust
    # weighting does not silently differ because one backend has larger P.
    weight = robust_weight(innovation, covariance, transition)
    r = covariance / max(weight, 1e-12)
    gain = solve_spd(jac @ p @ jac.T + r, jac @ p).T
    xn = x + gain @ innovation
    a = np.eye(len(x)) - gain @ jac
    pn = symmetric(a @ p @ a.T + gain @ r @ gain.T)  # Joseph form
    return xn, pn

class _Base:
    def reset(self, config, initialization):
        self.config = dict(DEFAULTS)
        self.config.update(config or {})
        if self.config['window_s'] <= 0:
            raise ValueError('window_s must be positive')
        for key in ('yaw_curvature_scale', 'translation_curvature_scale'):
            if not np.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(key + ' must be finite and positive')
        for key in ('yaw_bound_confidence', 'translation_bound_confidence'):
            if not np.isfinite(self.config[key]) or self.config[key] < 0:
                raise ValueError(key + ' must be finite and nonnegative')
        self.initialization = copy.deepcopy(initialization or {})
        self.time = float(self.initialization.get('reference_time', 0.0))
        self.output_time = self.time
        self.q_reference = self.initialization.get('q_reference')
        self.u_reference = self.initialization.get('u_reference')
        self.seen = set()
        self.odometry = {}
        self.latest = {}
        self.solve_ms = 0.0
        self.update_count = 0
        self.status = 'waiting_for_odom'
        self.diagnostics = dict(duplicate=0, missing_odom=0, late=0,
                                deadline_misses=0, updates=0)
        self._reset_state()
        return self

    def ingest(self, event):
        sensor = event['sensor']
        if sensor not in ('uav_odom', 'dog_odom', 'vision', 'uwb',
                          'uwb_range', 'uwb_bearing'):
            raise ValueError('Unknown sensor: ' + str(sensor))
        identity = (sensor, str(event['measurement_id']))
        if identity in self.seen:
            self.diagnostics['duplicate'] += 1
            return {'accepted': False, 'reason': 'duplicate'}
        self.seen.add(identity)
        stamp = float(event['sample_time'])
        arrival = float(event.get('processing_time', event['arrival_time']))
        self.output_time = max(self.output_time, arrival)
        if sensor.endswith('_odom'):
            if sensor not in self.odometry or stamp >= self.odometry[sensor][0]:
                self.odometry[sensor] = (stamp, copy.deepcopy(event['value']))
                self.latest[sensor] = stamp
            if sensor == 'uav_odom' and self.u_reference is None:
                self.u_reference = list(event['value']['position'])
            if sensor == 'dog_odom' and self.q_reference is None:
                self.q_reference = list(event['value']['position'])
            return {'accepted': True, 'reason': 'odometry'}
        if not event.get('odom'):
            # Exact-time standalone callers are supported; never clamp a late
            # sample to an unrelated latest odometry pose.
            if (len(self.odometry) == 2 and all(abs(self.odometry[k][0]-stamp) < 1e-8
                                                for k in ('uav_odom', 'dog_odom'))):
                event = dict(event, odom={'uav': self.odometry['uav_odom'][1],
                                         'dog': self.odometry['dog_odom'][1]})
            else:
                self.diagnostics['missing_odom'] += 1
                return {'accepted': False, 'reason': 'missing_matched_odometry'}
        if self.q_reference is None or self.u_reference is None:
            self.diagnostics['missing_odom'] += 1
            return {'accepted': False, 'reason': 'missing_reference'}
        if stamp < self.time - max(self.config['window_s'], self.config['maximum_relative_age_s']):
            self.diagnostics['late'] += 1
            return {'accepted': False, 'reason': 'outside_history'}
        start = time.perf_counter()
        self._update(event)
        self.solve_ms = (time.perf_counter() - start) * 1000.0
        self.latest[sensor] = max(stamp, self.latest.get(sensor, -np.inf))
        self.update_count += 1
        self.diagnostics['updates'] += 1
        if self.solve_ms > self.config['update_budget_ms']:
            self.diagnostics['deadline_misses'] += 1
        self.status = 'ready' if len(self.odometry) == 2 else 'waiting_for_odom'
        return {'accepted': True, 'reason': 'updated', 'solve_ms': self.solve_ms}

    def _odometry_at(self, now):
        if len(self.odometry) < 2:
            return None, None
        values = {}
        for sensor, (stamp, value) in self.odometry.items():
            value = copy.deepcopy(value)
            dt = max(0.0, float(now) - stamp)
            velocity = np.asarray(value.get('velocity', [0, 0, 0]), float)
            if sensor == 'dog_odom' and self.config['dog_velocity_frame'] == 'body':
                velocity = rz(float(value['yaw'])) @ velocity
            value['position'] = (np.asarray(value['position'], float) + velocity * dt).tolist()
            values[sensor] = value
        return values['dog_odom'], values['uav_odom']

    def _common(self, now):
        return dict(sample_time=float(now), solve_ms=float(self.solve_ms),
                    status=self.status,
                    sensor_ages={sensor: max(0.0, float(now)-self.latest[sensor])
                                 if sensor in self.latest else None
                                 for sensor in ('vision','uwb','dog_odom','uav_odom')},
                    latest_sample_times=dict(self.latest),
                    q_reference=None if self.q_reference is None else list(self.q_reference),
                    u_reference=None if self.u_reference is None else list(self.u_reference),
                    diagnostics=dict(self.diagnostics))

    def predict(self, now):
        now = float(now)
        if now < self.time - 1e-9:
            raise ValueError('predict cannot query before the latest completed update')
        self.output_time = now
        return self.snapshot()

    def snapshot(self):
        now = max(self.output_time, self.time)
        result = self._common(now)
        a, rate, covariance = self._alignment_at(now)
        dog, uav = self._odometry_at(now)
        if dog is None or self.q_reference is None or self.u_reference is None:
            result.update(r=None, position=None, velocity=None, yaw=None,
                          transported_odometry_velocity=None,
                          velocity_semantics='analytic_mapped_world_derivative',
                          psi=float(wrap(a[0])), b=a[1:4].tolist(),
                          rate=rate.tolist(), covariance=covariance.tolist(),
                          status='waiting_for_odom')
            return result
        r, unused = alignment_relative(a, dog, uav, self.q_reference, self.u_reference)
        result.update(r=r.tolist(), position=(np.asarray(uav['position'])+r).tolist(),
                      velocity=physical_velocity(a[0],dog,self.config['dog_velocity_frame'],
                                                 rate,self.q_reference).tolist(),
                      transported_odometry_velocity=physical_velocity(a[0],dog,
                                                 self.config['dog_velocity_frame']).tolist(),
                      velocity_semantics='analytic_mapped_world_derivative',
                      yaw=float(wrap(float(dog['yaw'])+a[0])), psi=float(wrap(a[0])),
                      b=a[1:4].tolist(), rate=rate.tolist(), covariance=covariance.tolist())
        return result


class AlignEKF8(_Base):
    def _reset_state(self):
        initial = self.initialization
        self.x = np.r_[float(initial.get('psi', 0.0)), initial.get('b', [0,0,0]),
                       initial.get('rate', [0,0,0,0])].astype(float)
        self.p = _initial_covariance(self.config, initial)
        self.sigma = _sigma(self.config)

    def _update(self, event):
        # Delayed factors are evaluated at their own sampling time. For delays
        # shorter than the interpolation horizon this is a constant-rate
        # conditional-state approximation; released historical outputs stay fixed.
        target = max(self.time, float(event['sample_time']))
        dt = target-self.time
        f, q = integrated_diffusion(dt, self.sigma)
        self.x = f @ self.x
        self.p = symmetric(f @ self.p @ f.T + q)
        self.time = target
        if dt > 0:
            bound, sig = _bounds(self.config)
            confidence = _bound_confidence(self.config)
            active = (np.abs(self.x[4:]) > bound) & (confidence > 0)
            if np.any(active):
                indices = np.where(active)[0]
                jac = np.zeros((len(indices), 8))
                jac[np.arange(len(indices)), 4+indices] = 1.0
                residual = np.sign(self.x[4+indices])*bound[indices] - self.x[4+indices]
                # Integral penalty: same seconds weighting as the window prior.
                self.x, self.p = _kalman_update(self.x, self.p, residual, jac,
                    np.diag(sig[indices]**2/(dt*confidence[indices])), float('inf'))
        lag = float(event['sample_time'])-self.time
        sample_map = np.column_stack((np.eye(4), lag*np.eye(4)))
        sample = sample_map @ self.x
        residual, jac, covariance = alignment_measurement(sample, event,
            self.q_reference, self.u_reference, self.config)
        # Unmodelled within-delay rate diffusion, never negative covariance.
        covariance = covariance + jac @ np.diag(self.sigma**2*abs(lag)**3/3) @ jac.T
        self.x, self.p = _kalman_update(self.x, self.p, residual, jac @ sample_map,
            covariance, self.config['huber_transition_whitened_norm'])

    def _alignment_at(self, now):
        f, q = integrated_diffusion(max(0.0, now-self.time), self.sigma)
        x = f @ self.x
        return x[:4], x[4:], symmetric(f @ self.p @ f.T + q)


class SlidingWindow3(_Base):
    """Ours-SW3: three 4D nodes with a retained history prior.

    The posterior is NEVER fed back as an independent new prior. Only leaving
    observation factors are absorbed. Grid transport is a documented Gaussian
    piecewise-linear projection, including its full joint covariance.
    """
    method = 'ours_sw3'

    def _reset_state(self):
        self.sigma = _sigma(self.config)
        w = float(self.config['window_s'])
        self.node_times = np.array([self.time-w, self.time-w/2, self.time])
        initial = np.r_[float(self.initialization.get('psi', 0.0)),
                         self.initialization.get('b', [0,0,0]),
                         self.initialization.get('rate', [0,0,0,0])].astype(float)
        mapping = np.vstack([np.column_stack((np.eye(4), (t-self.time)*np.eye(4)))
                             for t in self.node_times])
        self.prior_mean = mapping @ initial
        self.prior_covariance = mapping @ _initial_covariance(self.config, self.initialization) @ mapping.T
        # Curvature mode is independent of endpoint alignment/rate. It makes
        # the 12D seed full rank without inventing repeated observations.
        self.prior_covariance[:4,:4] += np.diag(self.sigma**2*w**3 + 1e-10)
        self.prior_covariance = psd(self.prior_covariance)
        self.nodes = self.prior_mean.copy()
        self.node_covariance = self.prior_covariance.copy()
        self.factors = []
        self.marginalized_factor_ids = []
        self.prior_linearization_point = self.nodes.copy()
        self.last_relative_time = None
        self.iterations = 0
        half=w/2
        self._first_derivative=np.zeros((4,12));self._last_derivative=np.zeros((4,12))
        self._first_derivative[:,:4]=-np.eye(4)/half
        self._first_derivative[:,4:8]=np.eye(4)/half
        self._last_derivative[:,4:8]=-np.eye(4)/half
        self._last_derivative[:,8:]=np.eye(4)/half
        curvature=self._last_derivative-self._first_derivative
        curvature_scale=np.array([self.config['yaw_curvature_scale']] +
                                 [self.config['translation_curvature_scale']]*3)
        variance=np.maximum(self.sigma**2*(2*half)/3*curvature_scale**2,1e-14)
        self._curvature_information=curvature.T @ (curvature/variance[:,None])
        self._bound,self._bound_sigma=_bounds(self.config)
        self._bound_confidence=_bound_confidence(self.config)

    def _factor_normal(self, nodes, event, times=None):
        times = self.node_times if times is None else times
        mapping = interpolation_matrix(times, float(event['sample_time']))
        residual, jac, covariance = alignment_measurement(mapping @ nodes,
            event, self.q_reference, self.u_reference, self.config)
        h = jac @ mapping
        weight = robust_weight(residual, covariance, self.config['huber_transition_whitened_norm'])
        precision_h = solve_spd(covariance, h) * weight
        return h.T @ precision_h, h.T @ solve_spd(covariance, residual) * weight

    def _advance_grid(self, target):
        if target <= self.time+1e-12:
            return
        # Large gaps are split to keep the deterministic remeshing map full
        # rank. No new information is introduced by subdivision.
        half = self.config['window_s']/2
        while target > self.time+1e-12:
            next_time = min(target, self.time + half/2)
            new_times = self.node_times + (next_time-self.time)
            departing = [event for event in self.factors if event['sample_time'] < new_times[0]]
            if departing:
                j = solve_spd(self.prior_covariance, np.eye(12))
                g = j @ (self.prior_mean-self.nodes)
                for event in departing:
                    jf, gf = self._factor_normal(self.nodes, event)
                    j += jf
                    g += gf
                    self.marginalized_factor_ids.append(str(event['measurement_id']))
                self.prior_mean = self.nodes + solve_spd(j, g)
                self.prior_covariance = solve_spd(j, np.eye(12))
                leaving_ids = {id(event) for event in departing}
                self.factors = [event for event in self.factors if id(event) not in leaving_ids]
            # Gaussian transition y=T*x+epsilon, marginalized over all x.
            # This covariance form is equivalent to a joint Schur complement
            # but remains stable for the rank-four stochastic augmentation.
            transport = np.vstack([interpolation_matrix(self.node_times, t) for t in new_times])
            dt = next_time-self.time
            noise = np.zeros((12,12))
            noise[8:,8:] = np.diag(self.sigma**2 * (dt*half**2 + dt**3/3))
            self.prior_mean = transport @ self.prior_mean
            self.prior_covariance = psd(transport @ self.prior_covariance @ transport.T + noise)
            self.nodes = transport @ self.nodes
            self.node_covariance = psd(transport @ self.node_covariance @ transport.T + noise)
            self.node_times = new_times
            self.time = next_time
            self.prior_linearization_point = self.prior_mean.copy()

    def _regularization(self, nodes):
        j=self._curvature_information.copy()
        g=-j @ nodes
        bound,sig=self._bound,self._bound_sigma
        width=self.config['window_s']/2
        for derivative in (self._first_derivative,self._last_derivative):
            rate = derivative @ nodes
            active = (np.abs(rate)>bound) & (self._bound_confidence>0)
            if np.any(active):
                d = derivative[active]
                residual = np.sign(rate[active])*bound[active]-rate[active]
                precision = width*self._bound_confidence[active]/sig[active]**2
                j += d.T @ (precision[:,None]*d)
                g += d.T @ (precision*residual)
        return j,g

    def _normal(self, nodes, prepared=None, prior_precision=None):
        j = (solve_spd(self.prior_covariance,np.eye(12)) if prior_precision is None
             else prior_precision.copy())
        g = j @ (self.prior_mean-nodes)
        jf,gf=alignment_batch_normal(nodes,self.factors,self.node_times,
                                    self.q_reference,self.u_reference,self.config,prepared)
        j+=jf;g+=gf
        jr,gr = self._regularization(nodes)
        return symmetric(j+jr),g+gr

    def _solve(self):
        """Repeatable solve: a solve without new factors cannot add information."""
        started = time.perf_counter()
        nodes = self.nodes.copy()
        prepared=prepare_alignment_factors(self.factors,self.node_times,self.q_reference,
                                          self.u_reference,self.config)
        prior_precision=solve_spd(self.prior_covariance,np.eye(12))
        self.iterations = 0
        for iteration in range(int(self.config['max_solver_iterations'])):
            j,g = self._normal(nodes,prepared,prior_precision)
            delta = solve_spd(j,g)
            nodes += delta
            self.iterations = iteration+1
            if np.linalg.norm(delta) < 1e-8:
                break
            if (time.perf_counter()-started)*1000 >= self.config['update_budget_ms']:
                break
        j,unused = self._normal(nodes,prepared,prior_precision)
        self.nodes = nodes
        self.node_covariance = psd(solve_spd(j,np.eye(12)))

    def _update(self,event):
        self._advance_grid(max(self.time,float(event['sample_time'])))
        self.factors.append(copy.deepcopy(event))
        self._solve()
        self.last_relative_time = max(float(event['sample_time']), self.last_relative_time or -np.inf)

    def _alignment_at(self, now):
        when = float(now)
        mapping = endpoint_matrix(self.node_times, when)
        joint = mapping @ self.nodes
        covariance = mapping @ self.node_covariance @ mapping.T
        unused,q = integrated_diffusion(max(0.0,when-self.time),self.sigma)
        covariance += q
        return joint[:4],joint[4:],psd(covariance)

    def snapshot(self):
        result = super().snapshot()
        result.update(node_times=self.node_times.tolist(),
                      node_linearization_point=self.nodes.reshape(3,4).tolist(),
                      prior_linearization_point=self.prior_linearization_point.reshape(3,4).tolist(),
                      retained_factors=len(self.factors),
                      marginalized_factors=len(self.marginalized_factor_ids),
                      solver_iterations=self.iterations,
                      approximation='piecewise_linear_Gaussian_grid_transport')
        return result


def make_estimator(config=None, initialization=None):
    """Construct the only estimator exported by the real-hardware package."""
    return SlidingWindow3().reset(config, initialization)
