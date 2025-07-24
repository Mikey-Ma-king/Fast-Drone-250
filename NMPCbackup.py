#!/usr/bin/env python
# coding=UTF-8

import os
import sys
import shutil
import errno
import timeit

from drone_model.drone_model_s_drag_delay import DroneModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

import numpy as np
import scipy.linalg

def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print('Error while removing directory {}'.format(directory))

class DroneOptimizer(object):
    def __init__(self, d_model, d_constraint, t_horizon, n_nodes):
        model = d_model
        self.T = t_horizon
        self.N = n_nodes

        # Ensure current working directory is current folder
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.acados_models_dir =os.path.join(os.getcwd(), 'acados_models')
        self.save_dir = os.path.join(os.getcwd(), 'data')
        safe_mkdir_recursive(self.acados_models_dir)
        safe_mkdir_recursive(self.save_dir)
        acados_source_path = os.environ['ACADOS_SOURCE_DIR']
        sys.path.insert(0, acados_source_path)

        nx = model.x.size()[0]
        self.nx = nx
        nu = model.u.size()[0]
        self.nu = nu
        ny = nx + nu
        n_params = len(model.p)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + '/include'
        ocp.acados_lib_path = acados_source_path + '/lib'
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.T

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)

        # cost
        # Q = np.diag([200, 200, 500, 1, 1, 1, 5, 5, 5, 200])
        # R = np.diag([6, 30, 30, 30])
        # Q = np.diag([2000, 2000, 5000, 1, 1, 1, 5, 5, 5, 2000, 30, 30, 30, 0])
        # Q = np.diag([2000, 2000, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30])
        Q = np.diag([2000, 2000, 5000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # R = np.diag([0.5, 18, 18, 18])
        # R = np.diag([0.05, 18, 18, 18])
        # R = np.diag([30, 10, 10, 10])
        # R = np.diag([0, 10, 10, 10])
        # R = np.diag([0.05, 0.05, 0.05, 0.05])
        R = np.diag([0.5, 0.5, 0.5, 0.5])
        # R = np.diag([0.01, 0.05, 0.05, 0.05])
        # R = np.diag([0, 0, 0, 0])
        # R = np.diag([20, 20, 20, 20])
        # R = np.diag([5, 5, 5, 5])
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = scipy.linalg.block_diag(Q, R)
        ocp.cost.W_e = Q
        
        # q(w,x,y,z)
        # Vx
        ocp.cost.Vx = np.zeros((ny, nx))
        # ocp.cost.Vx = np.zeros((ny-1, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        # ocp.cost.Vx[:6, :6] = np.eye(6)
        # ocp.cost.Vx[6:nx-1, 7:10] = np.eye(3)
        ocp.cost.Vx_e = ocp.cost.Vx[:nx, :nx]
        # ocp.cost.Vx_e = ocp.cost.Vx[:nx-1, :nx]
        # Vu
        ocp.cost.Vu = np.zeros((ny, nu))
        # ocp.cost.Vu = np.zeros((ny-1, nu))
        ocp.cost.Vu[-nu:, -nu:] = np.eye(nu)

        # set constraints
        # ocp.constraints.lbx = np.array([-6,-6,-6])
        # ocp.constraints.ubx = np.array([6,6,6])
        # ocp.constraints.lbx = np.array([d_constraint.z_min])
        # ocp.constraints.ubx = np.array([d_constraint.z_max])
        # ocp.constraints.lbx = np.array([d_constraint.z_min, d_constraint.T_min])
        # ocp.constraints.ubx = np.array([d_constraint.z_max, d_constraint.T_max])
        ocp.constraints.lbx = np.concatenate((d_constraint.z_min, d_constraint.T_min, d_constraint.w_min))
        ocp.constraints.ubx = np.concatenate((d_constraint.z_max, d_constraint.T_max, d_constraint.w_max))
        # ocp.constraints.idxbx = np.array([0,1,2])
        # ocp.constraints.idxbx = np.array([2])
        # ocp.constraints.idxbx = np.array([2, 13])
        ocp.constraints.idxbx = np.array([2, 13, 14, 15, 16])
        # ocp.constraints.lbu = np.concatenate((np.array([d_constraint.T_min]), d_constraint.w_min))
        # ocp.constraints.ubu = np.concatenate((np.array([d_constraint.T_max]), d_constraint.w_max))
        # ocp.constraints.lbu = np.concatenate((np.array([d_constraint.dT_min]), d_constraint.w_min))
        # ocp.constraints.ubu = np.concatenate((np.array([d_constraint.dT_max]), d_constraint.w_max))
        ocp.constraints.lbu = np.concatenate((d_constraint.dT_min, d_constraint.dw_min))
        ocp.constraints.ubu = np.concatenate((d_constraint.dT_max, d_constraint.dw_max))
        ocp.constraints.idxbu = np.array(range(nu))
        # ocp.constraints.lh = np.array([d_constraint.psi_min])
        # ocp.constraints.uh = np.array([d_constraint.psi_max])
        # ocp.constraints.lh_e = np.array([d_constraint.psi_min])
        # ocp.constraints.uh_e = np.array([d_constraint.psi_max])
        # ocp.constraints.lh = np.concatenate((d_constraint.phi_min, d_constraint.theta_min))
        # ocp.constraints.uh = np.concatenate((d_constraint.phi_max, d_constraint.theta_max))
        # ocp.constraints.lh_e = np.concatenate((d_constraint.phi_min, d_constraint.theta_min))
        # ocp.constraints.uh_e = np.concatenate((d_constraint.phi_max, d_constraint.theta_max))

        # initial state
        x_init = np.zeros(nx)
        x_init[6] = 1
        ocp.constraints.x0 = x_init
        
        # initial ref
        u_ref = np.zeros(nu)        
        x_ref = np.zeros(nx)
        # x_ref = np.zeros(nx-1)
        ocp.cost.yref = np.concatenate((x_ref, u_ref))
        ocp.cost.yref_e = x_ref

        # solver options
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # explicit Runge-Kutta integrator
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # compile acados ocp
        json_file = os.path.join(self.acados_models_dir, model.name+'_acados_ocp.json')
        self.solver = AcadosOcpSolver(ocp, json_file=json_file)
        self.integrator = AcadosSimSolver(ocp, json_file=json_file)

    def simulation(self, d_model, T_max, a_max, v_max, n,drone_state,target_p_dis,target_v,target_yaw):
        p0, q0 = self.track_cal(0, target_p=target_p_dis, target_v=target_v, n=n,target_yaw=target_yaw)
        v0 = np.array([0, 0, 0])
        q0 = np.array([1, 0, 0, 0])
        # x0 = np.concatenate((p0, v0, q0, np.zeros(3)))
        print("drone_state",drone_state.shape)
        x0 = np.concatenate((drone_state, q0, np.zeros(3), np.array([d_model.m * d_model.g]), np.zeros(3)))
        print("x0",x0.shape)
        Nsim = int(T_max * self.N / self.T)
        Tsim = 0
        
        simX = np.zeros((Nsim+1, self.nx))
        simU = np.zeros((Nsim, self.nu))
        # simTrack = np.zeros((Nsim + self.N, 3))
        simTrack = np.zeros((Nsim + self.N, 7))
        x_current = x0
        simX[0, :] = x0.reshape(1, -1)
        time_record = np.zeros(Nsim)

        # closed loop
        for i in range(Nsim):
            # yref_between = np.concatenate((p0, np.zeros(6), np.zeros(self.nu)))
            for j in range(self.N):
                # track_ref = self.track_cal(Tsim + self.T * j / self.N, a_max, v_max, n) 
                p_ref, q_ref = self.track_cal(Tsim + self.T * j / self.N, target_p=target_p_dis, target_v=target_v, n=n,target_yaw=target_yaw) 
                # simTrack[i+j, :] = track_ref
                simTrack[i+j, 0:3] = p_ref
                simTrack[i+j, 3:7] = q_ref
                yref_between = np.concatenate((p_ref, np.zeros(10), np.array([d_model.m * d_model.g]), np.zeros(3), np.zeros(self.nu)))
                # yref_between = np.concatenate((p_ref, np.zeros(3), q_ref, np.zeros(self.nu)))
                # yref_between = np.concatenate((p_ref, np.zeros(3), q_ref[1:4], np.zeros(self.nu)))              
                self.solver.set(j, 'yref', yref_between)
            # track_ref_N = self.track_cal(Tsim + self.T, a_max, v_max, n)
            p_ref_N, q_ref_N = self.track_cal(Tsim + self.T, target_p=target_p_dis, target_v=target_v, n=n,target_yaw=target_yaw)
            # simTrack[i+self.N, :] = track_ref_N
            simTrack[i+self.N, 0:3] = p_ref_N
            simTrack[i+self.N, 3:7] = q_ref_N
            yref_N = np.concatenate((p_ref_N, np.zeros(10), np.array([d_model.m * d_model.g]), np.zeros(3)))
            # yref_N = np.concatenate((p_ref_N, np.zeros(3), q_ref_N))
            # yref_N = np.concatenate((p_ref_N, np.zeros(3), q_ref_N[1:4]))
            # yref_N = np.concatenate((p0, np.zeros(6)))
            self.solver.set(self.N, 'yref', yref_N)
            
            ##  set inertial (stage 0)
            self.solver.set(0, 'lbx', x_current)
            self.solver.set(0, 'ubx', x_current)
            
            # solve
            start = timeit.default_timer()
            status = self.solver.solve()
            if status != 0 :
                raise Exception('acados acados_ocp_solver returned status {}. in closed loop iteration {}.'.format(status, i))
            time_record[i] =  timeit.default_timer() - start
            simU[i, :] = self.solver.get(0, 'u')
            
            # simulate system
            self.integrator.set('x', x_current)
            self.integrator.set('u', simU[i, :])
            status_s = self.integrator.solve()
            if status_s != 0:
                raise Exception('acados integrator returned status {}. in closed loop iteration {}.'.format(status, i))

            # update
            x_current = self.integrator.get('x')
            simX[i+1, :] = x_current
            Tsim = Tsim + self.T / self.N
            # print(Tsim)

        return simX,simU
        # print("average estimation time is {}".format(time_record.mean()))
        # print("max estimation time is {}".format(time_record.max()))
        # print("min estimation time is {}".format(time_record.min()))
        # np.savetxt(fname=self.save_dir + "/drone_state.csv", X=simX, fmt="%lf",delimiter=",")
        # np.savetxt(fname=self.save_dir + "/drone_control.csv", X=simU, fmt="%lf",delimiter=",")
        # np.savetxt(fname=self.save_dir + "/drone_track.csv", X=simTrack, fmt="%lf",delimiter=",")
        
    def track_cal(self, t,target_p, target_v, n,target_yaw):
        
        q_proc = np.array([np.cos(target_yaw/2), 0, np.sin(target_yaw/2), 0])
        return target_p + t*target_v, q_proc
    
    def euler2quatern(self, euler):
        q=np.zeros(4)
        phi = euler[0]
        theta = euler[1]
        psi = euler[2]
        
        q[0] = np.cos(phi/2)*np.cos(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        q[1] = np.sin(phi/2)*np.cos(theta/2)*np.cos(psi/2) - np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2)
        q[2] = np.cos(phi/2)*np.sin(theta/2)*np.cos(psi/2) + np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
        q[3] = np.cos(phi/2)*np.cos(theta/2)*np.sin(psi/2) - np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
        return q

class DroneNMPC(object):
    def __init__(self,N = 40,v_max = 1.5, a_max = 1.5):
        self.drone_model = DroneModel()
        self.opt = DroneOptimizer(d_model=self.drone_model.model,
                               d_constraint=self.drone_model.constraint, t_horizon=1, n_nodes=20)
        self.N = N
        self.v_max = v_max
        self.a_max = a_max
        self.dt = 0.05
        print("init done!")
    def solve(self,drone_state,target_p_dis,target_v,target_yaw):
        return self.opt.simulation(d_model=self.drone_model, T_max=2, a_max=self.a_max, v_max=self.v_max, n=1,drone_state=drone_state,target_p_dis=target_p_dis,target_v=target_v,target_yaw=target_yaw)