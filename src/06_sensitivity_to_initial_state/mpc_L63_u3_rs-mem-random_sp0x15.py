import numpy as np
from numpy import linalg as LA
import scipy.linalg as li
from scipy import optimize
import time
import datetime
import csv
import pickle
import copy
import random
import sys
import argparse
import os
pwd = os.path.dirname(os.path.abspath(__file__))


class Model() : 
    def __init__(self, SIM_STEP=2000) :
        self.SEED = 2024
        np.random.seed(seed=self.SEED)
        self.DT = 0.01
        self.SIM_STEP = SIM_STEP
        self.SIM_IDX = self.SIM_STEP + 1
        self.SIGMA = 10.
        self.RHO = 28.
        self.BETA = 8. / 3.
        self.MODEL_DIM = 3
        
    def lorenz63(self, x) : 
        x_dot = np.zeros((self.MODEL_DIM))
        x_dot[0] = - self.SIGMA * x[0] + self.SIGMA * x[1]
        x_dot[1] = - x[0] * x[2] + self.RHO * x[0] - x[1]
        x_dot[2] = x[0] * x[1] - self.BETA * x[2]
        return x_dot

    def runge_kutta(self, x, dt) :
        k1 = dt * self.lorenz63(x)
        k2 = dt * self.lorenz63(x+0.5*k1)
        k3 = dt * self.lorenz63(x+0.5*k2)
        k4 = dt * self.lorenz63(x+k3)
        x = x + (1. / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        return x
    

class DataAssimilation(Model) :
    def __init__(self, MEMBER_NUM, INFLATION, OBS_INTERVAL=8):
        super().__init__()
        self.time_da = 0.
        self.count_da = 0
        self.OBS_DIM = self.MODEL_DIM
        self.OBS_NOISE_MEAN = 0.
        self.OBS_NOISE_STD = np.sqrt(2.)
        self.OBS_INTERVAL = OBS_INTERVAL
        self.x_nr = np.load(pwd+"/../../data/x_nr.npy")
        self.y_o = np.zeros((self.SIM_IDX, self.OBS_DIM))
        self.R = np.identity((self.OBS_DIM)) * (self.OBS_NOISE_STD**2)
        self.H = np.zeros((self.OBS_DIM, self.MODEL_DIM))
        self.INFLATION = INFLATION
        for i in range(self.OBS_DIM) :
            self.H[i, i] = 1.0
        self.MEMBER_NUM = MEMBER_NUM
        self.X_a = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.X_b = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.dX_a = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.dX_b = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.x_a_mean = np.zeros((self.SIM_IDX, self.MODEL_DIM))
        self.x_b_mean = np.zeros((self.SIM_IDX, self.MODEL_DIM))
        self.X_a_all = np.load(pwd+"/../../data/X_a_PO_mem"+str(MEMBER_NUM)+"_inf"+"{:.2f}".format(INFLATION)+".npy")
        self.X_b_all = np.load(pwd+"/../../data/X_b_PO_mem"+str(MEMBER_NUM)+"_inf"+"{:.2f}".format(INFLATION)+".npy")
        self.P_a = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MODEL_DIM))
        self.P_b = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MODEL_DIM))

    def po(self, t) :
        self.x_b_mean[t, :] = np.average(self.X_b[t, :, :], axis=1)
        for i in range(self.MEMBER_NUM) :
            self.dX_b[t, :, i] = self.X_b[t, :, i] - self.x_b_mean[t, :]
        self.dX_b[t, :, :] *= self.INFLATION
        self.P_b[t, :, :] = self.dX_b[t, :, :] @ self.dX_b[t, :, :].T * (1. / (self.MEMBER_NUM - 1))
        Z_b = self.dX_b[t, :, :] / np.sqrt(self.MEMBER_NUM - 1.)
        Y_b = self.H @ Z_b
        K = Z_b @ Y_b.T @ LA.inv(Y_b @ Y_b.T + self.R)
        for i in range(self.MEMBER_NUM) :
            epsilon = np.random.normal(self.OBS_NOISE_MEAN, self.OBS_NOISE_STD, self.OBS_DIM)
            d_ob = self.y_o[t, :] - self.H @ self.X_b[t, :, i]
            self.X_a[t, :, i] = self.X_b[t, :, i] + K @ (d_ob + epsilon)
        self.x_a_mean[t, :] = np.average(self.X_a[t, :, :], axis=1)
        for i in range(self.MEMBER_NUM) :
            self.dX_a[t, :, i] = self.X_a[t, :, i] - self.x_a_mean[t, :]
        self.P_a[t, :, :] = self.dX_a[t, :, :] @ self.dX_a[t, :, :].T * (1. / (self.MEMBER_NUM - 1))


class ModelPredictiveControl(DataAssimilation) :
    def __init__(self, PRED_HORIZON_STEP=20, CNTL_HORIZON_STEP=8, MPC_INTERVAL=8, MEMBER_NUM=50, INFLATION=1.04, \
        INPUT_DIM=3, X_0_MIN=0., \
        OPT_METHOD="lm", PENALTY_PARAM_X=1e4, THRESHOLD=1e-4) :
        super().__init__(MEMBER_NUM=MEMBER_NUM, INFLATION=INFLATION)
        self.PRED_HORIZON_STEP = PRED_HORIZON_STEP
        self.PRED_HORIZON_IDX = self.PRED_HORIZON_STEP + 1
        self.CNTL_HORIZON_STEP = CNTL_HORIZON_STEP
        self.CNTL_HORIZON_IDX = self.CNTL_HORIZON_STEP + 1
        self.DTAU = self.DT
        self.MPC_INTERVAL = MPC_INTERVAL
        self.time_mpc = 0.
        self.count_mpc = 0
        self.INPUT_DIM = INPUT_DIM
        self.UNKNOWN_DIM = self.INPUT_DIM
        self.UNKNOWNVEC_DIM = self.UNKNOWN_DIM * self.CNTL_HORIZON_STEP
        self.X_0_MIN = X_0_MIN
        self.OPT_METHOD = OPT_METHOD
        self.PENALTY_PARAM_X = PENALTY_PARAM_X
        self.THRESHOLD = THRESHOLD
        self.x_opt = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM))
        self.l_opt = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM))
        self.u_opt = np.zeros((self.CNTL_HORIZON_STEP, self.INPUT_DIM))
        self.x_tru = np.zeros((self.SIM_IDX, self.MODEL_DIM))
        self.u_tru = np.zeros((self.SIM_STEP, self.INPUT_DIM))
        self.u_tru_norm = np.zeros((self.SIM_STEP))
        self.x_opt_list = []
        self.u_opt_list = []
        self.x_opt_list_list = []
        self.u_opt_list_list = []

    def initialize(self) : 
        INIT_IDX_ALL_LIST = []
        INIT_IDX_DONE_LIST = []
        if not os.path.exists(pwd+"/../../data/INIT_IDX_DONE_pred"+str(self.PRED_HORIZON_STEP)+"_cntl"+str(self.CNTL_HORIZON_STEP)+"_mem"+str(self.MEMBER_NUM)+"_inf"+"{:.2f}".format(self.INFLATION)+"_u3_rs-mem-random_sp0x15.csv") :
            with open(pwd+"/../../data/INIT_IDX_DONE_pred"+str(self.PRED_HORIZON_STEP)+"_cntl"+str(self.CNTL_HORIZON_STEP)+"_mem"+str(self.MEMBER_NUM)+"_inf"+"{:.2f}".format(self.INFLATION)+"_u3_rs-mem-random_sp0x15.csv", 'w', encoding="utf-8") as f :
                csv.writer(f)
        with open(pwd+"/../../data/INIT_IDX_ALL_sp0x15.csv", 'r', encoding="utf-8") as f :
            CSV_DATA = list(csv.reader(f, delimiter=','))
            INIT_IDX_ALL_LIST = [int(data[0]) for data in CSV_DATA]
        with open(pwd+"/../../data/INIT_IDX_DONE_pred"+str(self.PRED_HORIZON_STEP)+"_cntl"+str(self.CNTL_HORIZON_STEP)+"_mem"+str(self.MEMBER_NUM)+"_inf"+"{:.2f}".format(self.INFLATION)+"_u3_rs-mem-random_sp0x15.csv", 'r', encoding="utf-8") as f :
            CSV_DATA = list(csv.reader(f, delimiter=','))
            INIT_IDX_DONE_LIST = [int(data[0]) for data in CSV_DATA]
        INIT_IDX = random.choice(INIT_IDX_ALL_LIST)
        INIT_IDX_SET = set()
        INIT_IDX_ALL_SET = set(INIT_IDX_ALL_LIST)
        while (INIT_IDX in INIT_IDX_DONE_LIST) :
            INIT_IDX = random.choice(INIT_IDX_ALL_LIST)
            INIT_IDX_SET.add(INIT_IDX)
            if INIT_IDX_SET == INIT_IDX_ALL_SET :
                print("return")
                sys.exit()
        self.INIT_IDX = INIT_IDX
        with open(pwd+"/../../data/INIT_IDX_DONE_pred"+str(self.PRED_HORIZON_STEP)+"_cntl"+str(self.CNTL_HORIZON_STEP)+"_mem"+str(self.MEMBER_NUM)+"_inf"+"{:.2f}".format(self.INFLATION)+"_u3_rs-mem-random_sp0x15.csv", 'a', encoding="utf-8", newline="") as f :
            csv.writer(f).writerow([str(self.INIT_IDX)])
        self.SEED = 2024
        np.random.seed(seed=self.SEED)
        self.x_tru[0, :] = self.x_nr[self.INIT_IDX, :]
        self.X_b[0, :, :] = self.X_b_all[self.INIT_IDX, :, :]
        self.X_a[0, :, :] = self.X_a_all[self.INIT_IDX, :, :]
        self.x_b_mean[0, :] = np.average(self.X_b[0, :, :], axis=1)
        self.x_a_mean[0, :] = np.average(self.X_a[0, :, :], axis=1)
        for i in range(self.MEMBER_NUM) :
            self.dX_b[0, :, i] = self.X_b[0, :, i] - self.x_b_mean[0, :]
            self.dX_a[0, :, i] = self.X_a[0, :, i] - self.x_a_mean[0, :]
        self.P_b[0, :, :] = self.dX_b[0, :, :] @ self.dX_b[0, :, :].T * (1. / (self.MEMBER_NUM - 1))
        self.P_a[0, :, :] = self.dX_a[0, :, :] @ self.dX_a[0, :, :].T * (1. / (self.MEMBER_NUM - 1))

    def state_equation(self, x, u) :
        x_dot = np.zeros((self.MODEL_DIM))
        x_dot[0] = - self.SIGMA * x[0] + self.SIGMA * x[1] + u[0]
        x_dot[1] = - x[0] * x[2] + self.RHO * x[0] - x[1] + u[1]
        x_dot[2] = x[0] * x[1] - self.BETA * x[2] + u[2]
        return x_dot
        
    def adjoint_equation(self, l, x, u) : 
        l_dot = np.zeros((self.MODEL_DIM))
        l_dot[0] = l[0] * (- self.SIGMA) + l[1] * (- x[2] + self.RHO) + l[2] * x[1] - self.PENALTY_PARAM_X * max(-x[0], 0.)
        l_dot[1] = l[0] * self.SIGMA + l[1] * (-1.) + l[2] * x[0]
        l_dot[2] = l[1] * (- x[0]) + l[2] * (- self.BETA)
        return l_dot
    
    def runge_kutta_state(self, x, u, dt) :
        k1 = dt * self.state_equation(x, u)
        k2 = dt * self.state_equation(x+0.5*k1, u)
        k3 = dt * self.state_equation(x+0.5*k2, u)
        k4 = dt * self.state_equation(x+k3, u)
        x = x + (1. / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        return x
    
    def runge_kutta_adjoint(self, l, x, u, dt) : 
        k1 = dt * self.adjoint_equation(l, x, u)
        k2 = dt * self.adjoint_equation(l+0.5*k1, x, u)
        k3 = dt * self.adjoint_equation(l+0.5*k2, x, u)
        k4 = dt * self.adjoint_equation(l+k3, x, u)
        l = l + (1. / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        return l

    def calculate_state(self, x_init, u_opt) :
        x_opt = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM))
        x_opt[0, :] = x_init
        for k in range(self.CNTL_HORIZON_STEP) :
            x_opt[k+1, :] = self.runge_kutta_state(x_opt[k, :], u_opt[k, :], self.DTAU)
        for k in range(self.CNTL_HORIZON_STEP, self.PRED_HORIZON_STEP) :
            x_opt[k+1, :] = self.runge_kutta_state(x_opt[k, :], np.zeros((self.INPUT_DIM)), self.DTAU)
        return x_opt
    
    def calculate_adjoint(self, x_opt, u_opt) :
        l_opt = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM))
        l_opt[-1, 0] = - self.PENALTY_PARAM_X * max(-x_opt[-1, 0], 0.)
        l_opt[-1, 1] = 0.
        l_opt[-1, 2] = 0.
        for k in range(self.PRED_HORIZON_IDX-1, self.CNTL_HORIZON_STEP+1-1, -1) :
            l_opt[k-1] = self.runge_kutta_adjoint(l_opt[k], x_opt[k-1], np.zeros((self.INPUT_DIM)), self.DTAU)
        for k in range(self.CNTL_HORIZON_STEP, 1-1, -1) :
            l_opt[k-1] = self.runge_kutta_adjoint(l_opt[k], x_opt[k-1], u_opt[k-1], self.DTAU)
        return l_opt

    def calculate_zeta(self, mu, x_init) :
        u_opt = self.mu_to_opt(mu)
        zeta = np.zeros((self.UNKNOWNVEC_DIM))
        self.x_opt = self.calculate_state(x_init, u_opt)
        self.l_opt = self.calculate_adjoint(self.x_opt, u_opt)
        x = self.x_opt
        l = self.l_opt
        u = u_opt
        self.x_opt_list.append(copy.deepcopy(x))
        self.u_opt_list.append(copy.deepcopy(u))
        for k in range(self.CNTL_HORIZON_STEP) :
            zeta[k*self.UNKNOWN_DIM+0] = u[k, 0] + l[k+1, 0]
            zeta[k*self.UNKNOWN_DIM+1] = u[k, 1] + l[k+1, 1]
            zeta[k*self.UNKNOWN_DIM+2] = u[k, 2] + l[k+1, 2]
        return zeta

    def opt_to_mu(self, u_opt) :
        mu = np.zeros((self.UNKNOWNVEC_DIM))
        for k in range(self.CNTL_HORIZON_STEP) :
            mu[k*self.UNKNOWN_DIM+0] = u_opt[k, 0]
            mu[k*self.UNKNOWN_DIM+1] = u_opt[k, 1]
            mu[k*self.UNKNOWN_DIM+2] = u_opt[k, 2]
        return mu
    
    def mu_to_opt(self, mu) :
        u_opt = np.zeros((self.CNTL_HORIZON_STEP, self.INPUT_DIM))
        for k in range(self.CNTL_HORIZON_STEP) :
            u_opt[k, 0] = mu[k*self.UNKNOWN_DIM+0]
            u_opt[k, 1] = mu[k*self.UNKNOWN_DIM+1]
            u_opt[k, 2] = mu[k*self.UNKNOWN_DIM+2]
        return u_opt

    def main(self) :
        if self.OBS_INTERVAL % self.MPC_INTERVAL != 0 : 
            return
        self.initialize()
        for t in range(0, self.SIM_STEP, self.MPC_INTERVAL) :
            X_j = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM, self.MEMBER_NUM))
            X_j[0, :, :] = self.X_a[t, :, :] if t % self.OBS_INTERVAL == 0 else self.X_b[t, :, :]
            for i in range(self.MEMBER_NUM) :
                for k in range(self.PRED_HORIZON_STEP) :
                    X_j[k+1, :, i] = self.runge_kutta(X_j[k, :, i], self.DT)
            if np.any(X_j[:, 0, :] < self.X_0_MIN) :
                self.count_mpc += 1
                s_member_list = np.where(np.any(X_j[:, 0, :] < 0., axis=0))[0]
                s_member = np.random.choice(s_member_list)
                x_s = X_j[:, :, s_member]
                x_init = x_s[0, :]
                self.u_opt = np.zeros((self.CNTL_HORIZON_STEP, self.INPUT_DIM))
                mu = self.opt_to_mu(self.u_opt)
                time_mpc_start = time.perf_counter()
                solution = optimize.root(self.calculate_zeta, mu, args=(x_init), method=self.OPT_METHOD, tol=self.THRESHOLD)
                mu = solution.x
                time_mpc_end = time.perf_counter()
                self.time_mpc += time_mpc_end - time_mpc_start
                self.x_opt_list_list.append(copy.deepcopy(self.x_opt_list))
                self.u_opt_list_list.append(copy.deepcopy(self.u_opt_list))
                self.x_opt_list = []
                self.u_opt_list = []
                self.u_opt = self.mu_to_opt(mu)
                if self.CNTL_HORIZON_STEP >= self.MPC_INTERVAL :
                    self.u_tru[t:t+self.MPC_INTERVAL, :] = self.u_opt[0:self.MPC_INTERVAL, :]
                else : 
                    self.u_tru[t:t+self.CNTL_HORIZON_STEP, :] = self.u_opt[0:self.CNTL_HORIZON_STEP, :]
                for k in range(t, t+self.MPC_INTERVAL) :
                    self.x_tru[k+1, :] = self.runge_kutta_state(self.x_tru[k, :], self.u_tru[k, :], self.DT)
                    self.u_tru_norm[k] = LA.norm(self.u_tru[k, :])
                    print("t = {:>4}".format(k+1) + ", x = {: #7.2f}".format(self.x_tru[k+1, 0]) + ", u_norm = {: #7.2f}".format(self.u_tru_norm[k]))
            else :
                self.x_opt_list_list.append(copy.deepcopy([np.ones((self.PRED_HORIZON_IDX, self.MODEL_DIM))*np.nan]))
                self.u_opt_list_list.append(copy.deepcopy([np.ones((self.CNTL_HORIZON_STEP, self.INPUT_DIM))*np.nan]))
                for k in range(t, t+self.MPC_INTERVAL) :
                    self.x_tru[k+1, :] = self.runge_kutta(self.x_tru[k, :], self.DT)
                    print("t = {:>4}".format(k+1) + ", x = {: #7.2f}".format(self.x_tru[k+1, 0]) + ", u_norm = {: #7.2f}".format(self.u_tru_norm[k]))
            n = k
            for i in range(self.MEMBER_NUM) :
                self.X_b[t+1, :, i] = self.runge_kutta_state(self.X_a[t, :, i], self.u_tru[t, :], self.DT)
                for k in range(t+1, t+self.MPC_INTERVAL) :
                    self.X_b[k+1, :, i] = self.runge_kutta_state(self.X_b[k, : , i], self.u_tru[k, :], self.DT)
            if n % self.OBS_INTERVAL != (self.OBS_INTERVAL-1) :
                self.x_b_mean[n+1, :] = np.average(self.X_b[n+1, :, :], axis=1)
                for i in range(self.MEMBER_NUM) :
                    self.dX_b[n+1, :, i] = self.X_b[n+1, :, i] - self.x_b_mean[n+1, :]
                self.dX_b[n+1, :, :] *= self.INFLATION
                self.P_b[n+1, :, :] = self.dX_b[n+1, :, :] @ self.dX_b[n+1, :, :].T * (1. / (self.MEMBER_NUM - 1))
            else : 
                self.count_da += 1
                time_da_start = time.perf_counter()
                self.y_o[n+1, :] = self.H @ self.x_tru[n+1, :] + np.random.normal(self.OBS_NOISE_MEAN, self.OBS_NOISE_STD, self.OBS_DIM)
                self.po(n+1)
                time_da_end = time.perf_counter()
                self.time_da += time_da_end - time_da_start

    def calculate_success_bool(self) :
        if np.all(self.x_tru[:, 0] >= self.X_0_MIN) :
            return 1
        else :
            return 0
    
    def calculate_min_negative(self) :
        return min(self.x_tru[:, 0])

    def calculate_sum_negative(self) :
        sum_negative = 0.
        for t in range(self.SIM_IDX) :
            if self.x_tru[t, 0] < self.X_0_MIN :
                sum_negative += self.x_tru[t, 0] - self.X_0_MIN
        return sum_negative
    
    def calculate_max_u_norm(self) :
        return max(self.u_tru_norm[:])
    
    def calculate_sum_u_norm(self) :
        sum_u_norm = 0.
        for t in range(self.SIM_STEP) :
            sum_u_norm += self.u_tru_norm[t]
        return sum_u_norm

    def calculate_rmse(self) :
        square_diff_a = np.square(np.average(self.X_a[::self.OBS_INTERVAL, :, :], axis=2) - self.x_tru[::self.OBS_INTERVAL, :])
        rmse_a = np.sqrt(np.average(square_diff_a, axis = 1))
        rmse_avg = np.average(rmse_a[:])
        return rmse_avg


def get_args() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_horizon_step", type=int)
    parser.add_argument("cntl_horizon_step", type=int)
    parser.add_argument("member_num", type=int)
    parser.add_argument("inflation", type=float)
    args = parser.parse_args()
    return args

args = get_args()
experiment_num = 10
for i in range(experiment_num) :
    system = ModelPredictiveControl(PRED_HORIZON_STEP=args.pred_horizon_step, CNTL_HORIZON_STEP=args.cntl_horizon_step, MEMBER_NUM=args.member_num, INFLATION=args.inflation)
    system.main()
    file_name = pwd+"/../../data/pred"+str(system.PRED_HORIZON_STEP)+"_cntl"+str(system.CNTL_HORIZON_STEP)+"_mem"+str(system.MEMBER_NUM)+"_inf"+"{:.2f}".format(system.INFLATION)+"_u3_rs-mem-random_sp0x15.csv"
    success_bool = system.calculate_success_bool()
    min_negative = system.calculate_min_negative()
    sum_negative = system.calculate_sum_negative()
    max_u_norm = system.calculate_max_u_norm()
    sum_u_norm = system.calculate_sum_u_norm()
    rmse = system.calculate_rmse()
    count_mpc = system.count_mpc
    time_mpc = system.time_mpc
    count_da = system.count_da
    time_da = system.time_da
    with open(file_name, 'a', encoding="utf-8", newline='') as f :
        csv.writer(f).writerow([str(success_bool), str(min_negative), str(sum_negative), str(max_u_norm), str(sum_u_norm), str(rmse), str(count_mpc), str(time_mpc), str(count_da), str(time_da), str(system.INIT_IDX)])