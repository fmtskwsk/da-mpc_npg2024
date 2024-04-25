import os
import numpy as np
from numpy import linalg as LA
import scipy.linalg as li
from scipy import optimize
import time
import datetime
from matplotlib import pyplot as plt
from matplotlib import colors
import argparse
pwd = os.path.dirname(os.path.abspath(__file__))
SEED = 2024
np.random.seed(seed=SEED)


class Model() : 
    def __init__(self) :
        self.DT = 0.01
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
    def __init__(self, MEMBER_NUM=50, OBS_INTERVAL=8, INFLATION=1.04) :
        super().__init__()
        self.SPIN_UP_TRJ_STEP = int(1000)
        self.SPIN_UP_TRJ_IDX = self.SPIN_UP_TRJ_STEP + 1
        self.SPIN_UP_DA_STEP = int(8000)
        self.NATURE_RUN_STEP = int(2008000)
        self.NATURE_RUN_IDX = self.NATURE_RUN_STEP + 1 
        self.OBS_DIM = self.MODEL_DIM
        self.OBS_NOISE_MEAN = 0.
        self.OBS_NOISE_STD = np.sqrt(2.)
        self.OBS_INTERVAL = OBS_INTERVAL
        self.INITIAL_MEAN = 0.
        self.INITIAL_STD = np.sqrt(2.)
        self.x_sp = np.zeros((self.SPIN_UP_TRJ_IDX, self.MODEL_DIM))
        self.x_nr = np.zeros((self.NATURE_RUN_IDX, self.MODEL_DIM))
        self.y_o = np.zeros((self.NATURE_RUN_IDX, self.OBS_DIM))
        self.R = np.identity((self.OBS_DIM)) * (self.OBS_NOISE_STD**2)
        self.H = np.zeros((self.OBS_DIM, self.MODEL_DIM))
        self.INFLATION = INFLATION
        for i in range(self.OBS_DIM) :
            self.H[i, i] = 1.0
        self.MEMBER_NUM = MEMBER_NUM
        self.X_a = np.zeros((self.NATURE_RUN_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.X_b = np.zeros((self.NATURE_RUN_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.dX_a = np.zeros((self.NATURE_RUN_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.dX_b = np.zeros((self.NATURE_RUN_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.x_a_mean = np.zeros((self.NATURE_RUN_IDX, self.MODEL_DIM))
        self.x_b_mean = np.zeros((self.NATURE_RUN_IDX, self.MODEL_DIM))
    
    def initilize(self) :
        self.x_sp[0, :] = np.random.normal(self.INITIAL_MEAN, self.INITIAL_STD, self.MODEL_DIM)
        for t in range(self.SPIN_UP_TRJ_STEP) :
            self.x_sp[t+1, :] = self.runge_kutta(self.x_sp[t, :], self.DT)
        self.x_nr[0, :] = self.x_sp[self.SPIN_UP_TRJ_IDX-1, :]
        for t in range(self.NATURE_RUN_STEP) :
            self.x_nr[t+1, :] = self.runge_kutta(self.x_nr[t, :], self.DT)
        for k in range(self.NATURE_RUN_IDX) :
            self.y_o[k, :] = self.x_nr[k, :self.OBS_DIM] + np.random.normal(self.OBS_NOISE_MEAN, self.OBS_NOISE_STD, self.OBS_DIM)
        for i in range(self.MEMBER_NUM) :
            self.X_b[0, :, i] = self.x_nr[0, :] + np.random.normal(self.OBS_NOISE_MEAN, self.OBS_NOISE_STD, self.MODEL_DIM)
        self.X_a[0, :, :] = self.X_b[0, :, :]
        self.x_b_mean[0, :] = np.average(self.X_b[0, :, :], axis=1)
        self.x_a_mean[0, :] = self.x_b_mean[0, :]
        for i in range(self.MEMBER_NUM) :
            self.dX_b[0, :, i] = self.X_b[0, :, i] - self.x_b_mean[0, :]
        self.dX_a[0, :, :] = self.dX_b[0, :, :]

    def po(self) :
        for t in range(0, self.NATURE_RUN_STEP, self.OBS_INTERVAL) :
            for i in range(self.MEMBER_NUM) :
                self.X_b[t+1, :, i] = self.runge_kutta(self.X_a[t, :, i], self.DT)
                for k in range(t+1, t+self.OBS_INTERVAL) :
                    self.X_b[k+1, : , i] = self.runge_kutta(self.X_b[k, : , i], self.DT)
            if self.OBS_INTERVAL > 1 :
                n = k
            else :
                n = t
            self.x_b_mean[n+1, :] = np.average(self.X_b[n+1, :, :], axis=1)
            for i in range(self.MEMBER_NUM) :
                self.dX_b[n+1, :, i] = self.X_b[n+1, :, i] - self.x_b_mean[n+1, :]
            self.dX_b[n+1, :, :] *= self.INFLATION
            Z_b = self.dX_b[n+1, :, :] / np.sqrt(self.MEMBER_NUM - 1.)
            Y_b = self.H @ Z_b
            K = Z_b @ Y_b.T @ LA.inv(Y_b @ Y_b.T + self.R)
            for i in range(self.MEMBER_NUM) :
                epsilon = np.random.normal(self.OBS_NOISE_MEAN, self.OBS_NOISE_STD, self.OBS_DIM)
                d_ob = self.y_o[n+1, :] - self.H @ self.X_b[n+1, :, i]
                self.X_a[n+1, :, i] = self.X_b[n+1, :, i] + K @ (d_ob + epsilon)
            self.x_a_mean[n+1, :] = np.average(self.X_a[n+1, :, :], axis=1)
            
    def calculate_rmse(self, x_tru, X_a, obs_interval, i, member_num, sim_idx) :
        square_diff_a = np.square(np.average(X_a[::obs_interval, :, :], axis=2) - x_tru[::obs_interval, :])
        rmse_a = np.sqrt(np.average(square_diff_a, axis = 1))
        rmse_avg = np.average(rmse_a[:])
        print("RMSE =", rmse_avg)
        np.save(pwd+"/../../data/rmse_mem"+str(member_num)+"_inf"+"{:.2f}".format(i), rmse_avg)            
            
    def main(self) : 
        self.initilize()
        self.po()
        self.calculate_rmse(self.x_nr[self.SPIN_UP_DA_STEP:self.NATURE_RUN_IDX, :], self.X_a[self.SPIN_UP_DA_STEP:self.NATURE_RUN_IDX, :], self.OBS_INTERVAL, self.INFLATION, self.MEMBER_NUM, self.NATURE_RUN_IDX-self.SPIN_UP_DA_STEP)

def get_args() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("member_num", type=int)
    parser.add_argument("inflation", type=float)
    args = parser.parse_args()
    return args


args = get_args()
enkf = DataAssimilation(MEMBER_NUM=args.member_num, INFLATION=args.inflation)
enkf.main()
np.save(pwd+"/../../data/x_nr", enkf.x_nr[enkf.SPIN_UP_DA_STEP:enkf.NATURE_RUN_IDX, :])
np.save(pwd+"/../../data/y_o", enkf.y_o[enkf.SPIN_UP_DA_STEP:enkf.NATURE_RUN_IDX, :])
np.save(pwd+"/../../data/X_a_PO_mem"+str(enkf.MEMBER_NUM)+"_inf"+"{:.2f}".format(enkf.INFLATION), enkf.X_a[enkf.SPIN_UP_DA_STEP:enkf.NATURE_RUN_IDX, :])
np.save(pwd+"/../../data/X_b_PO_mem"+str(enkf.MEMBER_NUM)+"_inf"+"{:.2f}".format(enkf.INFLATION), enkf.X_b[enkf.SPIN_UP_DA_STEP:enkf.NATURE_RUN_IDX, :])
