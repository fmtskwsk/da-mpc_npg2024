import os
import numpy as np
from numpy import linalg as LA
import scipy.linalg as li
from scipy import optimize
import time
import datetime
import csv
import pickle
import copy
pwd = os.path.dirname(os.path.abspath(__file__))


x_nr = np.load(pwd+"/../../data/x_nr.npy")
SIM_STEP = 2000
SIM_IDX = SIM_STEP + 1
MPC_INTERVAL=8
X_0_MIN=0.
INIT_IDX_LIST = []
OLD_INIT_IDX = -8
CSE_NUM = 1000
LAST_POINT = x_nr.shape[0] - SIM_IDX
while len(INIT_IDX_LIST) < CSE_NUM :
    for init_idx in range(OLD_INIT_IDX+MPC_INTERVAL, LAST_POINT+MPC_INTERVAL, MPC_INTERVAL) :
        if (x_nr[init_idx, 0] >= X_0_MIN) and (x_nr[init_idx, 0] < 15.)  :
            INIT_IDX_LIST.append(init_idx)
            OLD_INIT_IDX = init_idx
            with open(pwd+"/../../data/INIT_IDX_ALL_sp0x15.csv", 'a', encoding="utf-8", newline="") as f :
                csv.writer(f).writerow([str(init_idx)])
            break