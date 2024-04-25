from cProfile import label
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
import scipy.linalg as li
import csv
import collections
import os
import sys
import random
from scipy.optimize import minimize
from fractions import Fraction
pwd = os.path.dirname(os.path.abspath(__file__))


DT = 0.01
PRED_HORIZON_STEP_LIST = [10, 20, 30, 40, 50]
MEM_INF_DICT = {10: 1.50, 20: 1.18, 30: 1.08, 40: 1.06, 50: 1.04, 100: 1.02}
MEM_LIST = [10, 20, 30, 40, 50, 100]
RESULT_LIST = []
for p in PRED_HORIZON_STEP_LIST :
    for i, m in enumerate(MEM_LIST) :
        print("-------------------")
        print("prediction horizon:", p, "member:", m)
        with open(pwd+"/../../data/pred"+str(p)+"_cntl8_mem"+str(m)+"_inf"+"{:.2f}".format(MEM_INF_DICT[m])+"_u3_sp0x15.csv", 'r', encoding="utf-8") as f :
            CSV_DATA = list(csv.reader(f, delimiter=','))
            RESULT_LIST.append([[int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])] for data in CSV_DATA])
        RESULT_ARR = np.array(RESULT_LIST[i])
        result_len = RESULT_ARR.shape[0]
        success_rate = np.sum(RESULT_ARR[:, 0]) / result_len
        min_x_mean = np.average(RESULT_ARR[:, 1])
        sum_x_mean = np.average(RESULT_ARR[:, 2])
        max_u_mean = np.average(RESULT_ARR[:, 3])
        sum_u_mean = np.average(RESULT_ARR[:, 4])
        print(f"result_len : {result_len}")
        print(f"sucesss_rate : {success_rate}")
        print(f"sum_x : {sum_x_mean*DT}")
        print(f"sum_u : {sum_u_mean*DT}")
        print("-------------------")
        print()
    np.save(pwd+"/../../data/result_pred"+str(p)+"_u3_sp0x15", np.array(RESULT_LIST))
    RESULT_LIST = []