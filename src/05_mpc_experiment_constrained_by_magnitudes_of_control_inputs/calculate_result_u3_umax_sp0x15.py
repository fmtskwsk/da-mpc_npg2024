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
U_NORM_MAX_LIST = [20, 30, 40]
U_PENALTY = 1000
RESULT_LIST = []
for i, u in enumerate(U_NORM_MAX_LIST) :
    print("-------------------")
    print("max norm of u : ", u)
    with open(pwd+"/../../data/umax"+str(u)+"_penalty"+str(U_PENALTY)+"_u3_umax_sp0x15.csv", 'r', encoding="utf-8") as f :
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
    print(f"sucesss_rate : {success_rate:.5f}")
    print(f"sum_x : {sum_x_mean*DT}")
    print(f"sum_u : {sum_u_mean*DT}")
    print("-------------------")
    print()