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
PRED_HORIZON_STEP = 20
MEM_INF_DICT = {10: 1.50, 20: 1.18, 30: 1.08, 40: 1.06, 50: 1.04, 100: 1.02}
MEMBER = 50
INPUT_TYPE_LIST = ["ux", "uy", "uz"]
RESULT_LIST = []
for u in INPUT_TYPE_LIST :
    print("-------------------")
    print("input type:", u)
    with open(pwd+"/../../data/pred"+str(PRED_HORIZON_STEP)+"_cntl8_mem"+str(MEMBER)+"_inf"+"{:.2f}".format(MEM_INF_DICT[MEMBER])+"_"+u+"_sp0x15.csv", 'r', encoding="utf-8") as f :
        CSV_DATA = list(csv.reader(f, delimiter=','))
        RESULT_LIST.append([[int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4])] for data in CSV_DATA])
    RESULT_ARR = np.array(RESULT_LIST[0])
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