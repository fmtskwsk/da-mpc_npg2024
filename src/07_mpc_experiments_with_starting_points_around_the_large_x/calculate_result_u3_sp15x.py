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


PRED_HORIZON_STEP = 20
MEMBER = 50
MEM_INF_DICT = {10: 1.50, 20: 1.18, 30: 1.08, 40: 1.06, 50: 1.04, 100: 1.02}
RESULT_LIST = []
SUCCESS_RATE_LIST = []
SUM_X_LIST = []
SUM_U_LIST = []
DT = 0.01

with open(pwd+"/../../data/pred"+str(PRED_HORIZON_STEP)+"_cntl8_mem"+str(MEMBER)+"_inf"+"{:.2f}".format(MEM_INF_DICT[MEMBER])+"_u3_sp15x.csv", 'r', encoding="utf-8") as f :
    CSV_DATA = list(csv.reader(f, delimiter=','))
    RESULT_LIST.append([[int(data[0]), float(data[2]), float(data[4]), float(data[9])] for data in CSV_DATA])
RESULT_ARR = np.array(RESULT_LIST[0])
result_len = RESULT_ARR.shape[0]
success_rate = np.sum(RESULT_ARR[:, 0]) / result_len
sum_x_mean = np.average(RESULT_ARR[:, 1])
sum_u_mean = np.average(RESULT_ARR[:, 2])
print("-------------------")
print("prediction horizon:", PRED_HORIZON_STEP, "member:", MEMBER)
print(f"result_len : {result_len}")
print(f"sucesss_rate : {success_rate}")
print(f"sum_x : {sum_x_mean*0.01}")
print(f"sum_u : {sum_u_mean*0.01}")
SUCCESS_RATE_LIST.append(success_rate)
SUM_X_LIST.append(sum_x_mean)
SUM_U_LIST.append(sum_u_mean)
print("-------------------")
print()
