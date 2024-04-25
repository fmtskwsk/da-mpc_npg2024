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
plt.rcParams["font.size"] = 22
pwd = os.path.dirname(os.path.abspath(__file__))


PRED_HORIZON_STEP = 20
MEMBER = 50
MEM_INF_DICT = {10: 1.50, 20: 1.18, 30: 1.08, 40: 1.06, 50: 1.04, 100: 1.02}
RESULT_LIST = []
INITIAL_STATE_LIST = ["all-mem-random", "all-mem-mean", "rs-mem-random", "rs-mem-mean", "rs-mem-largest"]
SUCCESS_RATE_LIST = []
SUM_X_LIST = []
SUM_U_LIST = []
DT = 0.01

for i, init in enumerate(INITIAL_STATE_LIST) :
    with open(pwd+"/../../data/pred"+str(PRED_HORIZON_STEP)+"_cntl8_mem"+str(MEMBER)+"_inf"+"{:.2f}".format(MEM_INF_DICT[MEMBER])+"_u3_"+init+"_sp0x15.csv", 'r', encoding="utf-8") as f :
        CSV_DATA = list(csv.reader(f, delimiter=','))
        RESULT_LIST.append([[int(data[0]), float(data[2]), float(data[4]), float(data[9])] for data in CSV_DATA])
    RESULT_ARR = np.array(RESULT_LIST[i])
    result_len = RESULT_ARR.shape[0]
    success_rate = np.sum(RESULT_ARR[:, 0]) / result_len
    sum_x_mean = np.average(RESULT_ARR[:, 1])
    sum_u_mean = np.average(RESULT_ARR[:, 2])
    SUCCESS_RATE_LIST.append(success_rate)
    SUM_X_LIST.append(sum_x_mean)
    SUM_U_LIST.append(sum_u_mean)

plt.rcParams["font.size"] = 36
plt.rcParams['axes.axisbelow'] = True
fig, ax = plt.subplots(1, 3, figsize=(40, 11), dpi=600)
INTERVAL = 5.
WIDTH = 3.
X = np.arange(0, len(INITIAL_STATE_LIST)*INTERVAL, INTERVAL)
ax[0].bar(X, SUCCESS_RATE_LIST, width=WIDTH, color="#F5741F", align="center", tick_label=["Random\n(all mem.)", "Mean\n(all mem.)", "Random\n(RS mem.)", "Mean\n(RS mem.)", "Largest\n(RS mem.)"])
SUCCESS_RATE_LIST = []
ax[0].set_title("(a) Success rate (SR)", fontsize=40)
ax[0].set_ylabel("Success rate")
ax[0].set_xlim(0.-INTERVAL/2, X[-1]+INTERVAL/2)
ax[0].set_ylim(0., 1.1)
ax[0].grid()
ax[0].tick_params(axis='x', which='major', labelsize=28)
ax[1].bar(X, np.array(SUM_X_LIST)*DT, width=WIDTH, color="#F14432", align="center", tick_label=["Random\n(all mem.)", "Mean\n(all mem.)", "Random\n(RS mem.)", "Mean\n(RS mem.)", "Largest\n(RS mem.)"])
SUCCESS_RATE_LIST = []
ax[1].set_title("(b) Mean total failure (MTF)", fontsize=40)
ax[1].set_ylabel("Total failure")
ax[1].set_xlim(0.-INTERVAL/2, X[-1]+INTERVAL/2)
ax[1].grid()
ax[1].tick_params(axis='x', which='major', labelsize=28)
ax[2].bar(X, np.array(SUM_U_LIST)*DT, width=WIDTH, color="#61A7D2", align="center", tick_label=["Random\n(all mem.)", "Mean\n(all mem.)", "Random\n(RS mem.)", "Mean\n(RS mem.)", "Largest\n(RS mem.)"])
SUCCESS_RATE_LIST = []
ax[2].set_title("(c) Mean total control inputs (MTC)", fontsize=40)
ax[2].set_ylabel("Total control inputs")
ax[2].set_xlim(0.-INTERVAL/2, X[-1]+INTERVAL/2)
ax[2].grid()
ax[2].tick_params(axis='x', which='major', labelsize=28)
fig.tight_layout()
plt.savefig(pwd+"/../../fig/fig09")
