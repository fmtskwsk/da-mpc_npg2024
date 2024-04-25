from cProfile import label
import os
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
import scipy.linalg as li
from scipy.optimize import minimize
from fractions import Fraction
import csv
import collections
pwd = os.path.dirname(os.path.abspath(__file__))


INIT_IDX_ALL_LIST = []
with open(pwd+"/../../data/INIT_IDX_ALL_sp0x15.csv", "r", encoding="utf-8") as f :
    CSV_DATA = list(csv.reader(f, delimiter=","))
    INIT_IDX_ALL_LIST.append([int(data[0]) for data in CSV_DATA])
INIT_IDX_ALL_SET = set(INIT_IDX_ALL_LIST[0])

U_NORM_MAX_LIST = [20, 30, 40]
U_PENALTY = 1000
INIT_IDX_DONE_LIST = []
for i, u in enumerate(U_NORM_MAX_LIST) :
    print("-------------------")
    print("max norm of u : ", u)
    with open(pwd+"/../../data/umax"+str(u)+"_penalty"+str(U_PENALTY)+"_u3_umax_sp0x15.csv", 'r', encoding="utf-8") as f :
        CSV_DATA = list(csv.reader(f, delimiter=','))
        INIT_IDX_DONE_LIST.append([int(data[-1]) for data in CSV_DATA])
    INIT_IDX_DONE_SET = set(INIT_IDX_DONE_LIST[i])
    print("lack:", INIT_IDX_ALL_SET-INIT_IDX_DONE_SET)
    print("excess:", [k for k, v in collections.Counter(INIT_IDX_DONE_LIST[i]).items() if v > 1])
    print("-------------------")
    print()
    with open(pwd+"/../../data/INIT_IDX_DONE_umax"+str(u)+"_penalty"+str(U_PENALTY)+"_u3_umax_sp0x15.csv", 'w', encoding="utf-8") as f :
        csv.writer(f)
    for idx in INIT_IDX_DONE_SET :
        with open(pwd+"/../../data/INIT_IDX_DONE_umax"+str(u)+"_penalty"+str(U_PENALTY)+"_u3_umax_sp0x15.csv", 'a', encoding="utf-8", newline="") as f :
            csv.writer(f).writerow([str(idx)])