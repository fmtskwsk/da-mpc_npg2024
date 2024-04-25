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

PRED_HORIZON_STEP = 20
MEM_INF_DICT = {10: 1.50, 20: 1.18, 30: 1.08, 40: 1.06, 50: 1.04, 100: 1.02}
MEMBER = 50
INPUT_TYPE_LIST = ["ux", "uy", "uz"]
INIT_IDX_DONE_LIST = []
for u in INPUT_TYPE_LIST :
    print("-------------------")
    print("input type: ", u)
    with open(pwd+"/../../data/pred"+str(PRED_HORIZON_STEP)+"_cntl8_mem"+str(MEMBER)+"_inf"+"{:.2f}".format(MEM_INF_DICT[MEMBER])+"_"+u+"_sp0x15.csv", 'r', encoding="utf-8") as f :
        CSV_DATA = list(csv.reader(f, delimiter=','))
        INIT_IDX_DONE_LIST.append([int(data[-1]) for data in CSV_DATA])
    INIT_IDX_DONE_SET = set(INIT_IDX_DONE_LIST[0])
    print("lack:", INIT_IDX_ALL_SET-INIT_IDX_DONE_SET)
    print("excess:", [k for k, v in collections.Counter(INIT_IDX_DONE_LIST[0]).items() if v > 1])
    print("-------------------")
    print()
    with open(pwd+"/../../data/INIT_IDX_DONE_pred"+str(PRED_HORIZON_STEP)+"_cntl8_mem"+str(MEMBER)+"_inf"+"{:.2f}".format(MEM_INF_DICT[MEMBER])+"_"+u+"_sp0x15.csv", 'w', encoding="utf-8") as f :
        csv.writer(f)
    for idx in INIT_IDX_DONE_SET :
        with open(pwd+"/../../data/INIT_IDX_DONE_pred"+str(PRED_HORIZON_STEP)+"_cntl8_mem"+str(MEMBER)+"_inf"+"{:.2f}".format(MEM_INF_DICT[MEMBER])+"_"+u+"_sp0x15.csv", 'a', encoding="utf-8", newline="") as f :
            csv.writer(f).writerow([str(idx)])