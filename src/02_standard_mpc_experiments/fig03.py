from cProfile import label
import os
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
import scipy.linalg as li
from scipy.optimize import minimize
from fractions import Fraction
plt.rcParams["font.size"] = 20
import pickle
import copy
pwd = os.path.dirname(os.path.abspath(__file__))


INIT_IDX = 24
SIM_IDX = 2000 + 1
MEMBER_NUM = 50
x_nr = np.load(pwd+"/../../data/x_nr.npy")[INIT_IDX:INIT_IDX+SIM_IDX, :]
x_tru = np.load(pwd+"/../../data/x_tru_u3.npy")

plt.rcParams["font.size"] = 18
fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(projection='3d'), dpi=600)
ax[0].plot(x_nr[:, 0], x_nr[:,1], x_nr[:, 2], linewidth=1.2, label="NR without Control", color="#8C8C8C")
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[0].set_zlabel(r"$z$")
ax[0].set_xlim(-20, 20)
ax[0].set_ylim(-30, 30)
ax[0].set_zlim(0, 50)
ax[0].set_title("(a) NR", fontsize=28)
ax[1].plot(x_tru[:, 0], x_tru[:,1], x_tru[:, 2], linewidth=1.2, label="NR with Control", color="#333333")
ax[1].set_xlabel(r"$x$")
ax[1].set_ylabel(r"$y$")
ax[1].set_zlabel(r"$z$")
ax[1].set_xlim(-20, 20)
ax[1].set_ylim(-30, 30)
ax[1].set_zlim(0, 50)
ax[1].set_title("(b) Controlled NR", fontsize=28)
fig.tight_layout()
plt.savefig(pwd+"/../../fig/fig03")