from cProfile import label
import os
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
import scipy.linalg as li
from scipy.optimize import minimize
from fractions import Fraction
import pickle
import copy
pwd = os.path.dirname(os.path.abspath(__file__))


INIT_IDX = 24
SIM_IDX = 2000 + 1
X_0_MIN = 0.
x_tru = np.load(pwd+"/../../data/x_tru_u3.npy")
u_tru_norm = np.load(pwd+"/../../data/u_tru_norm_u3.npy")

plt.rcParams["font.size"] = 18
fig, ax = plt.subplots(2, 1, figsize=(12, 7), dpi=600, gridspec_kw={"width_ratios": [1], "height_ratios": [1., 0.6]})
ax[0].plot(np.arange(SIM_IDX), x_tru[:, 0], linewidth=1.0, color="#333333", zorder=200)
ax[0].axhline(X_0_MIN, ls="--", color="black", linewidth=1.)
ax[0].set_xlabel(r"$t$")
ax[0].set_ylabel(r"$x$")
ax[0].set_title("(a) Controlled NR", fontsize=20)
ax[0].grid()
ax[1].plot(np.arange(SIM_IDX-1), u_tru_norm[:], linewidth=1., color="#255793")
ax[1].set_xlabel(r"$t$")
ax[1].set_ylabel("|u|")
ax[1].set_title("(b) Norm of control inputs", fontsize=20)
fig.tight_layout()
ax[1].grid()
plt.savefig(pwd+"/../../fig/fig04")