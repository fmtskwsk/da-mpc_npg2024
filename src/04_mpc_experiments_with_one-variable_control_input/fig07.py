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
plt.rcParams["font.size"] = 20
pwd = os.path.dirname(os.path.abspath(__file__))


x_tru_ux = np.load(pwd+"/../../data/x_tru_ux.npy")
x_tru_uy = np.load(pwd+"/../../data/x_tru_uy.npy")
x_tru_uz = np.load(pwd+"/../../data/x_tru_uz.npy")

plt.rcParams["font.size"] = 18
fig, ax = plt.subplots(1, 3, figsize=(20, 6), subplot_kw=dict(projection='3d'), dpi=600)
ax[0].plot(x_tru_ux[:, 0], x_tru_ux[:,1], x_tru_ux[:, 2], linewidth=1.2, color="#333333")
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[0].set_zlabel(r"$z$")
ax[0].set_xlim(-20, 20)
ax[0].set_ylim(-30, 30)
ax[0].set_zlim(0, 50)
ax[0].set_title("(a) NR controlled by "+r"$u_x$", fontsize=24)
ax[1].plot(x_tru_uy[:, 0], x_tru_uy[:,1], x_tru_uy[:, 2], linewidth=1.2, color="#333333")
ax[1].set_xlabel(r"$x$")
ax[1].set_ylabel(r"$y$")
ax[1].set_zlabel(r"$z$")
ax[1].set_xlim(-20, 20)
ax[1].set_ylim(-30, 30)
ax[1].set_zlim(0, 50)
ax[1].set_title("(b) NR controlled by "+r"$u_y$", fontsize=24)
ax[2].plot(x_tru_uz[:, 0], x_tru_uz[:,1], x_tru_uz[:, 2], linewidth=1.2, color="#333333")
ax[2].set_xlabel(r"$x$")
ax[2].set_ylabel(r"$y$")
ax[2].set_zlabel(r"$z$")
ax[2].set_xlim(-20, 20)
ax[2].set_ylim(-30, 30)
ax[2].set_zlim(0, 50)
ax[2].set_title("(c) NR controlled by "+r"$u_z$", fontsize=24)
fig.tight_layout()
plt.savefig(pwd+"/../../fig/fig07")
