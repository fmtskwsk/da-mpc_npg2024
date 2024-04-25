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
plt.rcParams["font.size"] = 16
pwd = os.path.dirname(os.path.abspath(__file__))


SIM_IDX = 2000 + 1
x_tru_UNM20 = np.load(pwd+"/../../data/x_tru_u3_umax20.npy")
u_tru_norm_UNM20 = np.load(pwd+"/../../data/u_tru_norm_u3_umax20.npy")
x_tru_UNM30 = np.load(pwd+"/../../data/x_tru_u3_umax30.npy")
u_tru_norm_UNM30 = np.load(pwd+"/../../data/u_tru_norm_u3_umax30.npy")
x_tru_UNM40 = np.load(pwd+"/../../data/x_tru_u3_umax40.npy")
u_tru_norm_UNM40 = np.load(pwd+"/../../data/u_tru_norm_u3_umax40.npy")

fig, ax = plt.subplots(3, 2, figsize=(20, 14), gridspec_kw={"width_ratios": [1., 3], "height_ratios": [1., 1., 1.]}, dpi=600)
ax = fig.add_subplot(321, projection='3d')
ax.plot(x_tru_UNM20[:, 0], x_tru_UNM20[:,1], x_tru_UNM20[:, 2], linewidth=0.8, color="#333333")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_title("(a) NR controlled with |u|"+r"$\leq 20$", fontsize=24)
ax = fig.add_subplot(322)
ax.plot(np.arange(SIM_IDX-1), u_tru_norm_UNM20[:], linewidth=1., color="#255793")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$|u|$")
ax.set_title("(b) Norm of control inputs with |u|"+r"$\leq 20$", fontsize=24)
ax.axhline(20, ls="--", color="black", linewidth=1.)
ax.grid()
ax = fig.add_subplot(323, projection='3d')
ax.plot(x_tru_UNM30[:, 0], x_tru_UNM30[:,1], x_tru_UNM30[:, 2], linewidth=0.8, color="#333333")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_title("(c) NR controlled with |u|"+r"$\leq 30$", fontsize=24)
ax = fig.add_subplot(324)
ax.plot(np.arange(SIM_IDX-1), u_tru_norm_UNM30[:], linewidth=1., color="#255793")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$|u|$")
ax.set_title("(d) Norm of control inputs with |u|"+r"$\leq 30$", fontsize=24)
ax.axhline(30, ls="--", color="black", linewidth=1.)
ax.grid()
ax = fig.add_subplot(325, projection='3d')
ax.plot(x_tru_UNM40[:, 0], x_tru_UNM40[:,1], x_tru_UNM40[:, 2], linewidth=0.8, color="#333333")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")
ax.set_xlim(-20, 20)
ax.set_ylim(-30, 30)
ax.set_zlim(0, 50)
ax.set_title("(e) NR controlled with |u|"+r"$\leq 40$", fontsize=24)
ax = fig.add_subplot(326)
ax.plot(np.arange(SIM_IDX-1), u_tru_norm_UNM40[:], linewidth=1., color="#255793")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$|u|$")
ax.set_title("(f) Norm of control inputs with |u|"+r"$\leq 40$", fontsize=24)
ax.axhline(40, ls="--", color="black", linewidth=1.)
ax.grid()
for i in range(6) :
    fig.axes[i].axis('off')
fig.tight_layout()
plt.savefig(pwd+"/../../fig/fig08")
