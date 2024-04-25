import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from matplotlib import colors
import datetime
import os
import scipy.linalg as li
from scipy import optimize
import time
import datetime
import csv
import pickle
import copy
import random
import sys
from matplotlib.colors import ListedColormap
pwd = os.path.dirname(os.path.abspath(__file__))


class Model() : 
    def __init__(self, SIM_STEP=2000) :
        self.SEED = 2024
        np.random.seed(seed=self.SEED)
        self.DT = 0.01
        self.SIM_STEP = SIM_STEP
        self.SIM_IDX = self.SIM_STEP + 1
        self.SIGMA = 10.
        self.RHO = 28.
        self.BETA = 8. / 3.
        self.MODEL_DIM = 3
        
    def lorenz63(self, x) : 
        x_dot = np.zeros((self.MODEL_DIM))
        x_dot[0] = - self.SIGMA * x[0] + self.SIGMA * x[1]
        x_dot[1] = - x[0] * x[2] + self.RHO * x[0] - x[1]
        x_dot[2] = x[0] * x[1] - self.BETA * x[2]
        return x_dot

    def runge_kutta(self, x, dt) :
        k1 = dt * self.lorenz63(x)
        k2 = dt * self.lorenz63(x+0.5*k1)
        k3 = dt * self.lorenz63(x+0.5*k2)
        k4 = dt * self.lorenz63(x+k3)
        x = x + (1. / 6.) * (k1 + 2. * k2 + 2. * k3 + k4)
        return x


class DataAssimilation(Model) :
    def __init__(self, MEMBER_NUM, INFLATION, OBS_INTERVAL=8):
        super().__init__()
        self.time_da = 0.
        self.count_da = 0
        self.OBS_DIM = self.MODEL_DIM
        self.OBS_NOISE_MEAN = 0.
        self.OBS_NOISE_STD = np.sqrt(2.)
        self.OBS_INTERVAL = OBS_INTERVAL
        self.x_nr = np.load(pwd+"/../../data/x_nr.npy")
        self.y_o = np.zeros((self.SIM_IDX, self.OBS_DIM))
        self.R = np.identity((self.OBS_DIM)) * (self.OBS_NOISE_STD**2)
        self.H = np.zeros((self.OBS_DIM, self.MODEL_DIM))
        self.INFLATION = INFLATION
        for i in range(self.OBS_DIM) :
            self.H[i, i] = 1.0
        self.MEMBER_NUM = MEMBER_NUM
        self.X_a = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.X_b = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.dX_a = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.dX_b = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MEMBER_NUM))
        self.x_a_mean = np.zeros((self.SIM_IDX, self.MODEL_DIM))
        self.x_b_mean = np.zeros((self.SIM_IDX, self.MODEL_DIM))
        self.X_a_all = np.load(pwd+"/../../data/X_a_PO_mem"+str(MEMBER_NUM)+"_inf"+"{:.2f}".format(INFLATION)+".npy")
        self.X_b_all = np.load(pwd+"/../../data/X_b_PO_mem"+str(MEMBER_NUM)+"_inf"+"{:.2f}".format(INFLATION)+".npy")
        self.P_a = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MODEL_DIM))
        self.P_b = np.zeros((self.SIM_IDX, self.MODEL_DIM, self.MODEL_DIM))


class ModelPredictiveControl(DataAssimilation) :
    def __init__(self, PRED_HORIZON_STEP=20, CNTL_HORIZON_STEP=8, MPC_INTERVAL=8, MEMBER_NUM=50, INFLATION=1.04, \
        INPUT_DIM=3, X_0_MIN=0., \
        OPT_METHOD="lm", PENALTY_PARAM_X=1e4, THRESHOLD=1e-4) :
        super().__init__(MEMBER_NUM=MEMBER_NUM, INFLATION=INFLATION)
        self.PRED_HORIZON_STEP = PRED_HORIZON_STEP
        self.PRED_HORIZON_IDX = self.PRED_HORIZON_STEP + 1
        self.CNTL_HORIZON_STEP = CNTL_HORIZON_STEP
        self.CNTL_HORIZON_IDX = self.CNTL_HORIZON_STEP + 1
        self.DTAU = self.DT
        self.MPC_INTERVAL = MPC_INTERVAL
        self.time_mpc = 0.
        self.count_mpc = 0
        self.INPUT_DIM = INPUT_DIM
        self.UNKNOWN_DIM = self.INPUT_DIM
        self.UNKNOWNVEC_DIM = self.UNKNOWN_DIM * self.CNTL_HORIZON_STEP
        self.X_0_MIN = X_0_MIN
        self.OPT_METHOD = OPT_METHOD
        self.PENALTY_PARAM_X = PENALTY_PARAM_X
        self.THRESHOLD = THRESHOLD
        self.x_opt = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM))
        self.l_opt = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM))
        self.u_opt = np.zeros((self.CNTL_HORIZON_STEP, self.INPUT_DIM))
        self.x_tru = np.zeros((self.SIM_IDX, self.MODEL_DIM))
        self.u_tru = np.zeros((self.SIM_STEP, self.INPUT_DIM))
        self.u_tru_norm = np.zeros((self.SIM_STEP))
        self.x_opt_list = []
        self.u_opt_list = []
        self.x_opt_list_list = []
        self.u_opt_list_list = []

    def run_x(self, x_0) :
        x = np.zeros((self.PRED_HORIZON_IDX, self.MODEL_DIM))
        x[0, :] = x_0
        for t in range(self.PRED_HORIZON_STEP) :
            x[t+1, :] = self.runge_kutta(x[t, :], self.DT)
        return x

system = ModelPredictiveControl()
OPT_POINT = 29
OPT_TIME = OPT_POINT * system.OBS_INTERVAL
ITERATION_INTERVAL = 1
X_MARKER_SIZE = 6
X_STAR_MARKER_SIZE = 4
x_tru = np.load(pwd+"/../../data/x_tru_u3.npy")
y_o = np.load(pwd+"/../../data/y_o_u3.npy")
X_a_w = np.load(pwd+"/../../data/X_a_u3.npy")
X_b_w = np.load(pwd+"/../../data/X_b_u3.npy")
u_tru_norm = np.load(pwd+"/../../data/u_tru_norm_u3.npy")
with open(pwd+"/../../data/x_opt_list_list_u3", mode="br") as f :
    x_opt_list_list = pickle.load(f)
with open(pwd+"/../../data/u_opt_list_list_u3", mode="br") as f :
    u_opt_list_list = pickle.load(f)
x_tru_w = x_tru[OPT_TIME:OPT_TIME+system.PRED_HORIZON_IDX]
x_tru_wo = system.run_x(x_tru[OPT_TIME])
x_b = system.run_x(x_opt_list_list[OPT_POINT][::ITERATION_INTERVAL][0][0, :])
cmap_reds = ListedColormap(plt.cm.YlOrRd(np.linspace(0., 1.0, 256)))
cmap_blues = ListedColormap(plt.cm.YlGnBu(np.linspace(0.15, 1., 256)))
plt.rcParams["font.size"] = 18

fig, ax = plt.subplots(4, 1, figsize=(12, 15), facecolor="w", gridspec_kw={"width_ratios": [1], "height_ratios": [1, 0.5, 0.5, 0.5]}, dpi=600)
x_opt_list = x_opt_list_list[OPT_POINT]
it_num = len(x_opt_list[::ITERATION_INTERVAL])
cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_reds), norm=plt.Normalize(vmin=1, vmax=it_num))
for k, i in enumerate(range(it_num)) :
    if i == it_num-1 :
        ax[0].plot(np.arange(system.PRED_HORIZON_IDX), x_opt_list[::ITERATION_INTERVAL][i][:, 0], linewidth=2.0, \
                    label="Controlled prediction", color=cmap.to_rgba(k+1), marker="^", markersize=X_MARKER_SIZE, markevery=(1, 1), zorder=998)
    else :
        ax[0].plot(np.arange(system.PRED_HORIZON_IDX), x_opt_list[::ITERATION_INTERVAL][i][:, 0], linewidth=0.8, \
                    color=cmap.to_rgba(k+1), linestyle="--", marker="^", markersize=X_STAR_MARKER_SIZE, alpha=float(k+1)/it_num, markevery=(1, 1))
ax[0].plot(np.arange(system.PRED_HORIZON_IDX), x_tru_w[:, 0], linewidth=2.0, label="Controlled NR", color="#333333", marker="o", markersize=X_MARKER_SIZE, zorder=1000)
ax[0].plot(np.arange(system.PRED_HORIZON_IDX), x_tru_wo[:, 0], linewidth=2.0, label="NR", color="#8C8C8C", marker="o", markersize=X_MARKER_SIZE, zorder=900, linestyle="--")
ax[0].plot(np.arange(system.PRED_HORIZON_IDX), x_b[:, 0], linewidth=2.0, label="Forecast from initial state", color="#137C95", marker="o", markersize=X_MARKER_SIZE, zorder=500, linestyle="--")
ax[0].scatter(0, x_b[0, 0], s=np.pi*(X_MARKER_SIZE/2)**2+25, color="#CC3399", edgecolor="#A02878", label="Analysis as initial state", zorder=999, linewidths=1.)
ax[0].scatter(0, y_o[OPT_TIME, 0], s=np.pi*(X_MARKER_SIZE/2)**2+100, color="#7C9C08", edgecolor="#5C7406", label="Observation", zorder=700, linewidths=1., marker="*")
ax[0].scatter(np.zeros((system.MEMBER_NUM)), X_a_w[OPT_TIME, 0, :], s=np.pi*(X_MARKER_SIZE/2)**2+10, color="#E9A9D4", edgecolor="#A02878", linewidths=0.5)
ax[0].set_ylabel(r'$x^{*}$') 
ax[0].set_xlabel(r'$\tau$')
ax[0].set_xticks(np.arange(system.PRED_HORIZON_IDX))
ax[0].set_ylim(-2.5, 9.8)
ax[0].axhline(system.X_0_MIN, ls="--", color="black", linewidth=1.)
ax[0].grid()
ax[0].set_title("(a) ")
ax[0].legend(fontsize=16, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5)
ax_add = ax[0].twinx()
ax_add.set_ylabel(r'$x$')
ax_add.set_ylim(-2.5, 9.8)
u_opt_list = u_opt_list_list[OPT_POINT]
cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_blues), norm=plt.Normalize(vmin=1, vmax=it_num))
for k, i in enumerate(range(it_num)) :
    if i == it_num-1 :
        ax[1].plot(np.arange(system.CNTL_HORIZON_STEP), u_opt_list[::ITERATION_INTERVAL][i][:, 0], linewidth=2.0, label="After iterations", color=cmap.to_rgba(k+1), marker="^", markersize=X_MARKER_SIZE)
        ax[2].plot(np.arange(system.CNTL_HORIZON_STEP), u_opt_list[::ITERATION_INTERVAL][i][:, 1], linewidth=2.0, label="After iterations", color=cmap.to_rgba(k+1), marker="^", markersize=X_MARKER_SIZE)
        ax[3].plot(np.arange(system.CNTL_HORIZON_STEP), u_opt_list[::ITERATION_INTERVAL][i][:, 2], linewidth=2.0, label="After iterations", color=cmap.to_rgba(k+1), marker="^", markersize=X_MARKER_SIZE)
    else :
        ax[1].plot(np.arange(system.CNTL_HORIZON_STEP), u_opt_list[::ITERATION_INTERVAL][i][:, 0], linewidth=0.8, color=cmap.to_rgba(k+1), linestyle="--", marker="^", markersize=X_STAR_MARKER_SIZE, alpha=float(k+1)/it_num)
        ax[2].plot(np.arange(system.CNTL_HORIZON_STEP), u_opt_list[::ITERATION_INTERVAL][i][:, 1], linewidth=0.8, color=cmap.to_rgba(k+1), linestyle="--", marker="^", markersize=X_STAR_MARKER_SIZE, alpha=float(k+1)/it_num)
        ax[3].plot(np.arange(system.CNTL_HORIZON_STEP), u_opt_list[::ITERATION_INTERVAL][i][:, 2], linewidth=0.8, color=cmap.to_rgba(k+1), linestyle="--", marker="^", markersize=X_STAR_MARKER_SIZE, alpha=float(k+1)/it_num)
ax[1].set_ylabel(r'$u_{x}^{*}$')
ax[1].set_xlabel(r'$\tau$')
ax[1].grid()
ax[1].set_title("(b)")
ax[1].legend(fontsize=16, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5)
ax[1].set_xticks(np.arange(8+1)-0.5, np.arange(8+1))
ax[1].set_xlim(0.-0.5, 8-0.5)
ax[1].set_ylim(-2, 45)
ax[2].set_ylabel(r'$u_{y}^{*}$')
ax[2].set_xlabel(r'$\tau$')
ax[2].grid()
ax[2].set_title("(c)")
ax[2].set_ylim(-2, 45)
ax[2].legend(fontsize=16, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5)
ax[2].set_xticks(np.arange(8+1)-0.5, np.arange(8+1))
ax[2].set_xlim(0.-0.5, 8-0.5)
ax[3].set_ylabel(r'$u_{z}^{*}$')
ax[3].set_xlabel(r'$\tau$')
ax[3].grid()
ax[3].set_title("(d)")
ax[3].legend(fontsize=16, bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0.5)
ax[3].set_xticks(np.arange(8+1)-0.5, np.arange(8+1))
ax[3].set_xlim(0.-0.5, 8-0.5)
cbar_ax_red = fig.add_axes([1.04, 0.697, 0.018, 0.266])
cbar_ax_blue = fig.add_axes([1.04, 0.058, 0.018, 0.560])
cbar_red = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_reds), cax=cbar_ax_red, orientation='vertical', ticks=[])
cbar_red.set_ticks([0, 1])
cbar_red.set_ticklabels([1, len(x_opt_list[::ITERATION_INTERVAL])])
cbar_red.ax.tick_params(labelsize=18)
cbar_red.set_label("The number of iteration", size=18)
cbar_blue = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_blues), cax=cbar_ax_blue, orientation='vertical')
cbar_blue.set_ticks([0, 1])
cbar_blue.set_ticklabels([1, len(x_opt_list[::ITERATION_INTERVAL])])
cbar_blue.ax.tick_params(labelsize=18)
cbar_blue.set_label("The number of iteration", size=18)
fig.tight_layout()
plt.savefig(pwd+"/../../fig/fig05", bbox_inches='tight')
