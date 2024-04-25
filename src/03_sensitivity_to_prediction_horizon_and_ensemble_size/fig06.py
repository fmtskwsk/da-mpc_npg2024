import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams["font.size"] = 20
pwd = os.path.dirname(os.path.abspath(__file__))


DT = 0.01
PRED_LIST = [10, 20, 30, 40, 50]
MEM_LIST = [10, 20, 30, 40, 50, 100]
RESULT_LIST = []
for p in PRED_LIST :
    RESULT_LIST.append(np.load(pwd+"/../../data/result_pred"+str(p)+"_u3_sp0x15.npy"))

success_rate = np.zeros((len(PRED_LIST), len(MEM_LIST)))
sum_x = np.zeros((len(PRED_LIST), len(MEM_LIST)))
sum_u = np.zeros((len(PRED_LIST), len(MEM_LIST)))
for i in range(len(PRED_LIST)) :
    for j in range(len(RESULT_LIST[i])) :
        success_rate[i, j] = np.average(RESULT_LIST[i][j][:, 0])
        sum_x[i, j] = np.average(RESULT_LIST[i][j][:, 2])
        sum_u[i, j] = np.average(RESULT_LIST[i][j][:, 4])
fig, ax = plt.subplots(1, 3, figsize=(28, 6.8), dpi=600)
cmap_oranges = ListedColormap(plt.cm.Oranges(np.linspace(0.05, 0.6, 256)))
cmap_reds = ListedColormap(plt.cm.Reds(np.linspace(0.05, 0.6, 256)))
cmap_blues = ListedColormap(plt.cm.Blues_r(np.linspace(0.4, 1., 256)))

im0 = ax[0].pcolormesh(success_rate, cmap=cmap_oranges, edgecolors='k', linewidth=1, antialiased=True, vmin=0.5, vmax=1.)
for i in range(len(PRED_LIST)):
    for j in range(len(MEM_LIST)):
        ax[0].text(j+0.5, i+0.46, f'{success_rate[i, j]:.3f}', ha='center', va='center', color='black', fontsize=16)
fig.colorbar(im0, ax=ax[0])
ax[0].set_title('(a) Success rate (SR)', fontsize=24)
ax[0].set_xlabel('The number of ensemble members '+r"$m$", fontsize=20)
ax[0].set_ylabel('Prediction horizon '+r"$T_p$", fontsize=20)
ax[0].set_xticks(np.arange(0.5, len(MEM_LIST)+0.5, 1), MEM_LIST)
ax[0].set_yticks(np.arange(0.5, len(PRED_LIST)+0.5, 1), PRED_LIST)
im1 = ax[1].pcolormesh(sum_x*DT, cmap=cmap_reds, edgecolors='k', linewidth=1, antialiased=True, vmin=-1.*DT, vmax=0.)
for i in range(len(PRED_LIST)):
    for j in range(len(MEM_LIST)):
        ax[1].text(j+0.5, i+0.46, f'{sum_x[i, j]*DT:#.1e}', ha='center', va='center', color='black', fontsize=15)
fig.colorbar(im1, ax=ax[1])
ax[1].set_title('(b) Mean total failure (MTF)', fontsize=24)
ax[1].set_xlabel('The number of ensemble members '+r"$m$", fontsize=20)
ax[1].set_ylabel('Prediction horizon '+r"$T_p$", fontsize=20)
ax[1].set_xticks(np.arange(0.5, len(MEM_LIST)+0.5, 1), MEM_LIST)
ax[1].set_yticks(np.arange(0.5, len(PRED_LIST)+0.5, 1), PRED_LIST)
im2 = ax[2].pcolormesh(sum_u*DT, cmap=cmap_blues, edgecolors='k', linewidth=1, antialiased=True, vmin=9000*DT, vmax=12000*DT)
for i in range(len(PRED_LIST)):
    for j in range(len(MEM_LIST)):
        ax[2].text(j+0.52, i+0.46, f'{sum_u[i, j]*DT:.1f}', ha='center', va='center', color='black', fontsize=16)
fig.colorbar(im2, ax=ax[2])
ax[2].set_title('(c) Mean total control inputs (MTC)', fontsize=24)
ax[2].set_xlabel('The number of ensemble members '+r"$m$", fontsize=20)
ax[2].set_ylabel('Prediction horizon '+r"$T_p$", fontsize=20)
ax[2].set_xticks(np.arange(0.5, len(MEM_LIST)+0.5, 1), MEM_LIST)
ax[2].set_yticks(np.arange(0.5, len(PRED_LIST)+0.5, 1), PRED_LIST)
fig.tight_layout()
plt.savefig(pwd+"/../../fig/fig06")