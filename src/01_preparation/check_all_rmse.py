import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
pwd = os.path.dirname(os.path.abspath(__file__))


inf_list = ["1.00", "1.02", "1.04", "1.06", "1.08", "1.10", "1.12", "1.14", "1.16", "1.18", "1.20", "1.30", "1.40", "1.50", "2.00"]
mem_list = ["10", "20", "30", "40", "50", "100"]
rmse_arr = np.zeros((len(mem_list), len(inf_list)))
for i, inf in enumerate(inf_list) :
    for j, mem in enumerate(mem_list) :
        rmse_arr[j, i] = np.load(pwd+"/../../data/rmse_mem"+mem+"_inf"+inf+".npy")
plt.rcParams["font.size"] = 16
fig, ax = plt.subplots(1, 1, figsize=(20, 6))
im1 = ax.pcolormesh(rmse_arr, cmap="Purples_r", edgecolors='k', linewidth=1, antialiased=True, vmin=0.27, vmax=0.3)
for i, inf in enumerate(inf_list) :
    for j, mem in enumerate(mem_list) :
        ax.text(i+0.5, j+0.46, f'{rmse_arr[j, i]:#.5g}', ha='center', va='center', color='black', fontsize=13)
fig.colorbar(im1, ax=ax)
ax.set_title('RMSE', fontsize=20)
ax.set_xlabel('inf.', fontsize=18)
ax.set_ylabel('mem.', fontsize=18)
ax.set_xticks(np.arange(0.5, len(inf_list)+0.5, 1), inf_list)
ax.set_yticks(np.arange(0.5, len(mem_list)+0.5, 1), mem_list)
plt.savefig(pwd+"/../../fig/check_all_rmse")
