import os
import subprocess
import numpy as np

pwd = os.path.dirname(os.path.abspath(__file__))
file_name_list = ["mpc_L63_u3_umax_sp0x15.py"]
run_num = 1000
u_norm_max_list = [20, 30, 40]
penalty_param_u_list = [1000]
for file_name in file_name_list :
    for u_norm_max in u_norm_max_list :
        for penalty_param_u in penalty_param_u_list :
            for i in range(run_num) :
                command_list = ["python3", pwd+"/"+file_name, str(u_norm_max), str(penalty_param_u)]
                subprocess.run(command_list)