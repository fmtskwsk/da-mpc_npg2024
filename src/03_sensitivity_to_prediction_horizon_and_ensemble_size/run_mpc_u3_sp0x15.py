import os
import subprocess
import numpy as np

pwd = os.path.dirname(os.path.abspath(__file__))
file_name_list = ["mpc_L63_u3_sp0x15.py"]
run_num = 1000
pred_horizon_step_list = [10, 20, 30, 40, 50]
cntl_horizon_step_list = [8]
member_num_list = [10, 20, 30, 40, 50, 100]
men_inf_dict = {10: 1.50, 20: 1.18, 30: 1.08, 40: 1.06, 50: 1.04, 100: 1.02}
for file_name in file_name_list :
    for pred_horizon_step in pred_horizon_step_list :
        for cntl_horizon_step in cntl_horizon_step_list :
            for member_num in member_num_list :
                for i in range(run_num) :
                    command_list = ["python3", pwd+"/"+file_name, str(pred_horizon_step), str(cntl_horizon_step), str(member_num), str(men_inf_dict[member_num])]
                    subprocess.run(command_list)