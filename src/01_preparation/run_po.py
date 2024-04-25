import os
import subprocess
import numpy as np

pwd = os.path.dirname(os.path.abspath(__file__))
file_name = "po.py"
member_num_list = [10, 20, 30, 40, 50, 100]
inflation_list = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.30, 1.40, 1.50, 2.00]
for member_num in member_num_list :
    for inflation in inflation_list :
        command_list = ["python3", pwd+"/"+file_name, str(member_num), str(inflation)]
        subprocess.run(command_list)
        