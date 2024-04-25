# README

## Information of the study
* Authors  : Fumitoshi Kawasaki and Shunji Kotsuki
* Title    : Leading the Lorenz-63 system toward the prescribed regime by model predictive control coupled with data assimilation
* Year     : 2024

## Requirements
These codes are tested on Python and its major libraries (NumPy, SciPy, Matplotlib, etc.).
In addition, these codes are assumed to be executed in a Linux environment.

## Experiments in this study
You can generate data and figures used in this study by executing these codes.

## Note
* These codes are not guaranteed to output completely identical data and figures used in this study. This is because the results would depend on your execution environment and random numbers.
* We conducted a number of assimilation experiments by `run_po.py` and 1,000 CSEs by `run_mpc_xxx.py` in parallel on our high performance computers. However, the codes were modified to avoid parallel computations for generality of use. Please note that the codes would require a very huge amount of time as they are. When performing these experiments, it is recommended to modify `run_po.py` and `run_mpc_xxx.py` so that parallel computation can be performed on your server.
* If you attempt to calculate 1,000 CSEs in parallel, the number of CSEs may be under/over output due to the specifications of the codes. Please use `check_result_xxx.py` to manually check for under/over computations. If the results are insufficient, please recalculate again using `INIT_IDX_DONE_xxx.csv`, which is automatically generated in the `data` directory by `check_result_xxx.py`. If there are duplicate results, please manually eliminate them from the CSV file of the corresponding results, and then please recalculate.