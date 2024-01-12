#!/bin/bash

# to run this file:
# $ bash puhti_intractive_gpu.sh
# OR ...
# $ source puhti_intractive_gpu.sh

# srun -J v100_gpu --account=project_2004072 --partition=gpu --gres=gpu:v100:1 --time=0-01:59:00 --mem=116G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
srun -J gpu_interactive --account=project_2004072 --partition=gpu --gres=gpu:v100:1 --time=0-10:15:00 --mem=64G --ntasks=1 --cpus-per-task=4 --pty /bin/bash -i

module load git