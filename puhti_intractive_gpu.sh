#!/bin/bash

# to run this file:
# $ bash puhti_intractive_gpu.sh
# OR ...
# $ source puhti_intractive_gpu.sh


srun -J v100_gpu --account=project_2004072 --partition=gpu --gres=gpu:v100:1 --time=0-10:59:00 --mem-per-cpu=20G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
# srun -J v100_interactive --account=project_2004072 --partition=gputest --gres=gpu:v100:1 --time=0-00:15:00 --mem-per-cpu=16G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i