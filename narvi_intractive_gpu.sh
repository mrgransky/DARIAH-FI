#!/bin/bash
srun -J gpu_interactive  --partition=gpu --gres=gpu:teslap100:1 --time=02-23:59:59 --mem=48--pty /bin/bash -i
#module load git
#module load pytorch