#!/bin/bash
srun -J gpu_interactive  --partition=gpu --gres=gpu:rtx100:1 --time=00-03:59:59 --mem=8G --pty /bin/bash -i
#module load git
#module load pytorch