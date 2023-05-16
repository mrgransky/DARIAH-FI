#!/bin/bash

# to run this file:
# $ bash puhti_intractive_gpu.sh
# OR ...
# $ source puhti_intractive_gpu.sh


# Image_Retrieval_TUNI

#srun -J v100_32gb --account=project_2004072 --partition=gpu --gres=gpu:v100:1 --time=0-10:59:00 --mem-per-cpu=22G --ntasks=1 --cpus-per-task=1 nvidia-smi
srun -J v100_interactive --account=project_2004072 --partition=gpu --gres=gpu:v100:1 --time=0-03:59:00 --mem-per-cpu=64G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i

# APP_CSC
#srun -J v100_intrc --account=Project_2004160 --partition=gpu --gres=gpu:v100:1 --time=1-23:59:00 --mem=128G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i

#module load git
#module load pytorch