#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

# Image_Retrieval_TUNI
# srun -J intrtv_cpu --account=project_2004072 --partition=interactive --time=02-23:59:00 --mem=64G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
srun -J intrtv_cpu --account=project_2004072 --partition=small --time=00-23:59:00 --mem=32G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
# srun -J small_cpu --account=project_2004072 --partition=large --time=2-23:59:00 --mem=128G --ntasks=1 --cpus-per-task=2 --pty /bin/bash -i

module load git