#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

# Image_Retrieval_TUNI
# srun -J intrtv_cpu --account=project_2004072 --partition=hugemem --time=00-05:15:00 --mem=999G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
srun -J intrtv_cpu --account=project_2004072 --partition=interactive --time=00-01:00:00 --mem=73G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
# srun -J cpu_large_interactive --account=project_2004072 --partition=large --time=00-23:59:00 --mem=286G --ntasks=1 --cpus-per-task=2 --pty /bin/bash -i

module load git