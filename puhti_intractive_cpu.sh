#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

# Image_Retrieval_TUNI
#srun -J cpu_350gb --account=project_2004072 --partition=hugemem --time=1-10:59:00 --mem=999G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
srun -J intrtv_cpu_600GB --account=project_2004072 --partition=small --time=0-23:59:00 --mem=26G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i

# APP_CSC
#srun -J v100_intrc --account=Project_2004160 --partition=hugemem_longrun --time=5-23:59:00 --mem=512G --ntasks=1 --pty /bin/bash -i

module load git