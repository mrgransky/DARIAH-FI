#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

srun -J cpu_hm  --partition=grid --time=01-23:59:00 --mem=2G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i
source activate py3_gpu
