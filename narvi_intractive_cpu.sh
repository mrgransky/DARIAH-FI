#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

srun -J cpu_grid  --partition=grid --time=01-23:59:00 --mem=256G --pty /bin/bash -i