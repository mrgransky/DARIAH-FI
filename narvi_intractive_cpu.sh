#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

srun -J cpu_intractive  --partition=small --time=00-23:59:59 --mem=256G --pty /bin/bash -i