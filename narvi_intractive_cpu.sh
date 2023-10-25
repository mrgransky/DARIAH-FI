#!/bin/bash

# to run this file:
# $ bash puhti_intractive_cpu.sh
# OR ...
# $ source puhti_intractive_cpu.sh

srun -J cpu_intractive  --partition= --time=00-23:59:59 --mem=32G --ntasks=1 --cpus-per-task=1 --pty /bin/bash -i