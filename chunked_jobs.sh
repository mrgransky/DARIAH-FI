#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --job-name=pi-array-grouped
#SBATCH --output=pi-array-grouped_%a.out
#SBATCH --array=0-4

# Lets create a new folder for our output files
mkdir -p json_files

CHUNKSIZE=10
n=$SLURM_ARRAY_TASK_ID
indexes=`seq $((n*CHUNKSIZE)) $(((n + 1)*CHUNKSIZE - 1))`
echo "$indexes"
'''
for i in $indexes
do
	python pi.py --seed=$i > json_files/pi_$i.json
done
'''