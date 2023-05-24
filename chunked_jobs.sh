#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH -J dfQ
#SBATCH -o /scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --partition=interactive
#SBATCH --mem-per-cpu=1G
#SBATCH --time=2-23:59:59
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
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