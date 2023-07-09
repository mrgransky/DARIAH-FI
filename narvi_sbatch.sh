#!/bin/bash

#SBATCH -J irQ
#SBATCH -o /lustre/sgn-data/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=6-23:59:59
#SBATCH --partition=normal
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=2
# # # # # # # # SBATCH --array=0-1097%5

stars=$(printf '%*s' 110 '')
txt="SLURM JOB STARTED AT: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "HOST: $SLURM_SUBMIT_HOST, CLUSTER: $SLURM_CLUSTER_NAME, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "JOB: name: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID"
echo "NUM_NODES: $SLURM_JOB_NUM_NODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "NTASKS: $SLURM_NTASKS, PROCID: $SLURM_PROCID"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "${stars// /*}"

user="`whoami`"
echo "Cluster: $SLURM_CLUSTER_NAME | $user | arrayTaskID: $SLURM_ARRAY_TASK_ID | arrayJobID: $SLURM_ARRAY_JOB_ID"

if [ $user == 'alijani' ]; then
	echo ">> Using Narvi conda env from Anaconda..."
	source activate py39
elif [ $user == 'alijanif' ]; then
	echo ">> Using Puhti Conda Environment..."
fi

python -u information_retrieval.py --saveDF True --query $SLURM_ARRAY_TASK_ID

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
# echo "${stars// /*}"