#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH -J dfQ_
#SBATCH -o /scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/q%a_%x_%N_%j%A.out
#SBATCH --partition=large
#SBATCH --mem=16G
#SBATCH --time=2-23:59:59
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-1000
# # # # array: 0-1096
stars=$(printf '%*s' 90 '')
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
	source activate py3_gpu
elif [ $user == 'alijanif' ]; then
	echo ">> Using Puhti Conda Environment..."
	#source /projappl/project_2004072/miniconda3/bin/activate py3_gpu
fi

python -u nationalbiblioteket_logs.py --saveDF True --query $SLURM_ARRAY_TASK_ID

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"