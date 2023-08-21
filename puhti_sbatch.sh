#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH -J dfQ
#SBATCH -o /scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --partition=small
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03-00:00:00
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=649-729%6
# # # # array: 730-1095 -> useless, no valuable information

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
	source activate py39
	logFiles=(/scratch/project_2004072/Nationalbiblioteket/NLF_Pseudonymized_Logs/*.log) #TODO: must be adjusted!
elif [ $user == 'alijanif' ]; then
	echo ">> Using Puhti Conda Environment..."
	logFiles=(/scratch/project_2004072/Nationalbiblioteket/NLF_Pseudonymized_Logs/*.log)
fi

echo "<> Loading Q[$SLURM_ARRAY_TASK_ID] : ${logFiles[$SLURM_ARRAY_TASK_ID]}"
python -u information_retrieval.py --saveDF True --queryLogFile ${logFiles[$SLURM_ARRAY_TASK_ID]}

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"