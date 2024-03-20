#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=irQ
#SBATCH --output=/scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=13G
#SBATCH --partition=small
#SBATCH --time=03-00:00:00
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=480-740
#########SBATCH --array=641-890%10

stars=$(printf '%*s' 100 '')
txt="SLURM JOB STARTED AT: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "$SLURM_JOB_ACCOUNT, SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE, SLURM_PROFILE: $SLURM_PROFILE"
echo "HOST: $SLURM_SUBMIT_HOST, CLUSTER: $SLURM_CLUSTER_NAME, WRK_DIR: $SLURM_SUBMIT_DIR, SLURM_DISTRIBUTION: $SLURM_DISTRIBUTION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, Array_ID: $SLURM_ARRAY_JOB_ID, arrayTaskID: $SLURM_ARRAY_TASK_ID"
echo "NUM_NODES: $SLURM_JOB_NUM_NODES, SLURM_NNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID, $SLURM_JOB_PARTITION"
echo "NTASKS: $SLURM_NTASKS, PROCID: $SLURM_PROCID, SLURM_NPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU, MEM/NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_NTASKS_PER_CORE: $SLURM_NTASKS_PER_CORE, SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE, SLURM_THREADS_PER_CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"

user="`whoami`"
echo "<> Using $SLURM_CLUSTER_NAME conda env from tykky module..."
logFiles=(/scratch/project_2004072/Nationalbiblioteket/NLF_Pseudonymized_Logs/*.log)
dataset_path="/scratch/project_2004072/Nationalbiblioteket/NLF_DATASET"

echo "Q[$SLURM_ARRAY_TASK_ID]: ${logFiles[$SLURM_ARRAY_TASK_ID]}"
python -u information_retrieval.py --queryLogFile ${logFiles[$SLURM_ARRAY_TASK_ID]} --dsPath $dataset_path

done_txt="SLURM JOB ENDED @ `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
# echo "${stars// /*}"