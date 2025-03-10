#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=irQ
#SBATCH --output=/scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=small
#SBATCH --time=03-00:00:00
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-14
#####SBATCH --array=367-731%10
#####SBATCH --array=0-366

set -e
set -u
set -o pipefail

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "$SLURM_SUBMIT_HOST conda virtual env from tykky module..."
echo "${stars// /*}"

NLF_LOG_FILES=(/scratch/project_2004072/Nationalbiblioteket/NLF_Pseudonymized_Logs/*.log)
DATASET_DIRECTORY="/scratch/project_2004072/Nationalbiblioteket/NLF_DATASET"

echo "Q[$SLURM_ARRAY_TASK_ID]: ${NLF_LOG_FILES[$SLURM_ARRAY_TASK_ID]}"
python -u information_retrieval.py --queryLogFile ${NLF_LOG_FILES[$SLURM_ARRAY_TASK_ID]} --dataset_dir $DATASET_DIRECTORY

done_txt="SLURM JOB ENDED @ `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"