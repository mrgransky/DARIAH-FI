#!/bin/bash

#SBATCH --job-name=df_concat
#SBATCH --output=/lustre/sgn-data/Nationalbiblioteket/trash/NLF_logs/%x_%N_%n_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=07-00:00:00
#SBATCH --mem=750G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --nodes=1

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "$SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, node: $SLURMD_NODENAME, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION, $SLURM_JOB_GPUS"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "nTASKS/CORE: $SLURM_NTASKS_PER_CORE, nTASKS/NODE: $SLURM_NTASKS_PER_NODE"
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"

echo "$SLURM_CLUSTER_NAME conda env from Anaconda..."
source activate py39
dfsDIR="/lustre/sgn-data/Nationalbiblioteket/dataframes" ########## must be adjusted! ##########

python -u concat_dfs.py --dfsPath $dfsDIR --lmMethod 'stanza' --qphrase 'Helsingin Pörssi ja Suomen Pankki'

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"