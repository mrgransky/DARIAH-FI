#!/bin/bash

#SBATCH --job-name=nikeQ_XY
#SBATCH --output=/lustre/sgn-data/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%n_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=07-00:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=656G
# # # # # SBATCH --gres=gpu:teslav100:1
#SBATCH --array=730-731

stars=$(printf '%*s' 100 '')
txt="SLURM JOB STARTED @ `date`"
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

user="`whoami`"

if [ $user == 'alijani' ]; then
	echo ">> Using $SLURM_CLUSTER_NAME conda env from Anaconda..."
	source activate py39
	files=(/lustre/sgn-data/Nationalbiblioteket/datasets/*.dump)
	ddir="/lustre/sgn-data/Nationalbiblioteket/dataframes"
elif [ $user == 'alijanif' ]; then
	echo ">> Using $SLURM_CLUSTER_NAME conda env from tykky module..."
	ddir="/scratch/project_2004072/Nationalbiblioteket/dataframes_$SLURM_JOB_PARTITION"
	files=(/scratch/project_2004072/Nationalbiblioteket/datasets/*.dump)
fi

echo "Query[$SLURM_ARRAY_TASK_ID]: ${files[$SLURM_ARRAY_TASK_ID]}"
python -u user_token.py --inputDF ${files[$SLURM_ARRAY_TASK_ID]} --outDIR $ddir --lmMethod 'stanza' --qphrase 'Åbo Akademi'
# python -u tkRecSys.py --dsPath $ddir --lmMethod 'stanza' --qphrase 'Stockholms Universitet'
# python -u tkRecSys.py --lmMethod 'stanza' --qphrase 'Stockholms Universitet'

done_txt="SLURM JOB ENDED @ `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"