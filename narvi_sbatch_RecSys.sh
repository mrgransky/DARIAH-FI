#!/bin/bash

#SBATCH -J nkQ
#SBATCH -o /lustre/sgn-data/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=07-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --mem=302G
#SBATCH --array=0-29

stars=$(printf '%*s' 100 '')
txt="SLURM JOB STARTED @ `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "$SLURM_JOB_ACCOUNT, SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE, SLURM_PROFILE: $SLURM_PROFILE"
echo "HOST: $SLURM_SUBMIT_HOST, CLUSTER: $SLURM_CLUSTER_NAME, WRK_DIR: $SLURM_SUBMIT_DIR, SLURM_DISTRIBUTION: $SLURM_DISTRIBUTION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, Array_ID: $SLURM_ARRAY_JOB_ID"
echo "NUM_NODES: $SLURM_JOB_NUM_NODES, SLURM_NNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID, $SLURM_JOB_PARTITION"
echo "NTASKS: $SLURM_NTASKS, PROCID: $SLURM_PROCID, SLURM_NPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU, MEM/NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_NTASKS_PER_CORE: $SLURM_NTASKS_PER_CORE, SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE, SLURM_THREADS_PER_CORE: $SLURM_THREADS_PER_CORE"
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

echo "Q[$SLURM_ARRAY_TASK_ID]: ${files[$SLURM_ARRAY_TASK_ID]}"
python -u user_token.py --inputDF ${files[$SLURM_ARRAY_TASK_ID]} --outDIR $ddir --lmMethod 'stanza' --qphrase 'Stockholms universitet'
# python -u tkRecSys.py --dsPath $ddir --lmMethod 'stanza' --qphrase 'Stockholms Universitet'
# python -u tkRecSys.py --lmMethod 'stanza' --qphrase 'Stockholms Universitet'

done_txt="SLURM JOB ENDED @ `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"