#!/bin/bash

#SBATCH -J nike_Q
#SBATCH -o /lustre/sgn-data/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=06-23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:teslav100:1
#SBATCH --mem=128G
#SBATCH --array=0-259

stars=$(printf '%*s' 90 '')
txt="SLURM JOB STARTED AT: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "HOST: $SLURM_SUBMIT_HOST, CLUSTER: $SLURM_CLUSTER_NAME, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "JOB: name: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, Array_ID: $SLURM_ARRAY_JOB_ID"
echo "NUM_NODES: $SLURM_JOB_NUM_NODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "NTASKS: $SLURM_NTASKS, PROCID: $SLURM_PROCID"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "${stars// /*}"

user="`whoami`"
cluster="$SLURM_CLUSTER_NAME"
echo "Cluster: $cluster Current User: $user"

if [ $user == 'alijani' ]; then
	echo ">> Using Narvi conda env from Anaconda..."
	source activate py39
	files=(/lustre/sgn-data/Nationalbiblioteket/datasets/*.dump)
elif [ $user == 'alijanif' ]; then
	echo ">> Using Puhti conda env from tykky module..."
	dfs_dir="/scratch/project_2004072/Nationalbiblioteket/datasets"
	files=(/scratch/project_2004072/Nationalbiblioteket/datasets/*.dump)
fi

echo "<> Loading Q[$SLURM_ARRAY_TASK_ID] : ${files[$SLURM_ARRAY_TASK_ID]}"
python -u user_token.py --inputDF ${files[$SLURM_ARRAY_TASK_ID]} --lmMethod 'stanza' --qphrase 'Stockholms universitet'
# python -u tkRecSys.py --dsPath $dfs_dir --lmMethod 'stanza' --qphrase 'Stockholms Universitet'

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"