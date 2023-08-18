#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH -J qXY_gpu
#SBATCH -o /scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --mem=14G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=03-00:00:00
#SBATCH --array=0-1
# # # # # # SBATCH --array=1

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
	ddir="/scratch/project_2004072/Nationalbiblioteket/dataframes_tmp" # currently df_concat only available in dataframes_tmp
	files=(/scratch/project_2004072/Nationalbiblioteket/datasets/*.dump)
fi

echo "<> Loading Q[$SLURM_ARRAY_TASK_ID] : ${files[$SLURM_ARRAY_TASK_ID]}"
python -u user_token.py --inputDF ${files[$SLURM_ARRAY_TASK_ID]} --lmMethod 'stanza' --qphrase 'Stockholms universitet'
# python -u tkRecSys.py --dsPath $ddir --lmMethod 'stanza' --qphrase 'Stockholms Universitet'
# python -u tkRecSys.py --lmMethod 'stanza' --qphrase 'Stockholms Universitet'

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"