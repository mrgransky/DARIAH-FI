#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH -J df_concat_stanzaTK_cBoW_su
#SBATCH -o /scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%N_%j.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02-23:59:59
# # # # SBATCH --partition=small
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48G

# # # # # # SBATCH --array=0-69
# # # # # # SBATCH -o NLF_logs/q%a_%x_%N_%j.out

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
	source activate py3_gpu
elif [ $user == 'alijanif' ]; then
	echo ">> Using Puhti conda env from tykky module..."
	dfs_dir="/scratch/project_2004072/Nationalbiblioteket/datasets"
fi

#python -u RecSys_usr_token.py --inputDF $dfs_dir/nikeX.docworks.lib.helsinki.fi_access_log.07_02_2021.log.dump --lmMethod 'stanza' --qphrase 'Stockholms universitet'
python -u tkRecSys.py --dsPath $dfs_dir --lmMethod 'stanza' --qphrase 'Stockholms Universitet'

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"