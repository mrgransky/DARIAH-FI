#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH -J df_concat_XY
#SBATCH -o /scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=370G
#SBATCH --partition=longrun
#SBATCH --time=14-00:00:00
# # # # # # SBATCH --gres=gpu:v100:1
# # # # # # SBATCH --array=22-69 ############## PAY ATTENTION TO RUN user_token.py ############## 

stars=$(printf '%*s' 100 '')
txt="SLURM JOB STARTED @ `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE"
echo "HOST: $SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "nTASKS/CORE: $SLURM_NTASKS_PER_CORE, nTASKS/NODE: $SLURM_NTASKS_PER_NODE"
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"

user="`whoami`"

if [ $user == 'alijani' ]; then
	echo ">> Using Narvi conda env from Anaconda..."
	source activate py39
	ddir="/lustre/sgn-data/Nationalbiblioteket/dataframes" # currently nothing available!
	files=(/lustre/sgn-data/Nationalbiblioteket/datasets/*.dump)
elif [ $user == 'alijanif' ]; then
	echo ">> Using $SLURM_CLUSTER_NAME conda env from tykky module..."
	ddir="/scratch/project_2004072/Nationalbiblioteket/dataframes_XY" # currently df_concat only available in dataframes_tmp only to run tkRecSys.py
	files=(/scratch/project_2004072/Nationalbiblioteket/datasets/*.dump)
fi

# echo "Query[$SLURM_ARRAY_TASK_ID]: ${files[$SLURM_ARRAY_TASK_ID]}"
# python -u user_token.py --inputDF ${files[$SLURM_ARRAY_TASK_ID]} --outDIR $ddir --lmMethod 'stanza' --qphrase 'Helsingin Pörssi ja Suomen Pankki'

python -u tkRecSys.py --dsPath $ddir --lmMethod 'stanza' --qphrase 'Helsingin Pörssi ja Suomen Pankki'

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"