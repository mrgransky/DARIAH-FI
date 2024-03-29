#!/bin/bash

#SBATCH --job-name=nikeQ
#SBATCH --output=/lustre/sgn-data/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%n_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --time=07-00:00:00
#SBATCH --mem=192G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:teslav100:1
#SBATCH --array=725-729

user="`whoami`"
stars=$(printf '%*s' 100 '')
txt="$user began Slurm job: `date`"
ch="#"
echo -e "${txt//?/$ch}\n${txt}\n${txt//?/$ch}"
echo "${stars// /*}"
echo "CPUS/NODE: $SLURM_JOB_CPUS_PER_NODE, MEM/NODE(--mem): $SLURM_MEM_PER_NODE, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "$SLURM_SUBMIT_HOST @ $SLURM_JOB_ACCOUNT, node: $SLURMD_NODENAME, CLUSTER: $SLURM_CLUSTER_NAME, Partition: $SLURM_JOB_PARTITION"
echo "JOBname: $SLURM_JOB_NAME, ID: $SLURM_JOB_ID, WRK_DIR: $SLURM_SUBMIT_DIR"
echo "nNODES: $SLURM_NNODES, NODELIST: $SLURM_JOB_NODELIST, NODE_ID: $SLURM_NODEID"
echo "nTASKS: $SLURM_NTASKS, TASKS/NODE: $SLURM_TASKS_PER_NODE, nPROCS: $SLURM_NPROCS"
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK"
echo "nTASKS/CORE: $SLURM_NTASKS_PER_CORE, nTASKS/NODE: $SLURM_NTASKS_PER_NODE"
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"

echo ">> Using $SLURM_CLUSTER_NAME conda env from Anaconda..."
source activate py39
files=(/lustre/sgn-data/Nationalbiblioteket/datasets/*.dump)
ddir="/lustre/sgn-data/Nationalbiblioteket/dataframes_xx60" #### must be adjusted ####
# maxNumFeatures=$(awk -v x="1.1e+6" 'BEGIN {printf("%d\n",x)}') # adjust values 2.2e+6
maxNumFeatures=-1

echo "Query[$SLURM_ARRAY_TASK_ID]: ${files[$SLURM_ARRAY_TASK_ID]}"
# for mx in 1.0 0.9 0.8 0.7 0.6 0.5
for mx in 1.0
do
	# for mn in 10 15 30 # 1 3 5 already done (only for max 1.0)
	for mn in 1
	do
		# ddir="/lustre/sgn-data/Nationalbiblioteket/dfXY_${mx}_max_df_${mn}_min_df" #### must be adjusted ####
		echo "max doc_freq $mx | min doc_freq $mn | maxNumFeat: $maxNumFeatures | outDIR $ddir"
		python -u user_token.py \
						--inputDF ${files[$SLURM_ARRAY_TASK_ID]} \
						--outDIR $ddir \
						--lmMethod 'stanza' \
						--qphrase 'Helsingin Pörssi ja Suomen Pankki' \
						--maxDocFreq $mx \
						--minDocFreq $mn \
						--maxNumFeat $maxNumFeatures \

	done
done

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"