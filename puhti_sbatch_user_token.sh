#!/bin/bash

#SBATCH --account=project_2004072
#SBATCH --job-name=nikeQ_XY
#SBATCH --output=/scratch/project_2004072/Nationalbiblioteket/trash/NLF_logs/%x_%a_%N_%j_%A.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --array=275,276

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
echo "CPUS_ON_NODE: $SLURM_CPUS_ON_NODE, CPUS/TASK: $SLURM_CPUS_PER_TASK, MEM/CPU: $SLURM_MEM_PER_CPU"
echo "nTASKS/CORE: $SLURM_NTASKS_PER_CORE, nTASKS/NODE: $SLURM_NTASKS_PER_NODE"
echo "THREADS/CORE: $SLURM_THREADS_PER_CORE"
echo "${stars// /*}"

echo "$SLURM_CLUSTER_NAME conda env from tykky module..."
files=(/scratch/project_2004072/Nationalbiblioteket/datasets/*.dump)
ddir="/scratch/project_2004072/Nationalbiblioteket/dataframes_XY"

echo "Query[$SLURM_ARRAY_TASK_ID]: ${files[$SLURM_ARRAY_TASK_ID]}"

# for mx in 0.95 0.85 0.75 0.55
for mx in 1.0
do
	# for mn in 50 25 20 10 3
	for mn in 1
	do
		# ddir="/scratch/project_2004072/Nationalbiblioteket/dfXY_${mx}_max_df_${mn}_min_df"
		echo "max doc_freq $mx | min doc_freq $mn | outDIR $ddir"
		python -u user_token.py \
						--inputDF ${files[$SLURM_ARRAY_TASK_ID]} \
						--outDIR $ddir \
						--lmMethod 'stanza' \
						--qphrase 'Helsingin Pörssi ja Suomen Pankki' \
						--maxDocFreq $mx \
						--minDocFreq $mn
	done
done

# ### dependent ###
# # for mn in 1 3 5 7 10
# for mn in 1
# do
# 	# mx=$(echo "scale=2; $mn/10" | bc)
# 	mx=1.0
# 	# ddir="/scratch/project_2004072/Nationalbiblioteket/dfXY_${mx}_max_df_${mn}_min_df"
# 	echo "max doc_freq $mx | min doc_freq $mn | outDIR $ddir"
# 	python -u user_token.py \
# 					--inputDF ${files[$SLURM_ARRAY_TASK_ID]} \
# 					--outDIR $ddir \
# 					--lmMethod 'stanza' \
# 					--qphrase 'Helsingin Pörssi ja Suomen Pankki' \
# 					--maxDocFreq $mx \
# 					--minDocFreq $mn
# done

done_txt="$user finished Slurm job: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"