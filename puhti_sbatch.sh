#!/bin/bash
#SBATCH --account=project_2004072
#SBATCH -J df_update_usr_referer_ts
#SBATCH -o NLF_logs/q%a_%x_%N_%j.out
#SBATCH --partition=hugemem_longrun
#SBATCH --mem-per-cpu=354G
#SBATCH --time=13-23:59:59
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
# # # # # # SBATCH --array=0-69
# # # # # # SBATCH --mem=8G # fatal: --mem, --mem-per-cpu, and --mem-per-gpu are mutually exclusive

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
	echo ">> Using Puhti Conda Environment..."
	#source /projappl/project_2004072/miniconda3/bin/activate py3_gpu
fi

#python nationalbiblioteket.py
python nationalbiblioteket_logs.py --query $SLURM_ARRAY_TASK_ID
#python nationalbiblioteket_logs.py

done_txt="SLURM JOB ENDED AT: `date`"
echo -e "${done_txt//?/$ch}\n${done_txt}\n${done_txt//?/$ch}"
echo "${stars// /*}"